"""Pinned CPU-only NLI inference over exact, previously validated source quotes.

This is a *measured experiment*, not a calibrated factuality or safety gate.
The gold labels are never supplied to the model. No model is downloaded by default.
"""

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from aql.claim_evaluation import ClaimPrediction, LockedClaims

MODEL_REPO = "cross-encoder/nli-MiniLM2-L6-H768"
MODEL_REVISION = "9aa22b250d25a10ac39ab4a6b2bc28213507cb0c"
MODEL_FILE = "onnx/model.onnx"
MODEL_KEY = f"{MODEL_REPO}@{MODEL_REVISION}/{MODEL_FILE}"
EXPECTED_LABELS = {0: "contradiction", 1: "entailment", 2: "neutral"}
MAX_PAIR_TOKENS = 256


@dataclass(frozen=True)
class NliObservation:
    case_id: str
    nli_label: str
    token_count: int
    logits: tuple[float, float, float]

    @property
    def prediction(self) -> ClaimPrediction:
        return ClaimPrediction(
            case_id=self.case_id,
            label="entailed" if self.nli_label == "entailment" else "not_entailed",
        )

    def as_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "nli_label": self.nli_label,
            "token_count": self.token_count,
            "logits": [round(value, 6) for value in self.logits],
        }


def validate_model_labels(config: dict) -> None:
    """Reject revisions with a different output-label meaning or model family."""
    labels = config.get("id2label")
    if (
        config.get("model_type") != "roberta"
        or config.get("pad_token_id") != 1
        or labels != {str(key): value for key, value in EXPECTED_LABELS.items()}
    ):
        raise ValueError("Pinned NLI model configuration is incompatible with the evaluation")


def predicted_nli_label(logits: Sequence[float]) -> str:
    """Take the highest raw logit; no test-label-based tuning or confidence threshold."""
    if len(logits) != len(EXPECTED_LABELS):
        raise ValueError("NLI model must output exactly three class scores")
    values = tuple(float(value) for value in logits)
    if any(not math.isfinite(value) for value in values):
        raise ValueError("NLI logits must be finite")
    return EXPECTED_LABELS[max(range(len(values)), key=lambda index: values[index])]


def premise_for_claim(claims: LockedClaims, case_id: str) -> str:
    """Use quoted source passages, not retrieval text, labels or human rationales."""
    anchors = claims.anchors_by_claim[case_id]
    if not anchors:
        raise ValueError(f"{case_id} has no verified source quote")
    return "\n".join(anchor.span.quote for anchor in anchors)


class PinnedNliCrossEncoder:
    """Load the frozen HF snapshot with ONNX Runtime only (no Torch/remote code)."""

    def __init__(self, *, allow_download: bool = False, cache_dir: str | None = None) -> None:
        try:
            import onnxruntime as ort
            from huggingface_hub import snapshot_download
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Install the optional ONNX dependencies: uv sync --extra semantic"
            ) from exc

        local_dir = Path(
            snapshot_download(
                repo_id=MODEL_REPO,
                revision=MODEL_REVISION,
                allow_patterns=[
                    MODEL_FILE,
                    "config.json",
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "special_tokens_map.json",
                ],
                cache_dir=cache_dir,
                local_files_only=not allow_download,
            )
        )
        validate_model_labels(json.loads((local_dir / "config.json").read_text()))
        self._tokenizer = Tokenizer.from_file(str(local_dir / "tokenizer.json"))
        if self._tokenizer.token_to_id("<pad>") != 1:
            raise ValueError("NLI tokenizer pad ID does not match pinned configuration")
        self._tokenizer.enable_padding(pad_id=1, pad_token="<pad>")

        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 2
        self._session = ort.InferenceSession(
            str(local_dir / MODEL_FILE), sess_options=options, providers=["CPUExecutionProvider"]
        )
        inputs = {value.name for value in self._session.get_inputs()}
        if inputs not in (
            {"input_ids", "attention_mask"},
            {"input_ids", "attention_mask", "token_type_ids"},
        ):
            raise ValueError(f"Unexpected pinned NLI model inputs: {sorted(inputs)}")
        outputs = self._session.get_outputs()
        if len(outputs) != 1 or outputs[0].name != "logits":
            raise ValueError("Unexpected pinned NLI model output")
        self._inputs = inputs

    @property
    def model_key(self) -> str:
        return MODEL_KEY

    def infer(self, claims: LockedClaims, *, batch_size: int = 4) -> list[NliObservation]:
        """Evaluate full quote–hypothesis pairs without silent token truncation."""
        return self.infer_pairs(
            [
                (case.case_id, premise_for_claim(claims, case.case_id), case.claim)
                for case in claims.data.cases
            ],
            batch_size=batch_size,
        )

    def infer_pairs(
        self, pairs: Sequence[tuple[str, str, str]], *, batch_size: int = 4
    ) -> list[NliObservation]:
        """Infer caller-prepared (ID, premise, hypothesis) triples; never supply labels."""
        if not 1 <= batch_size <= 32:
            raise ValueError("Batch size must be from 1 to 32")
        if len({item[0] for item in pairs}) != len(pairs):
            raise ValueError("NLI pair IDs must be unique")
        if any(
            not case_id or not premise.strip() or not hypothesis.strip()
            for case_id, premise, hypothesis in pairs
        ):
            raise ValueError("NLI pairs require ID, inspected premise and claim hypothesis")
        import numpy as np

        observations: list[NliObservation] = []
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start : start + batch_size]
            encodings = self._tokenizer.encode_batch(
                [(premise, hypothesis) for _, premise, hypothesis in batch]
            )
            # Padding is enabled, but *truncation is not*. Overlong inputs are an error.
            sizes = [sum(encoding.attention_mask) for encoding in encodings]
            if any(size > MAX_PAIR_TOKENS for size in sizes):
                raise ValueError("NLI pair exceeded the token budget; refusing silent truncation")
            inputs = {
                "input_ids": np.asarray([value.ids for value in encodings], dtype=np.int64),
                "attention_mask": np.asarray(
                    [value.attention_mask for value in encodings], dtype=np.int64
                ),
            }
            if "token_type_ids" in self._inputs:
                inputs["token_type_ids"] = np.asarray(
                    [value.type_ids for value in encodings], dtype=np.int64
                )
            logits = self._session.run(["logits"], inputs)[0]
            if logits.shape != (len(batch), len(EXPECTED_LABELS)):
                raise ValueError("NLI model returned an unexpected batch/logit shape")
            for (case_id, _, _), token_count, values in zip(batch, sizes, logits, strict=True):
                scores = tuple(float(value) for value in values)
                observations.append(
                    NliObservation(
                        case_id=case_id,
                        nli_label=predicted_nli_label(scores),
                        token_count=token_count,
                        logits=scores,
                    )
                )
        return observations
