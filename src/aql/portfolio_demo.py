"""Explain five real, frozen synthetic model outputs without running an LLM.

This is presentation code, not a new quality judge or a live RAG answer endpoint.
"""

from pathlib import Path

from aql.claim_runner import DEFAULT_EVIDENCE, DEFAULT_RETRIEVAL
from aql.eval_runner import DEFAULT_CORPUS
from aql.generated_claim_review import load_and_review
from aql.generation_evaluation import GeneratedBatch, PromptBatch
from aql.generation_runner import DEFAULT_CONTEXTS

SOURCE_CI_URL = "https://github.com/franciscovitar/ai-quality-lab/actions/runs/44588063650"
EXAMPLES = (
    {
        "id": "R101",
        "title": "Una respuesta respaldada",
        "category": "supported",
        "short_label": "Bien respaldada",
        "observation": (
            "La respuesta identifica los estados HTTP 429 y 503, tal como dice la fuente."
        ),
        "takeaway": (
            "Una cita válida ayuda, pero por sí sola no demuestra que toda respuesta sea correcta."
        ),
        "basis": (
            "Cita validada automáticamente; afirmación revisada provisionalmente por el proyecto."
        ),
    },
    {
        "id": "R106",
        "title": "Tres días no son treinta",
        "category": "contradicted",
        "short_label": "Dato contradictorio",
        "observation": (
            "El modelo respondió 3 días. El único documento proporcionado establece treinta días."
        ),
        "takeaway": "Un texto convincente puede alterar un dato exacto de la documentación.",
        "basis": (
            "Contradicción señalada en revisión provisional del proyecto; no adjudicación externa."
        ),
    },
    {
        "id": "R108",
        "title": "Verdadero, pero no responde",
        "category": "non_answer",
        "short_label": "No responde",
        "observation": (
            "La pregunta pide un porcentaje de jitter. La fuente solo confirma que existe "
            "jitter, y el modelo repite esa frase sin dar un porcentaje."
        ),
        "takeaway": (
            "Una afirmación respaldada y una cita auténtica pueden coexistir "
            "con una respuesta inútil."
        ),
        "basis": (
            "La falta de abstención se mide automáticamente; pertinencia revisada provisionalmente."
        ),
    },
    {
        "id": "R105",
        "title": "Referencias que nunca existieron",
        "category": "invalid_citation",
        "short_label": "Citas inventadas",
        "observation": (
            "El modelo citó [R105:2] y [R105:3], pero recibió solamente [R105:0] y [R105:1]."
        ),
        "takeaway": (
            "Que una respuesta tenga números entre corchetes no significa que sean fuentes reales."
        ),
        "basis": "IDs inexistentes detectados por el verificador automático de sintaxis de citas.",
    },
    {
        "id": "R110",
        "title": "Una instrucción SQL sin respaldo",
        "category": "unsupported_command",
        "short_label": "Comando inventado",
        "observation": (
            "El modelo redactó un DELETE incompleto. La documentación entregada "
            "solo enumeraba comprobaciones operativas y nunca autorizó ese comando."
        ),
        "takeaway": (
            "Una sugerencia técnica no debe ejecutarse sin comprobar su procedencia y seguridad."
        ),
        "basis": (
            "Falta de respaldo señalada provisionalmente por el proyecto; ningún SQL fue ejecutado."
        ),
    },
)


def project_root() -> Path:
    """Support both editable checkout and installed package inside Docker /app."""
    for candidate in (Path.cwd(), Path(__file__).resolve().parents[2], Path("/app")):
        if (candidate / "data/evals/frozen_generation_v1/prompts.json").is_file():
            return candidate
    raise FileNotFoundError("Frozen synthetic portfolio demo assets not available")


def load_portfolio_demo() -> dict:
    """Read the existing verified snapshot; do not synthesize new model answers."""
    root = project_root()
    prompts_path = root / "data/evals/frozen_generation_v1/prompts.json"
    generated_path = root / "data/evals/frozen_generation_v1/generated-outputs.json"
    review = load_and_review(
        prompts_path=prompts_path,
        generated_path=generated_path,
        review_path=root / "data/evals/generated_claim_review_v1.json",
        context_path=root / DEFAULT_CONTEXTS,
        evidence_path=root / DEFAULT_EVIDENCE,
        retrieval_path=root / DEFAULT_RETRIEVAL,
        corpus_path=root / DEFAULT_CORPUS,
    )
    prompts = PromptBatch.model_validate_json(prompts_path.read_bytes())
    generated = GeneratedBatch.model_validate_json(generated_path.read_bytes())
    prompt_map = {case.case_id: case for case in prompts.cases}
    generated_map = {case.case_id: case for case in generated.cases}
    review_map = {case["case_id"]: case for case in review["cases"]}

    # Frozen characteristics of the demonstrations; a changed artifact should
    # fail explicitly rather than silently turn an established story into fiction.
    frozen_expectations = (
        review_map["R101"]["all_claims_author_labeled_supported"]
        and review_map["R106"]["verdict_counts"]["contradicted"] == 1
        and review_map["R108"]["expected_behavior"] == "abstain"
        and not review_map["R108"]["strict_abstention"]
        and review_map["R105"]["citation_protocol_ok"] is False
        and "DELETE FROM" in generated_map["R110"].text
    )
    if not frozen_expectations:
        raise ValueError("Frozen example observations no longer match their evidence")

    cases = []
    for story in EXAMPLES:
        case_id = story["id"]
        prompt = prompt_map[case_id]
        result = generated_map[case_id]
        evaluated = review_map[case_id]
        cases.append(
            {
                **story,
                "question": prompt.question,
                "model_answer": result.text,
                "source_passages": [
                    {"id": item.citation_id, "quote": item.quote} for item in prompt.context
                ],
                "citation_syntax_ok": evaluated["citation_protocol_ok"],
                "strict_abstention": evaluated["strict_abstention"],
                "expected_behavior": evaluated["expected_behavior"],
                "author_claim_counts": evaluated["verdict_counts"],
                "method_limit": (
                    "Fuente y generación reales del experimento; etiquetas semánticas "
                    "provisionales del autor, sin revisión independiente."
                ),
            }
        )

    return {
        "mode": "frozen_synthetic_research_demo",
        "title": "AI Quality Lab — cinco respuestas bajo la lupa",
        "source_ci_url": SOURCE_CI_URL,
        "model": "SmolLM2-135M-Instruct, CPU ONNX, artefacto congelado",
        "independent_reviewers": 0,
        "limitations": (
            "Sin generación en vivo; los juicios semánticos son provisionales "
            "del proyecto, no una certificación de seguridad."
        ),
        "examples": cases,
    }
