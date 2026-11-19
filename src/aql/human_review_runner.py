"""Prepare label-hidden reviewer packets; compare *real* completed files if provided."""

import argparse
from pathlib import Path

from aql.evaluation import report_json
from aql.human_review import (
    blank_submission_template,
    build_review_packet,
    compare_submissions,
    packet_json,
    packet_sha256,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "compare"))
    parser.add_argument("--out-dir", type=Path, default=Path("test-results/human-review"))
    parser.add_argument("--left", type=Path)
    parser.add_argument("--right", type=Path)
    args = parser.parse_args()

    packet = build_review_packet()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.phase == "prepare":
        (args.out_dir / "packet.json").write_text(packet_json(packet), encoding="utf-8")
        for reviewer in ("reviewer_a", "reviewer_b"):
            (args.out_dir / f"template-{reviewer}.json").write_text(
                report_json(blank_submission_template(packet, reviewer)), encoding="utf-8"
            )
        print(
            f"Prepared label-hidden synthetic review packet: {len(packet.cases)} answers, "
            f"{sum(len(c.claims) for c in packet.cases)} author-segmented claims; "
            f"fingerprint={packet_sha256(packet)}. "
            "Both reviewer templates are EMPTY and invalid until completed."
        )
    else:
        if args.left is None or args.right is None:
            parser.error("compare requires --left and --right completed reviewer JSON files")
        report = compare_submissions(packet, args.left.read_bytes(), args.right.read_bytes())
        (args.out_dir / "reviewer-agreement.json").write_text(report_json(report), encoding="utf-8")
        print(
            f"Compared two submission files: "
            f"claim agreement={report['claims_three_class']['raw_agreement']}, "
            f"kappa={report['claims_three_class']['cohen_kappa']}; "
            "human identities unverified, NO adjudicated gold"
        )


if __name__ == "__main__":
    main()
