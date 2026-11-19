"""Fail CI if PostgreSQL tests disappeared, failed, or were silently skipped."""

import argparse
from pathlib import Path
from xml.etree import ElementTree


def verify_junit(path: Path, *, min_integration: int = 5) -> tuple[int, int]:
    root = ElementTree.parse(path).getroot()  # noqa: S314 - local generated CI artifact
    cases = root.findall(".//testcase")
    if not cases:
        raise ValueError("No testcases in JUnit report")
    invalid = [
        case for case in cases if case.find("failure") is not None or case.find("error") is not None
    ]
    skipped = [case for case in cases if case.find("skipped") is not None]
    integration = [
        case
        for case in cases
        if "tests.integration." in case.get("classname", "")
        or "/integration/" in case.get("classname", "")
    ]
    if invalid or skipped:
        raise ValueError(f"Test report contains failed={len(invalid)}, skipped={len(skipped)}")
    if len(integration) < min_integration:
        raise ValueError(
            f"Integration suite shrank: {len(integration)} observed, minimum {min_integration}"
        )
    return len(cases), len(integration)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("--min-integration", type=int, default=5)
    args = parser.parse_args()
    total, integration = verify_junit(args.report, min_integration=args.min_integration)
    print(f"JUnit verified: total_passed={total}, integration_passed={integration}, skipped=0")


if __name__ == "__main__":
    main()
