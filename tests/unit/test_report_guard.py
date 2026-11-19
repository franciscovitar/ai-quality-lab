from pathlib import Path

import pytest

from scripts.check_test_report import verify_junit


def test_report_guard_accepts_real_integration_cases(tmp_path: Path) -> None:
    report = tmp_path / "report.xml"
    report.write_text(
        '<testsuite><testcase classname="tests.unit.test_api" name="health"/>'
        + '<testcase classname="tests.integration.test_database" name="db"/>' * 5
        + "</testsuite>",
        encoding="utf-8",
    )
    assert verify_junit(report) == (6, 5)


@pytest.mark.parametrize("status", ["skipped", "failure", "error"])
def test_report_guard_rejects_nonpassing_case(tmp_path: Path, status: str) -> None:
    report = tmp_path / "report.xml"
    report.write_text(
        '<testsuite><testcase classname="tests.integration.test_database" name="db">'
        f"<{status}/></testcase></testsuite>",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        verify_junit(report, min_integration=1)


def test_report_guard_rejects_missing_integration(tmp_path: Path) -> None:
    report = tmp_path / "report.xml"
    report.write_text(
        '<testsuite><testcase classname="tests.unit.test_api" name="health"/></testsuite>',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="shrank"):
        verify_junit(report)
