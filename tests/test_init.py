import airbornegeo


def test_report_instantiates_and_lists_core_packages():
    """Report() should build a scooby report listing the package's core dependencies."""
    report = airbornegeo.Report()
    report_str = str(report)
    assert "numpy" in report_str
    assert "harmonica" in report_str
