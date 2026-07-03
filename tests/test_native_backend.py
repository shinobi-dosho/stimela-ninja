from shinobi.policies import build_args
from shinobi.schema import CabDef, ParamSchema


def test_echo_and_wrangler_extraction(native):
    cab = CabDef(
        name="flagsummary",
        command="/bin/echo",
        inputs={
            "text": ParamSchema(
                dtype="str", default="Total Flagged: 12.5% Total Counts: 100"
            )
        },
        wranglers={
            r"Total Flagged: (?P<percentage>[\d.]+)% Total Counts: .*": [
                "PARSE_OUTPUT:percentage:float"
            ]
        },
    )
    argv = build_args(cab, {})
    result = native.run(cab, argv)

    assert result.success
    assert result.outputs["percentage"] == 12.5


def test_failing_command_reports_nonzero(native):
    cab = CabDef(name="fail", command="/bin/false")
    result = native.run(cab, build_args(cab, {}))
    assert not result.success
    assert result.returncode != 0
