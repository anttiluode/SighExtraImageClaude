import pytest

from sighextraimage.cli import build_parser, main


def test_help_exits_successfully():
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(["--help"])
    assert exc.value.code == 0


@pytest.mark.parametrize("command", ["synth", "extract", "invert", "gate0", "gate1", "benchmark"])
def test_core_subcommands_parse_without_model_imports(command):
    args = build_parser().parse_args([command])
    assert args.command == command


def test_main_without_command_prints_help_and_returns_two(capsys):
    assert main([]) == 2
    assert "SighExtraImage" in capsys.readouterr().out
