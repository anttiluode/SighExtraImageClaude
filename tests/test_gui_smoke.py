from sighextraimage.cli import build_parser
from sighextraimage.gui import build_app


def test_gui_parser_accepts_local_launch_options():
    args = build_parser().parse_args(["gui", "--port", "7861"])
    assert args.command == "gui"
    assert args.port == 7861
    assert args.share is False


def test_app_contains_synthetic_and_photo_tabs():
    app = build_app()
    cfg = app.get_config_file()
    text = str(cfg)
    assert "Synthetic Lab" in text
    assert "Photo Inspector" in text
