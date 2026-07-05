from __future__ import annotations

from pathlib import Path


def test_import_brainregion_package():
    """package discovery 改坏第一时间发现（包名/import/__version__ 一次覆盖）。"""
    import brainregion

    assert brainregion.__version__


def test_ping_reports_brainregion_name():
    import brainregion
    from brainregion.server import ping

    got = ping()

    assert got["name"] == "brainregion"
    assert got["legacy_name"] == "brain_region"
    assert got["version"] == brainregion.__version__


def test_package_version_matches_pyproject():
    import brainregion

    text = Path("pyproject.toml").read_text(encoding="utf-8")

    assert f'version = "{brainregion.__version__}"' in text


def test_pyproject_exposes_brainregion_command_and_legacy_aliases():
    text = Path("pyproject.toml").read_text(encoding="utf-8")

    assert 'name = "brainregion"' in text
    assert 'brainregion = "brainregion.cli:main"' in text
    assert 'brain-region = "brainregion.cli:main"' in text
    assert 'brain-region-mcp = "brainregion.server:main"' in text
    assert 'design-review-mcp = "brainregion.server:main"' in text
    assert 'design-review = "brainregion.cli:main"' in text
