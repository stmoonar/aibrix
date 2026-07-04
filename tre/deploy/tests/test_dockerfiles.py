from __future__ import annotations

from pathlib import Path


TRE_ROOT = Path(__file__).resolve().parents[2]


def test_n2_component_dockerfiles_have_shared_build_contract() -> None:
    requirements = (TRE_ROOT / "requirements-runtime.txt").read_text(encoding="utf-8")
    test_requirements = (TRE_ROOT / "requirements-test.txt").read_text(encoding="utf-8")

    assert "fastapi" in requirements
    assert "redis" in requirements
    assert "pytest" in test_requirements

    expected = {
        "controller": {
            "path": TRE_ROOT / "controller" / "Dockerfile",
            "copies": ("COPY common", "COPY deploy", "COPY controller", "COPY service-manager"),
            "cmd": 'CMD ["python", "-m", "tre_controller"]',
        },
        "service-manager": {
            "path": TRE_ROOT / "service-manager" / "Dockerfile",
            "copies": ("COPY common", "COPY deploy", "COPY service-manager"),
            "cmd": '"uvicorn", "tre_sm.server:create_app", "--factory"',
        },
        "ui": {
            "path": TRE_ROOT / "ui" / "Dockerfile",
            "copies": ("COPY common", "COPY deploy", "COPY ui"),
            "cmd": '"uvicorn", "tre_ui.server:create_app", "--factory"',
        },
    }

    for component, spec in expected.items():
        dockerfile = spec["path"].read_text(encoding="utf-8")
        assert "FROM python:3.11-slim" in dockerfile
        assert "latest" not in dockerfile.lower()
        assert "requirements-test.txt" in dockerfile
        assert "PYTHONPATH" in dockerfile
        for copy_directive in spec["copies"]:
            assert copy_directive in dockerfile, component
        assert spec["cmd"] in dockerfile
