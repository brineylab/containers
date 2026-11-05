"""The architecture probe behind the x86_only skip."""

import subprocess

import pytest

import conftest

pytestmark = pytest.mark.images("brineylab/datascience")


def _stub_run(monkeypatch, returncode: int, stdout: str) -> None:
    def run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, returncode, stdout=stdout, stderr="")

    monkeypatch.setattr(conftest.subprocess, "run", run)


def test_image_arch_returns_inspect_output(monkeypatch):
    _stub_run(monkeypatch, 0, "arm64\n")
    assert conftest.image_arch("brineylab/datascience:arch-probe-arm") == "arm64"


def test_image_arch_raises_when_inspect_fails(monkeypatch):
    _stub_run(monkeypatch, 1, "")
    with pytest.raises(RuntimeError):
        conftest.image_arch("brineylab/datascience:arch-probe-missing")
