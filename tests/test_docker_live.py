"""Live integration tests against a real container runtime and a real
radio-astronomy tool image. These are skipped if docker isn't available or
the image isn't present locally -- they're not meant to trigger an image
pull in CI, just to prove the container backend actually works against
something real whenever it's run somewhere that has it (as it was
developed and verified here, against a cached quay.io/stimela/wsclean
image).
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

from shinobi.backends.container import DockerBackend
from shinobi.loaders._modelgen import build_model
from shinobi.steps.schema import Cab

WSCLEAN_IMAGE = "quay.io/stimela/wsclean:1.8.0"


def _image_available(image: str) -> bool:
    if not shutil.which("docker"):
        return False
    return (
        subprocess.run(
            ["docker", "image", "inspect", image],
            capture_output=True,
        ).returncode
        == 0
    )


requires_wsclean_image = pytest.mark.skipif(
    not _image_available(WSCLEAN_IMAGE),
    reason=f"docker or {WSCLEAN_IMAGE} not available locally",
)


@requires_wsclean_image
def test_real_tool_runs_inside_container():
    """wsclean --version, for real, inside the real image."""
    cab = Cab(
        name="wsclean",
        command="wsclean",
        image=WSCLEAN_IMAGE,
        inputs_model=build_model("In", {"version": ("bool", False, None)}),
        outputs_model=build_model("Out", {}),
    )
    backend = DockerBackend()
    result = backend.run(cab, ["wsclean", "--version"], {"version": True})

    assert result.success
    assert "WSClean" in result.stdout


@requires_wsclean_image
def test_host_file_visible_at_same_path_via_bind_mount(tmp_path):
    """Proves the bind-mount logic actually works: a host file, outside
    any hardcoded workdir, must be readable at its own path inside the
    container purely because its parent dir was mounted.
    """
    host_file = tmp_path / "hello.txt"
    host_file.write_text("hello from the host\n")

    cab = Cab(
        name="probe",
        command="/bin/cat",
        image=WSCLEAN_IMAGE,
        inputs_model=build_model("In", {"path": ("File", True, None)}),
        outputs_model=build_model("Out", {}),
    )
    backend = DockerBackend()
    # hand-built argv (positional, not --flag-style) -- this test is about
    # mount visibility, not the arg-building policy (covered elsewhere)
    result = backend.run(cab, ["/bin/cat", str(host_file)], {"path": str(host_file)})

    assert result.success
    assert result.stdout == "hello from the host\n"
