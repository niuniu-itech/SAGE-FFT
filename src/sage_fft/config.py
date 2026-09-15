# Author: even
"""Runtime configuration. Importing this module never contacts a server."""

import os
from pathlib import Path, PurePosixPath
import shlex


def workspace():
    """Keep generated data outside installed package sources."""
    return Path(os.environ.get("SAGE_WORKDIR", "runs")).expanduser().resolve()


def prepare_workspace():
    """Create output roots only when an experiment is explicitly launched."""
    root = workspace()
    for directory in (root, root / "experiments", root / "evidence"):
        directory.mkdir(parents=True, exist_ok=True)
    return root


def remote_root(study):
    root = os.environ.get("SAGE_REMOTE_ROOT", "/tmp/sage")
    if not root.startswith("/") or ".." in PurePosixPath(root).parts:
        raise ValueError("SAGE_REMOTE_ROOT must be an absolute POSIX path without '..'")
    return str(PurePosixPath(root) / study)


def npu_environment():
    root = os.environ.get("SAGE_CANN_ROOT", "/usr/local/Ascend/ascend-toolkit/latest")
    setup = os.environ.get("SAGE_NPU_SETUP", str(PurePosixPath(root).parent / "set_env.sh"))
    return f"set -e\nsource {shlex.quote(setup)}\nexport ASCEND_CANN_PACKAGE_PATH={shlex.quote(root)}\n"


def cuda_environment():
    root = os.environ.get("SAGE_CUDA_ROOT", "/usr/local/cuda")
    setup = os.environ.get("SAGE_CUDA_SETUP")
    prefix = "set -e\n"
    if setup:
        prefix += f"source {shlex.quote(setup)}\n"
    return (
        prefix
        + f"export PATH={shlex.quote(root + '/bin')}:$PATH\n"
        + f"export LD_LIBRARY_PATH={shlex.quote(root + '/lib64')}:${{LD_LIBRARY_PATH:-}}\n"
    )


def connect_ssh(device):
    """Connect using caller configuration and existing known-host keys."""
    import paramiko

    prefix = f"SAGE_{device.upper()}_"
    host = os.environ.get(prefix + "HOST")
    user = os.environ.get(prefix + "USER")
    if not host or not user:
        raise ValueError(f"Set {prefix}HOST and {prefix}USER before a hardware run")
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    known_hosts = os.environ.get("SAGE_KNOWN_HOSTS")
    if known_hosts:
        client.load_host_keys(str(Path(known_hosts).expanduser()))
    client.set_missing_host_key_policy(paramiko.RejectPolicy())
    try:
        client.connect(
            host,
            port=int(os.environ.get(prefix + "PORT", "22")),
            username=user,
            key_filename=os.environ.get(prefix + "KEY_FILE"),
            password=os.environ.get(prefix + "PASSWORD"),
            timeout=30,
            auth_timeout=30,
            banner_timeout=30,
            look_for_keys=True,
            allow_agent=True,
        )
    except Exception:
        client.close()
        raise
    return client
