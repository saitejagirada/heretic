# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Union

import torch
import tomli_w

from .config import Settings


def write_config_toml(settings: Settings, path: Union[str, Path]) -> None:
    """
    Write the configuration used for this run to a TOML file.
    The settings are dumped to a dictionary and then serialized to TOML.
    """
    config_dict = settings.model_dump(exclude_none=True)  # ← add exclude_none=True
    with open(path, "wb") as f:
        tomli_w.dump(config_dict, f)


def write_requirements_txt(path: Union[str, Path]) -> None:
    """
    Write the output of `pip freeze` to a requirements.txt file.
    If `pip freeze` fails, writes an error message.
    """
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True,
            text=True,
            check=True,
        )
        with open(path, "w") as f:
            f.write(result.stdout)
    except (subprocess.SubprocessError, FileNotFoundError) as e:
        with open(path, "w") as f:
            f.write(f"# Failed to capture requirements: {e}\n")


def write_environment_txt(path: Union[str, Path]) -> None:
    """
    Write system and environment information to a text file.
    Includes platform, Python version, PyTorch version, CUDA info, etc.
    """
    lines = []
    lines.append(f"Platform: {platform.platform()}")
    lines.append(f"Python: {sys.version}")
    lines.append(f"PyTorch: {torch.__version__}")

    if torch.cuda.is_available():
        lines.append("CUDA available: True")
        lines.append(f"CUDA version: {torch.version.cuda}")
        lines.append(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            lines.append(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        # Try to get nvidia-smi output for more details
        try:
            nvidia_smi = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                check=True,
            )
            lines.append("\n# nvidia-smi GPU info:")
            for line in nvidia_smi.stdout.strip().split("\n"):
                lines.append(line)
        except (subprocess.SubprocessError, FileNotFoundError):
            lines.append("# nvidia-smi not available")
    else:
        lines.append("CUDA available: False")

    # Check for other accelerators (MPS, ROCm, etc.)
    if torch.backends.mps.is_available():
        lines.append("MPS available: True")
    else:
        lines.append("MPS available: False")

    # Optionally add environment variables that affect behavior
    import os

    relevant_env_vars = [
        "CUDA_VISIBLE_DEVICES",
        "PYTORCH_CUDA_ALLOC_CONF",
        "OMP_NUM_THREADS",
    ]
    lines.append("\n# Relevant environment variables:")
    for var in relevant_env_vars:
        lines.append(f"{var}={os.environ.get(var, '')}")

    with open(path, "w") as f:
        f.write("\n".join(lines))