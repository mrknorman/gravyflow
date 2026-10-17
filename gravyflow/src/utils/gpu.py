"""
Each NVIDIA GPU's memory and utilisation, as ``nvidia-smi`` reports them: what
the process manager (``gf.Manager``) places its processes by.
"""
import logging
import subprocess
from pathlib import Path
from typing import List, Optional

#: Where nvidia-smi is looked for.
NVIDIA_SMI_PATH = Path("/usr/bin/nvidia-smi")

logger = logging.getLogger(__name__)


def get_gpu_memory_info() -> Optional[List[dict]]:
    """
    Queries nvidia-smi to obtain GPU information.

    Returns:
        A list of dictionaries with keys: 'index', 'total', 'used', 'free', 'utilization'.
        Returns None if nvidia-smi is not available or fails.
    """
    if not NVIDIA_SMI_PATH.exists():
        logger.error("nvidia-smi not found at %s", NVIDIA_SMI_PATH)
        return None

    try:
        output = subprocess.check_output(
            [str(NVIDIA_SMI_PATH),
             "--query-gpu=index,memory.total,memory.used,memory.free,utilization.gpu",
             "--format=csv,noheader,nounits"],
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
    except subprocess.CalledProcessError as e:
        logger.error("Error running nvidia-smi: %s", e.output)
        return None

    gpu_info_list = []
    for line in output.strip().split("\n"):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        gpu_info_list.append({
            "index": parts[0],
            "total": int(parts[1]),
            "used": int(parts[2]),
            "free": int(parts[3]),
            "utilization": int(parts[4])
        })
    return gpu_info_list
