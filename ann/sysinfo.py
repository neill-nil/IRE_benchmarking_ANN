"""Record machine specs and library versions for the report."""
import json
import os
import platform
import subprocess
import sys

import faiss
import h5py
import matplotlib
import numpy as np
import pandas as pd


def _cpu_model():
    try:
        out = subprocess.run(["lscpu"], capture_output=True, text=True).stdout
        for line in out.splitlines():
            if line.startswith("Model name"):
                return line.split(":", 1)[1].strip()
    except FileNotFoundError:
        pass
    return platform.processor()


def _ram_gb():
    with open("/proc/meminfo") as f:
        kb = int(f.readline().split()[1])
    return round(kb / 1024 ** 2, 1)


def collect():
    return {
        "cpu": _cpu_model(),
        "logical_cpus": os.cpu_count(),
        "ram_gb": _ram_gb(),
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "faiss": faiss.__version__,
        "faiss_compile": faiss.get_compile_options().strip(),
        "numpy": np.__version__,
        "h5py": h5py.__version__,
        "pandas": pd.__version__,
        "matplotlib": matplotlib.__version__,
    }


if __name__ == "__main__":
    info = collect()
    out = sys.argv[1] if len(sys.argv) > 1 else None
    text = json.dumps(info, indent=2)
    print(text)
    if out:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        with open(out, "w") as f:
            f.write(text + "\n")
