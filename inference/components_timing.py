"""Measure isolated calls and save timing data."""

import json
from pathlib import Path
import time

import numpy as np
import torch


def _summary(samples):
    return {
        "samples": samples,
        "mean": float(np.mean(samples)),
        "p50": float(np.percentile(samples, 50)),
        "p95": float(np.percentile(samples, 95)),
    }

def measure_call(name, function, device, wait_for_gpu, warmup=10, repeats=100):
    """Measure one component alone"""

    # block CPU sumbitting until GPU work done
    torch.cuda.synchronize(device)
    for _ in range(warmup):
        output = function()
        torch.cuda.synchronize(device)
        del output

    wall_samples = []
    for index in range(repeats):
        torch.cuda.synchronize(device)

        start_ns = time.perf_counter_ns()
        output = function()

        if wait_for_gpu:
            torch.cuda.synchronize(device)

        wall_samples.append((time.perf_counter_ns() - start_ns) / 1e6)

        # Release this output before the next measured call.
        if index != repeats - 1:
            del output

    record = {
        "component": name,
        "warmup": warmup,
        "repeats": repeats,
        "wall_ms": _summary(wall_samples),
    }
    return output, record

def save_results(stages, device, mode="isolated", compiled_components=()):
    report = {
        "mode": mode,
        "compiled_components": list(compiled_components),
        "cuda_graph_capture": "automatic_unverified" if compiled_components else "disabled",
        "cpu_threads": torch.get_num_threads(),
        "stages": stages,
    }
    output = Path("results") / f"components_{mode}_{time.time_ns()}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    return output