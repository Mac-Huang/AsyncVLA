"""Optional edge-stage diagnostics. No stage adds a CUDA synchronization.

CUDA spans include stream scheduling and host launch gaps. CPU thread durations
cover the calling thread, not the OpenMP workers. Nested stage times overlap.
"""

import csv
import os
from pathlib import Path
import subprocess
import threading
import time

import torch


def cpu_snapshot():
    paths = (
        "/proc/self/cgroup",
        "/sys/fs/cgroup/cpu.max",
        "/sys/fs/cgroup/cpu.stat",
        "/sys/fs/cgroup/cpuset.cpus.effective",
        "/sys/fs/cgroup/cpu/cpu.cfs_quota_us",
        "/sys/fs/cgroup/cpu/cpu.cfs_period_us",
        "/sys/fs/cgroup/cpu/cpu.stat",
    )
    values = {}
    for name in paths:
        try:
            values[name] = Path(name).read_text().strip()
        except OSError:
            pass
    return values


def cpu_counter_delta(before, after):
    result = {}
    for name in before.keys() & after.keys():
        if not name.endswith("/cpu.stat"):
            continue
        def counters(text):
            return {
                key: int(value) for key, value in (
                    line.split() for line in text.splitlines()
                )
            }
        first, last = counters(before[name]), counters(after[name])
        result[name] = {
            key: last[key] - first[key] for key in first.keys() & last.keys()
        }
    return result


def execution_context(device):
    context = {
        "thread": threading.current_thread().name,
        "native_thread_id": threading.get_native_id(),
        "torch_cpu_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "parallel_info": torch.__config__.parallel_info(),
        "cuda_stream": int(torch.cuda.current_stream(device).cuda_stream),
        "cpu_cgroup": cpu_snapshot(),
    }
    if hasattr(os, "sched_getaffinity"):
        context["cpu_affinity"] = sorted(os.sched_getaffinity(0))
    return context


class EdgeStageTimer:
    """Collect host durations and CUDA event spans without changing the forward."""

    def __init__(self, shead, device):
        self.device = device
        self.local = threading.local()
        self.handles = []
        # extract_features() bypasses the EfficientNet module's forward hooks.
        # The stem starts each encoder; compression finishes its visual token.
        self._hook_start(shead.cat_encoder._conv_stem, "pair_encoder")
        self._hook_finish(shead.compress_cat_enc, "pair_encoder")
        self._hook_start(shead.obs_encoder._conv_stem, "current_encoder")
        self._hook_finish(shead.compress_obs_enc, "current_encoder")
        for module, name in (
            (shead, "network"),
            (shead.decoder, "transformer"),
            (shead.action_predictor, "action_head"),
        ):
            self._hook_start(module, name)
            self._hook_finish(module, name)

    def _hook_start(self, module, name):
        def before(module, inputs):
            self.start_gpu(name)
        self.handles.append(module.register_forward_pre_hook(before))

    def _hook_finish(self, module, name):
        def after(module, inputs, output):
            self.finish_gpu(name)
        self.handles.append(module.register_forward_hook(after))

    def begin(self):
        self.local.current = {
            "cpu_wall_ms": {},
            "cpu_thread_ms": {},
            "events": {},
            "open": {},
            "calls": {},
        }

    def abort(self):
        self.local.current = None

    def cpu_call(self, name, fn):
        state = getattr(self.local, "current", None)
        if state is None:
            return fn()
        start = time.perf_counter_ns()
        thread_start = time.thread_time_ns()
        result = fn()
        thread_finish = time.thread_time_ns()
        state["cpu_wall_ms"][name] = (time.perf_counter_ns() - start) / 1e6
        state["cpu_thread_ms"][name] = (thread_finish - thread_start) / 1e6
        return result

    def start_gpu(self, name):
        state = getattr(self.local, "current", None)
        if state is None:
            return
        if name in state["open"] or name in state["events"]:
            raise RuntimeError(f"Duplicate edge stage: {name}")
        first = torch.cuda.Event(enable_timing=True)
        last = torch.cuda.Event(enable_timing=True)
        first.record(torch.cuda.current_stream(self.device))
        state["open"][name] = (
            first, last, time.perf_counter_ns(), time.thread_time_ns(),
        )
        state["calls"][name] = state["calls"].get(name, 0) + 1

    def finish_gpu(self, name):
        state = getattr(self.local, "current", None)
        if state is None:
            return
        first, last, start, thread_start = state["open"].pop(name)
        thread_finish = time.thread_time_ns()
        finish = time.perf_counter_ns()
        last.record(torch.cuda.current_stream(self.device))
        state["events"][name] = (first, last)
        state["cpu_wall_ms"][name + ".dispatch"] = (finish - start) / 1e6
        state["cpu_thread_ms"][name + ".dispatch"] = (
            thread_finish - thread_start
        ) / 1e6

    def transfer(self, name, fn):
        self.start_gpu(name)
        result = self.cpu_call(name, fn)
        self.finish_gpu(name)
        return result

    def collect(self):
        # The caller must complete its existing edge/command wait first.
        state = self.local.current
        if state["open"]:
            raise RuntimeError(f"Unfinished edge stages: {list(state['open'])}")
        required = (
            "pair_encoder", "current_encoder", "transformer", "action_head",
            "network", "current.h2d_bf16", "past.h2d_bf16",
        )
        for name in required:
            if state["calls"].get(name) != 1:
                raise RuntimeError(f"Expected one execution of edge stage: {name}")
        spans = {}
        for name, (first, last) in state["events"].items():
            if not last.query():
                raise RuntimeError("Collect edge timings after GPU completion.")
            spans[name] = first.elapsed_time(last)
        cpu = state["cpu_wall_ms"]
        preparation = sum(
            value for name, value in cpu.items()
            if name.rsplit(".", 1)[-1] in (
                "resize224", "to_tensor", "resize96", "normalize",
            )
        )
        result = {
            "cpu_wall_ms": cpu,
            "cpu_thread_ms": state["cpu_thread_ms"],
            "cuda_span_ms": spans,
            "cpu_prepare_ms": preparation,
            "cpu_prepare_thread_ms": sum(
                state["cpu_thread_ms"][name] for name in cpu
                if name.rsplit(".", 1)[-1] in (
                    "resize224", "to_tensor", "resize96", "normalize",
                )
            ),
            "h2d_bf16_wall_ms": (
                cpu["current.h2d_bf16"] + cpu["past.h2d_bf16"]
            ),
            "shead_dispatch_ms": cpu["shead_dispatch"],
            "encoder_calls": {
                "pair": state["calls"]["pair_encoder"],
                "current": state["calls"]["current_encoder"],
            },
        }
        self.local.current = None
        return result

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


class GpuTelemetry:
    """Sample operating state independently of the inference loops."""

    fields = (
        "timestamp", "index", "pstate", "clocks.current.sm",
        "clocks.current.memory", "power.draw", "utilization.gpu",
        "utilization.memory", "memory.used",
    )

    def __init__(self, device):
        self.index = torch.device(device).index
        if self.index is None:
            self.index = torch.cuda.current_device()
        self.rows = []
        self.error = None
        self.process = None
        self.reader = None

    def start(self):
        try:
            self.process = subprocess.Popen(
                [
                    "nvidia-smi", "-i", str(self.index),
                    "--query-gpu=" + ",".join(self.fields),
                    "--format=csv,noheader,nounits", "--loop-ms=250",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except OSError as exc:
            self.error = str(exc)
            return

        def read():
            for row in csv.reader(self.process.stdout):
                if len(row) == len(self.fields):
                    item = dict(zip(self.fields, (value.strip() for value in row)))
                    item["received_ns"] = time.perf_counter_ns()
                    self.rows.append(item)

        self.reader = threading.Thread(target=read, name="gpu-telemetry")
        self.reader.start()

    def finish(self):
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
            self.reader.join(timeout=5)
            stderr = self.process.stderr.read().strip()
            if stderr:
                self.error = stderr
            elif not self.rows:
                self.error = "nvidia-smi returned no GPU samples."
            self.process.stdout.close()
            self.process.stderr.close()
        return {
            "sample_interval_ms": 250,
            "samples": self.rows,
            "error": self.error,
        }


def observe_benchmark(fn, device, telemetry=False):
    before = cpu_snapshot()
    monitor = GpuTelemetry(device) if telemetry else None
    if monitor is not None:
        monitor.start()
    try:
        records = fn()
    finally:
        gpu = monitor.finish() if monitor is not None else None
    records["cpu_cgroup_before"] = before
    records["cpu_cgroup_after"] = cpu_snapshot()
    records["cpu_cgroup_delta"] = cpu_counter_delta(
        before, records["cpu_cgroup_after"],
    )
    if gpu is not None:
        records["gpu_telemetry"] = gpu
    return records
