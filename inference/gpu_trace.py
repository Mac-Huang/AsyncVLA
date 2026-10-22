"""Measure CUDA kernel activity per annotated model call, using Nsight SQLite."""

import argparse
from bisect import bisect_right
from collections import defaultdict
import json
from pathlib import Path
import sqlite3
import statistics


PREFIX = "asyncvla_gpu/"


def interval_union_ns(intervals):
    total, last = 0, None
    for start, end in sorted(intervals):
        if end < start:
            raise ValueError("A kernel finishes before it starts.")
        if last is None or start > last:
            total += end - start
        elif end > last:
            total += end - last
        last = end if last is None else max(last, end)
    return total


def process_id(serialized):
    return (serialized >> 24) & 0xFFFFFF


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize(calls):
    groups = defaultdict(list)
    for call in calls:
        groups[(call["condition"], call["component"])].append(call)
    result = {}
    for (condition, component), samples in sorted(groups.items()):
        values = [s["gpu_kernel_busy_ms"] for s in samples]
        result[f"{condition}/{component}"] = {
            "calls": len(samples),
            "gpu_kernel_busy_ms": {
                "mean": statistics.mean(values),
                "p50": percentile(values, 0.5),
                "p95": percentile(values, 0.95),
            },
        }
    return result


def analyze_trace(database, dispatch_path, output_path):
    dispatch = json.loads(Path(dispatch_path).read_text())
    expected = {call["label"]: call for call in dispatch["calls"]}
    if len(expected) != len(dispatch["calls"]) or not expected:
        raise ValueError("Model call labels must be unique and nonempty.")
    connection = sqlite3.connect(Path(database).resolve().as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        tables = {row[0] for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )}
        required = {"NVTX_EVENTS", "StringIds", "CUPTI_ACTIVITY_KIND_KERNEL"}
        if not required <= tables:
            raise ValueError(f"Missing Nsight tables: {sorted(required - tables)}")
        ranges = defaultdict(list)
        found = set()
        for row in connection.execute(
            "SELECT n.start, n.end, n.globalTid, COALESCE(n.text, s.value) AS label "
            "FROM NVTX_EVENTS n LEFT JOIN StringIds s ON n.textId=s.id "
            "WHERE n.end IS NOT NULL"
        ):
            label = row["label"]
            if label not in expected:
                continue
            if label in found:
                raise ValueError(f"Duplicate NVTX model range: {label}")
            found.add(label)
            ranges[row["globalTid"]].append((row["start"], row["end"], label))
        if found != expected.keys():
            raise ValueError(f"Missing NVTX calls: {sorted(expected.keys() - found)[:5]}")
        starts = {}
        for tid, values in ranges.items():
            values.sort()
            if any(a[1] > b[0] for a, b in zip(values, values[1:])):
                raise ValueError("Model ranges overlap on the same host thread.")
            starts[tid] = [value[0] for value in values]

        # Correlate the launch API on its host thread with the model NVTX range.
        launches = {}
        for table in ("CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_DRIVER"):
            if table not in tables:
                continue
            for row in connection.execute(
                f"SELECT start, end, globalTid, correlationId FROM {table}"
            ):
                tid = row["globalTid"]
                if tid not in ranges:
                    continue
                index = bisect_right(starts[tid], row["start"]) - 1
                if index < 0:
                    continue
                begin, end, label = ranges[tid][index]
                if begin <= row["start"] and row["end"] <= end:
                    key = (process_id(tid), row["correlationId"])
                    if key in launches and launches[key] != label:
                        raise ValueError("One CUDA correlation belongs to multiple calls.")
                    launches[key] = label
        if not launches:
            raise ValueError("No CUDA launch APIs correlate with model calls.")

        kernels = defaultdict(list)
        names = {row["id"]: row["value"] for row in connection.execute(
            "SELECT id, value FROM StringIds"
        )}
        for row in connection.execute(
            "SELECT start, end, globalPid, correlationId, streamId, deviceId, shortName "
            "FROM CUPTI_ACTIVITY_KIND_KERNEL"
        ):
            label = launches.get((process_id(row["globalPid"]), row["correlationId"]))
            if label is not None:
                kernels[label].append({
                    "start_ns": row["start"], "finish_ns": row["end"],
                    "stream_id": row["streamId"], "device_id": row["deviceId"],
                    "name": names[row["shortName"]],
                })
        calls = []
        for label, call in expected.items():
            events = kernels[label]
            if not events:
                raise ValueError(f"No GPU kernels were recorded for {label}")
            devices = {e["device_id"] for e in events}
            if len(devices) != 1:
                raise ValueError("This benchmark requires one GPU per model call.")
            intervals = [(e["start_ns"], e["finish_ns"]) for e in events]
            busy = interval_union_ns(intervals)
            first = min(a for a, b in intervals)
            finish = max(b for a, b in intervals)
            calls.append({
                **call,
                "gpu_first_kernel_start_ns": first,
                "gpu_last_kernel_finish_ns": finish,
                "gpu_kernel_busy_ms": busy / 1e6,
                "gpu_kernel_sum_ms": sum(b - a for a, b in intervals) / 1e6,
                "gpu_kernel_span_ms": (finish - first) / 1e6,
                "gpu_gap_ms": (finish - first - busy) / 1e6,
                "gpu_kernel_count": len(events),
                "cuda_stream_ids": sorted({e["stream_id"] for e in events}),
            })
        result = {
            **{k: v for k, v in dispatch.items() if k != "calls"},
            "metric": {
                "primary": "gpu_kernel_busy_ms",
                "definition": "Union of CUDA kernel execution intervals for each model call.",
                "excluded": [
                    "CPU preprocessing and tokenization", "host-to-device transfers",
                    "CUDA memcpy and memset activities", "CPU launch and synchronization time",
                    "gaps with none of this call's kernels executing", "robot command conversion",
                ],
                "limits": [
                    "Kernel durations include device-memory stalls and contention.",
                    "Different components can overlap on the GPU; their times are not additive.",
                    "CUDA event spans and host wall times remain separate reference fields.",
                    "Tracing can perturb execution; this is an instrumented measurement.",
                ],
            },
            "summary": summarize(calls),
            "calls": calls,
        }
        Path(output_path).write_text(json.dumps(result, indent=2) + "\n")
        for name, values in result["summary"].items():
            stats = values["gpu_kernel_busy_ms"]
            print(f"{name}: {values['calls']} calls, GPU kernel busy "
                  f"mean={stats['mean']:.3f} ms, p50={stats['p50']:.3f} ms, "
                  f"p95={stats['p95']:.3f} ms")
        return result
    finally:
        connection.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path)
    parser.add_argument("dispatch", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    analyze_trace(args.database, args.dispatch, args.output)
