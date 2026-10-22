"""Run fresh GPU model forwards and measure their CUDA kernel activity."""

import argparse
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

from gpu_trace import PREFIX, analyze_trace


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("all", "isolated", "async"), default="all")
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--duration", type=float, default=10)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trace-child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if (args.repeats <= 0 or args.warmup < 0 or args.cpu_threads <= 0
            or not math.isfinite(args.duration) or args.duration <= 0):
        parser.error("Require positive repeats, duration, CPU threads, and nonnegative warmup.")
    args.output_dir = args.output_dir.resolve()
    return args


def trace_command(args, nsys):
    child = [
        sys.executable, str(Path(__file__).resolve()), "--trace-child",
        "--mode", args.mode, "--repeats", str(args.repeats),
        "--warmup", str(args.warmup), "--duration", str(args.duration),
        "--cpu-threads", str(args.cpu_threads),
        "--output-dir", str(args.output_dir),
    ]
    return [
        nsys, "profile", "--trace=cuda,nvtx", "--sample=none", "--cpuctxsw=none",
        "--capture-range=cudaProfilerApi", "--capture-range-end=stop",
        "--force-overwrite=false", "--output", str(args.output_dir / "gpu_trace"),
        *child,
    ]


def launch(args):
    nsys = shutil.which("nsys")
    if nsys is None:
        raise RuntimeError("Nsight Systems CLI (nsys) is required for GPU kernel timing.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.iterdir()):
        raise FileExistsError("Use an empty output directory to preserve previous results.")
    command = trace_command(args, nsys)
    version = subprocess.check_output([nsys, "--version"], text=True).strip()
    (args.output_dir / "command.json").write_text(json.dumps({
        "command": command, "nsight_systems": version,
        "repository": str(Path.cwd()),
    }, indent=2) + "\n")
    print(version, flush=True)
    with (args.output_dir / "run.log").open("w") as log:
        with subprocess.Popen(command, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
            if process.wait() != 0:
                raise RuntimeError("GPU benchmark failed. Read run.log for diagnostics.")
    reports = list(args.output_dir.glob("gpu_trace.nsys-rep"))
    if len(reports) != 1:
        raise RuntimeError("Nsight did not produce the expected GPU trace report.")
    database = args.output_dir / "gpu_trace.sqlite"
    if not database.exists():
        subprocess.run([
            nsys, "export", "--type", "sqlite", "--output", str(database),
            str(reports[0]),
        ], check=True)
    analyze_trace(database, args.output_dir / "dispatch.json",
                  args.output_dir / "gpu_results.json")


def benchmark(args):
    # Import the ML stack only in the traced child process.
    import numpy as np
    from PIL import Image
    import torch
    from run_asyncvla_baseline import (
        BaseVLM, EdgeAdaptor, Inference, InferenceConfig,
        define_model, finish_robot_command,
    )

    torch.set_num_threads(args.cpu_threads)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for GPU kernel timing.")
    cfg = InferenceConfig()
    (vla, action_head, pose_projector, shead, action_proj, device,
     patches, tokenizer, processor) = define_model(cfg)

    def load_rgb(name):
        with Image.open(Path("inference") / name) as image:
            return image.convert("RGB")

    raw = {
        "base_image": load_rgb("past.png"), "current_image": load_rgb("cur.png"),
        "goal_image": load_rgb("goal.png"),
        "goal_pose": [10.0, -100.0, np.cos(-90 / 180 * 3.1415),
                      np.sin(-90 / 180 * 3.1415)],
        "language": "move toward blue trash bin", "modality_id": 4,
    }
    inference = Inference(
        save_dir="./inference", lan_inst_prompt=raw["language"],
        goal_utm=(0.0, 0.0), goal_compass=0.0,
        goal_image_PIL=raw["goal_image"],
        action_tokenizer=tokenizer, processor=processor,
    )
    base = BaseVLM(vla, pose_projector, action_proj, inference, device, patches).eval()
    edge = EdgeAdaptor(shead, device).eval()
    prepared_base = base.prepare_inputs(raw)
    current = edge.prepare_image(raw["current_image"], "current")
    past = edge.prepare_image(raw["base_image"], "past")
    torch.cuda.synchronize(device)
    with torch.no_grad():
        reference_tokens = base(raw)
        prepared_tokens = base.forward_prepared(*prepared_base)
        torch.testing.assert_close(reference_tokens, prepared_tokens, rtol=0, atol=0)
        reference_deltas = edge(raw["current_image"], raw["base_image"], reference_tokens)
        prepared_deltas = shead(current, past, prepared_tokens)
        torch.testing.assert_close(reference_deltas, prepared_deltas, rtol=0, atol=0)
        if not torch.isfinite(prepared_tokens).all() or not torch.isfinite(prepared_deltas).all():
            raise RuntimeError("The prepared-input model produced nonfinite values.")
    torch.cuda.synchronize(device)
    print("Raw-input and prepared-input outputs match exactly for both models.", flush=True)
    del reference_tokens, prepared_tokens, reference_deltas, prepared_deltas

    # Count actual backbone executions. These counters never store visual features.
    counts = {"base_vision": 0, "edge_pair_vision": 0, "edge_current_vision": 0}
    keys = {
        "base": ("base_vision",),
        "edge": ("edge_pair_vision", "edge_current_vision"),
    }
    handles = []
    for module, name in (
        (vla.vision_backbone, "base_vision"),
        (shead.cat_encoder._conv_stem, "edge_pair_vision"),
        (shead.obs_encoder._conv_stem, "edge_current_vision"),
    ):
        def increment(module, inputs, name=name):
            counts[name] += 1
        handles.append(module.register_forward_pre_hook(increment))

    calls = []
    origin_ns = time.perf_counter_ns()

    def base_forward():
        return base.forward_prepared(*prepared_base)

    def edge_forward(tokens):
        return shead(current, past, tokens)

    def measured(condition, component, call_id, fn, stream):
        label = f"{PREFIX}{condition}/{component}/{call_id}"
        before = {key: counts[key] for key in keys[component]}
        first = torch.cuda.Event(enable_timing=True)
        last = torch.cuda.Event(enable_timing=True)
        start = time.perf_counter_ns()
        torch.cuda.nvtx.range_push(label)
        try:
            first.record(stream)
            output = fn()
            last.record(stream)
        finally:
            torch.cuda.nvtx.range_pop()
        stream.synchronize()
        finish = time.perf_counter_ns()
        executed = {key: counts[key] - before[key] for key in keys[component]}
        if any(value <= 0 for value in executed.values()):
            raise RuntimeError(f"A vision encoder did not execute in {label}")
        if component == "edge" and any(value != 1 for value in executed.values()):
            raise RuntimeError(f"Expected one execution of each edge encoder in {label}")
        record = {
            "label": label, "condition": condition, "component": component,
            "call_id": call_id, "cpu_start_ns": start, "cpu_finish_ns": finish,
            "model_wall_ms": (finish - start) / 1e6,
            "gpu_event_span_ms": first.elapsed_time(last),
            "encoder_calls": executed,
        }
        return output, record

    def command_record(deltas, record, stream):
        command = finish_robot_command(deltas, inference)
        stream.synchronize()
        finish = time.perf_counter_ns()
        record.update(
            command_finish_ns=finish,
            framework_ms=(finish - record["cpu_finish_ns"]) / 1e6,
            edge_to_command_ms=(finish - record["cpu_start_ns"]) / 1e6,
            linear_velocity=float(command[0]), angular_velocity=float(command[1]),
        )

    def run_workers(condition, components, seed, limit=None, duration=None):
        stop = threading.Event()
        gate = threading.Event()
        ready = threading.Barrier(len(components) + 1)
        condition_lock = threading.Condition()
        latest = {"tokens": seed, "base_id": -1}
        errors, records = [], {component: [] for component in components}

        def worker(component):
            nonlocal latest
            try:
                torch.set_num_threads(args.cpu_threads)
                stream = torch.cuda.Stream(device=device)
                with torch.cuda.device(device), torch.cuda.stream(stream), torch.no_grad():
                    seed.record_stream(stream)
                    for _ in range(args.warmup):
                        if stop.is_set():
                            return
                        if component == "base":
                            base_forward()
                        else:
                            deltas = edge_forward(seed)
                            finish_robot_command(deltas, inference)
                        stream.synchronize()
                    ready.wait()
                    gate.wait()
                    call_id = 0
                    while not stop.is_set() and (limit is None or call_id < limit):
                        if component == "base":
                            output, record = measured(
                                condition, component, call_id, base_forward, stream,
                            )
                            with condition_lock:
                                latest = {"tokens": output, "base_id": call_id}
                        else:
                            with condition_lock:
                                packet = latest
                            packet["tokens"].record_stream(stream)
                            output, record = measured(
                                condition, component, call_id,
                                lambda: edge_forward(packet["tokens"]), stream,
                            )
                            record["base_id"] = packet["base_id"]
                            command_record(output, record, stream)
                        records[component].append(record)
                        call_id += 1
            except Exception as exc:
                errors.append(exc)
                stop.set()
                ready.abort()
                gate.set()

        threads = [threading.Thread(target=worker, args=(name,), name=name)
                   for name in components]
        for thread in threads:
            thread.start()
        try:
            ready.wait()
            gate.set()
            if duration is not None:
                stop.wait(duration)
            else:
                for thread in threads:
                    thread.join()
        except threading.BrokenBarrierError:
            pass
        finally:
            stop.set()
            gate.set()
            for thread in threads:
                thread.join()
        if errors:
            raise RuntimeError(f"{condition} GPU worker failed.") from errors[0]
        for component, values in records.items():
            if not values:
                raise RuntimeError(f"No measured {component} calls in {condition}.")
            calls.extend(values)
            print(f"{condition}/{component}: {len(values)} completed calls", flush=True)
        return latest["tokens"]

    try:
        with torch.no_grad():
            seed = base_forward()
            torch.cuda.synchronize(device)
        torch.cuda.cudart().cudaProfilerStart()
        try:
            if args.mode in ("all", "isolated"):
                seed = run_workers("isolated", ("base",), seed, limit=args.repeats)
                run_workers("isolated", ("edge",), seed, limit=args.repeats)
            if args.mode in ("all", "async"):
                run_workers("async", ("base", "edge"), seed, duration=args.duration)
        finally:
            torch.cuda.synchronize(device)
            torch.cuda.cudart().cudaProfilerStop()
        result = {
            "metadata": {
                "gpu": torch.cuda.get_device_name(device), "batch_size": 1,
                "dtype": "bfloat16", "language_instruction": raw["language"],
                "goal_pose": raw["goal_pose"], "cpu_threads": args.cpu_threads,
            },
            "origin_ns": origin_ns,
            "input_policy": "Images, text tokens, and pose prepared on GPU before measurement.",
            "vision_policy": "Fresh encoder execution on every forward; no vision feature cache.",
            "base_policy": "KV cache disabled; async edge uses latest completed base tokens.",
            "calls": calls,
        }
        (args.output_dir / "dispatch.json").write_text(json.dumps(result, indent=2) + "\n")
    finally:
        for handle in handles:
            handle.remove()


if __name__ == "__main__":
    args = arguments()
    if args.trace_child:
        benchmark(args)
    else:
        launch(args)
