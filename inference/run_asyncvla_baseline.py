"""Fresh BaseVLM/EdgeAdaptor calls, isolated and concurrent, with GPU activity timing.

Run from the repository root: python inference/run_asyncvla_baseline.py
Requires Nsight Systems (nsys). latency_ms counts the union of each call's CUDA
kernels, memcpy, and memset intervals. CPU work and GPU idle gaps are excluded.
Concurrent component times can overlap and must not be added to obtain E2E time.
"""

import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import threading
from bisect import bisect_right
from collections import defaultdict
from datetime import datetime

WARMUP = 10
REPEATS = 100
DURATION = 30.0
PREFIX = "asyncvla/"


def activity_ms(intervals):
    total, previous = 0, 0
    for start, end in sorted(intervals):
        if end < start:
            raise ValueError("GPU activity ends before it starts.")
        total += max(0, end - max(start, previous))
        previous = max(previous, end)
    return total / 1e6


def read_latencies(database):
    """Match each GPU activity to the CUDA API inside its host-thread NVTX call."""
    with sqlite3.connect(f"{Path(database).resolve().as_uri()}?mode=ro", uri=True) as db:
        tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        ranges, calls = defaultdict(list), {}
        for start, end, tid, label in db.execute(
            "SELECT n.start, n.end, n.globalTid, COALESCE(n.text, s.value) "
            "FROM NVTX_EVENTS n LEFT JOIN StringIds s ON n.textId=s.id WHERE n.end IS NOT NULL"
        ):
            if label and label.startswith(PREFIX):
                if label in calls:
                    raise ValueError(f"Duplicate model call: {label}")
                calls[label] = []
                ranges[tid].append((start, end, label))
        starts = {}
        for tid, values in ranges.items():
            values.sort()
            if any(a[1] > b[0] for a, b in zip(values, values[1:])):
                raise ValueError("Model calls overlap on one host thread.")
            starts[tid] = [v[0] for v in values]
        launches = {}
        for table in ("CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_DRIVER"):
            if table not in tables:
                continue
            for start, end, tid, correlation in db.execute(
                f"SELECT start, end, globalTid, correlationId FROM {table}"
            ):
                if tid not in ranges:
                    continue
                index = bisect_right(starts[tid], start) - 1
                if index >= 0 and end <= ranges[tid][index][1]:
                    key = ((tid >> 24) & 0xFFFFFF, correlation)
                    label = ranges[tid][index][2]
                    if key in launches and launches[key] != label:
                        raise ValueError("A CUDA launch belongs to multiple model calls.")
                    launches[key] = label
        for kind in ("KERNEL", "MEMCPY", "MEMSET"):
            table = f"CUPTI_ACTIVITY_KIND_{kind}"
            if table not in tables:
                if kind == "KERNEL":
                    raise ValueError("The trace contains no CUDA kernel table.")
                continue
            for start, end, pid, correlation in db.execute(
                f"SELECT start, end, globalPid, correlationId FROM {table}"
            ):
                label = launches.get(((pid >> 24) & 0xFFFFFF, correlation))
                if label is not None:
                    calls[label].append((start, end))
    result = {}
    for mode in ("isolated", "async"):
        result[mode] = {}
        for component in ("base", "edge"):
            labels = [s for s in calls if s.startswith(f"{PREFIX}{mode}/{component}/")]
            labels.sort(key=lambda s: int(s.rsplit("/", 1)[1]))
            if not labels or (mode == "isolated" and len(labels) != REPEATS):
                raise ValueError(f"Incomplete calls: {mode}/{component}")
            if [int(s.rsplit("/", 1)[1]) for s in labels] != list(range(len(labels))):
                raise ValueError(f"Missing call IDs: {mode}/{component}")
            if any(not calls[s] for s in labels):
                raise ValueError(f"Missing GPU activity: {mode}/{component}")
            values = [activity_ms(calls[s]) for s in labels]
            result[mode][component] = {
                "latency_ms": values, "mean_latency_ms": sum(values) / len(values),
            }
    return result


def launch():
    nsys = shutil.which("nsys")
    if nsys is None:
        raise RuntimeError("Install the Nsight Systems CLI (nsys) before this benchmark.")
    output = Path("results_on_H100") / datetime.now().strftime("gpu_activity_%Y%m%d_%H%M%S")
    evidence = output / "evidence"
    evidence.mkdir(parents=True, exist_ok=False)
    trace = evidence / "gpu_trace"
    # This internal marker prevents recursive profiling. There are no CLI flags.
    env = dict(os.environ, ASYNCVLA_PROFILED="1")
    subprocess.run([
        nsys, "profile", "--trace=cuda,nvtx", "--sample=none", "--cpuctxsw=none",
        "--capture-range=cudaProfilerApi", "--capture-range-end=stop",
        "--output", str(trace), sys.executable, str(Path(__file__).resolve()),
    ], env=env, check=True)
    database = trace.with_suffix(".sqlite")
    if not database.exists():
        subprocess.run([nsys, "export", "--type", "sqlite", "--output", str(database),
                        str(trace.with_suffix(".nsys-rep"))], check=True)
    result = read_latencies(database)
    (output / "latency.json").write_text(json.dumps(result, indent=2) + "\n")
    for mode, components in result.items():
        for component, values in components.items():
            print(f"{mode}/{component}: {len(values['latency_ms'])} calls, "
                  f"mean GPU activity = {values['mean_latency_ms']:.3f} ms")
    print(f"Saved {output / 'latency.json'}", flush=True)


# The controller process needs only the standard library.
if __name__ == "__main__" and os.environ.get("ASYNCVLA_PROFILED") != "1":
    launch()
    sys.exit()

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import to_tensor
import yaml
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction_MMNv1
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.models.projectors import ProprioProjector
from prismatic.models.small_head import Edge_adapter, Proj_Actiontokens
from prismatic.training.train_utils import get_current_action_mask, get_next_actions_mask
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK, POSE_DIM

transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

def delta_to_pose(delta):
    """
    Fully differentiable, no inplace ops.
    delta: [N, T, 4]
    """
    dx = delta[..., 0]
    dy = delta[..., 1]
    dtheta = torch.atan2(delta[..., 3], delta[..., 2])
    N, T = dx.shape
    poses = []
    x = dx[:, 0]
    y = dy[:, 0]
    theta = dtheta[:, 0]
    poses.append(torch.stack([x, y, torch.cos(theta), torch.sin(theta)], dim=-1))
    for t in range(1, T):
        ct = torch.cos(theta)
        st = torch.sin(theta)
        dx_w = ct * dx[:, t] - st * dy[:, t]
        dy_w = st * dx[:, t] + ct * dy[:, t]
        x = x + dx_w
        y = y + dy_w
        theta = theta + dtheta[:, t]
        poses.append(torch.stack([x, y, torch.cos(theta), torch.sin(theta)], dim=-1))
    return torch.stack(poses, dim=1)

class Inference:

    def __init__(self, processor):
        self.processor = processor
        self.action_tokenizer = ActionTokenizer(processor.tokenizer)

    def collator_custom(self, instances, model_max_length, pad_token_id):
        IGNORE_INDEX = -100
        input_ids = pad_sequence([inst['input_ids'] for inst in instances], batch_first=True, padding_value=pad_token_id)
        labels = pad_sequence([inst['labels'] for inst in instances], batch_first=True, padding_value=IGNORE_INDEX)
        input_ids, labels = (input_ids[:, :model_max_length], labels[:, :model_max_length])
        attention_mask = input_ids.ne(pad_token_id)
        pixel_values = [inst['pixel_values_current'] for inst in instances]
        if 'dataset_name' in instances[0]:
            dataset_names = [inst['dataset_name'] for inst in instances]
        else:
            dataset_names = None
        if isinstance(pixel_values[0], torch.Tensor):
            if 'pixel_values_goal' in instances[0]:
                pixel_values_goal = [inst['pixel_values_goal'] for inst in instances]
                pixel_values = torch.cat((torch.stack(pixel_values), torch.stack(pixel_values_goal)), dim=1)
            else:
                pixel_values = torch.stack(pixel_values)
        else:
            raise ValueError(f'Unsupported `pixel_values` type: {type(pixel_values)}')
        actions = torch.stack([torch.from_numpy(np.copy(inst['actions'])) for inst in instances])
        goal_pose = torch.stack([torch.from_numpy(np.copy(inst['goal_pose'])) for inst in instances])
        output = dict(pixel_values=pixel_values.to(), input_ids=input_ids, attention_mask=attention_mask, labels=labels, actions=actions, goal_pose=goal_pose)
        if dataset_names is not None:
            output['dataset_names'] = dataset_names
        return output

    def transform_datatype(self, inst_obj, actions, goal_pose_cos_sin, current_image_PIL, goal_image_PIL, prompt_builder, action_tokenizer, base_tokenizer, image_transform, predict_stop_token=True):
        IGNORE_INDEX = -100
        current_action = actions[0]
        future_actions = actions[1:]
        future_actions_string = ''.join(action_tokenizer(future_actions))
        current_action_string = action_tokenizer(current_action)
        action_chunk_string = current_action_string + future_actions_string
        action_chunk_len = len(action_chunk_string)
        if inst_obj == 'xxxx':
            conversation = [{'from': 'human', 'value': 'No language instruction'}, {'from': 'gpt', 'value': action_chunk_string}]
        else:
            conversation = [{'from': 'human', 'value': f'What action should the robot take to {inst_obj}?'}, {'from': 'gpt', 'value': action_chunk_string}]
        prompt_builder = prompt_builder('openvla')
        for turn in conversation:
            prompt_builder.add_turn(turn['from'], turn['value'])
        input_ids = torch.tensor(base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids)
        labels = input_ids.clone()
        labels[:-(action_chunk_len + 1)] = IGNORE_INDEX
        if not predict_stop_token:
            labels[-1] = IGNORE_INDEX
        pixel_values_current = image_transform(current_image_PIL)
        pixel_values_goal = image_transform(goal_image_PIL)
        dataset_name = 'lelan'
        return dict(pixel_values_current=pixel_values_current, pixel_values_goal=pixel_values_goal, input_ids=input_ids, labels=labels, dataset_name=dataset_name, actions=torch.as_tensor(actions), goal_pose=goal_pose_cos_sin, img_PIL=current_image_PIL, inst=inst_obj)

    def pd_controller(self, actions, metric_waypoint_spacing):
        waypoints = actions.float().cpu().numpy()
        waypoint_select = 4
        chosen_waypoint = waypoints[0][waypoint_select].copy()
        chosen_waypoint[:2] *= metric_waypoint_spacing
        dx, dy, hx, hy = chosen_waypoint
        EPS = 1e-08
        DT = 1 / 3
        if np.abs(dx) < EPS and np.abs(dy) < EPS:
            linear_vel_value = 0
            angular_vel_value = 1.0 * clip_angle(np.arctan2(hy, hx)) / DT
        elif np.abs(dx) < EPS:
            linear_vel_value = 0
            angular_vel_value = 1.0 * np.sign(dy) * np.pi / (2 * DT)
        else:
            linear_vel_value = dx / DT
            angular_vel_value = np.arctan(dy / dx) / DT
        linear_vel_value = np.clip(linear_vel_value, 0, 0.5)
        angular_vel_value = np.clip(angular_vel_value, -1.0, 1.0)
        maxv, maxw = (0.3, 0.3)
        if np.abs(linear_vel_value) <= maxv:
            if np.abs(angular_vel_value) <= maxw:
                linear_vel_value_limit = linear_vel_value
                angular_vel_value_limit = angular_vel_value
            else:
                rd = linear_vel_value / angular_vel_value
                linear_vel_value_limit = maxw * np.sign(linear_vel_value) * np.abs(rd)
                angular_vel_value_limit = maxw * np.sign(angular_vel_value)
        elif np.abs(angular_vel_value) <= 0.001:
            linear_vel_value_limit = maxv * np.sign(linear_vel_value)
            angular_vel_value_limit = 0.0
        else:
            rd = linear_vel_value / angular_vel_value
            if np.abs(rd) >= maxv / maxw:
                linear_vel_value_limit = maxv * np.sign(linear_vel_value)
                angular_vel_value_limit = maxv * np.sign(angular_vel_value) / np.abs(rd)
            else:
                linear_vel_value_limit = maxw * np.sign(linear_vel_value) * np.abs(rd)
                angular_vel_value_limit = maxw * np.sign(angular_vel_value)
        return (linear_vel_value_limit, angular_vel_value_limit)

def clip_angle(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))

def resize_rgb(image):
    return image.convert('RGB').resize((224, 224), Image.BILINEAR)

class BaseVLM(nn.Module):
    """Raw images, goal pose, and language -> projected base tokens."""

    def __init__(self, vla, pose_projector, action_proj, inference,
                 device, num_patches):
        super().__init__()
        self.vla = vla
        self.pose_projector = pose_projector
        self.action_proj = action_proj
        self.io = inference
        self.device = device
        self.num_patches = num_patches
        self.vla.config.use_cache = False
        self.vla.language_model.config.use_cache = False

    @torch.no_grad()
    def prepare_inputs(self, raw):
        # Rebuild the prompt, tokenize, and transform both images on every call.
        instruction = raw["language"] if raw["modality_id"] in (7, 8) else "xxxx"
        instance = self.io.transform_datatype(
            instruction,
            np.zeros((NUM_ACTIONS_CHUNK, ACTION_DIM)),  # Placeholder actions only.
            np.asarray(raw["goal_pose"], dtype=np.float64),
            resize_rgb(raw["base_image"]),
            resize_rgb(raw["goal_image"]),
            prompt_builder=PurePromptBuilder,
            action_tokenizer=self.io.action_tokenizer,
            base_tokenizer=self.io.processor.tokenizer,
            image_transform=self.io.processor.image_processor.apply_transform,
        )
        batch = self.io.collator_custom(
            [instance],
            model_max_length=self.io.processor.tokenizer.model_max_length,
            pad_token_id=self.io.processor.tokenizer.pad_token_id,
        )
        batch = {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
        modality = torch.tensor(
            [raw["modality_id"]], dtype=torch.bfloat16, device=self.device
        )

        batch["pixel_values"] = batch["pixel_values"].to(torch.bfloat16)
        batch["goal_pose"] = batch["goal_pose"].to(torch.bfloat16)
        return batch, modality

    @torch.no_grad()
    def forward(self, raw):
        batch, modality = self.prepare_inputs(raw)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = self.vla(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                pixel_values=batch["pixel_values"],
                modality_id=modality,
                labels=batch["labels"],
                output_hidden_states=True,
                proprio=batch["goal_pose"],
                proprio_projector=self.pose_projector,
                use_film=False,
                use_cache=False,
                past_key_values=None,
            )
        if output.past_key_values is not None:
            raise RuntimeError("BaseVLM returned a KV cache while caching was disabled.")

        # Preserve the original label/hidden-state alignment.
        token_ids = batch["labels"][:, 1:]
        mask = get_current_action_mask(token_ids) | get_next_actions_mask(token_ids)
        hidden = output.hidden_states[-1][:, self.num_patches:-1]
        hidden = hidden[mask].reshape(
            batch["input_ids"].shape[0], NUM_ACTIONS_CHUNK * ACTION_DIM, -1
        ).to(torch.bfloat16)
        return self.action_proj.predict_action(hidden, modality)


class EdgeAdaptor(nn.Module):
    """Fresh current/past image encoders and base tokens -> action-head deltas."""

    def __init__(self, shead, device):
        super().__init__()
        self.shead, self.device = shead, device

    def prepare_image(self, image):
        small = TF.resize(to_tensor(resize_rgb(image)), (96, 96)).unsqueeze(0)
        return transform(small).to(device=self.device, dtype=torch.bfloat16)

    @torch.no_grad()
    def forward(self, current_image, past_image, base_tokens):
        return self.shead(self.prepare_image(current_image),
                          self.prepare_image(past_image), base_tokens)


def run_base(base, raw):
    return base(raw)


def run_edge(edge, raw, tokens):
    return edge(raw["current_image"], raw["base_image"], tokens)


def finish_robot_command(deltas, inference):
    return inference.pd_controller(delta_to_pose(deltas).cpu(), metric_waypoint_spacing=0.1)


def define_model():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    checkpoint = Path("AsyncVLA_release")
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction_MMNv1)
    processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        checkpoint, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
    ).to(device)
    vla.vision_backbone.set_num_images_in_input(2)

    def load(module, name, bf16=True):
        path = checkpoint / f"{name}--750000_checkpoint.pt"
        if name == "pose_projector" and not path.exists():
            path = checkpoint / "proprio_projector--750000_checkpoint.pt"
        state = torch.load(path, map_location="cpu")
        module.load_state_dict({k.removeprefix("module."): v for k, v in state.items()})
        return module.to(device=device, dtype=torch.bfloat16 if bf16 else torch.float32)

    pose = load(ProprioProjector(llm_dim=vla.llm_dim, proprio_dim=POSE_DIM),
                "pose_projector", bf16=False)
    projection = load(Proj_Actiontokens(input_dim=vla.llm_dim, hidden_dim=vla.llm_dim,
                                       action_dim=1024), "action_proj")
    with open("config_nav/dataset_config.yaml") as file:
        config = yaml.safe_load(file)
    shead = load(Edge_adapter(**{key: config[key] for key in (
        "obs_encoding_size", "mha_num_attention_heads", "mha_num_attention_layers", "mha_ff_dim_factor",
    )}), "shead")
    inference = Inference(processor)
    patches = vla.vision_backbone.get_num_patches() * 2 + 1
    base = BaseVLM(vla, pose, projection, inference, device, patches).eval()
    edge = EdgeAdaptor(shead, device).eval()
    return base, edge, inference, device


def run_workers(mode, components, base, edge, raw, inference, seed, device):
    stop, gate = threading.Event(), threading.Event()
    ready, lock = threading.Barrier(len(components) + 1), threading.Lock()
    latest, errors = [seed], []

    def worker(component):
        try:
            stream = torch.cuda.Stream(device=device)
            with torch.cuda.device(device), torch.cuda.stream(stream), torch.no_grad():
                seed.record_stream(stream)
                for _ in range(WARMUP):
                    if component == "base":
                        run_base(base, raw)
                    else:
                        finish_robot_command(run_edge(edge, raw, seed), inference)
                    stream.synchronize()
                ready.wait()
                gate.wait()
                call_id = 0
                while not stop.is_set() and (mode == "async" or call_id < REPEATS):
                    with lock:
                        tokens = latest[0]
                    tokens.record_stream(stream)
                    # Include input copies and fresh encoders. Synchronization stays outside the range.
                    with torch.cuda.nvtx.range(f"{PREFIX}{mode}/{component}/{call_id}"):
                        output = (run_base(base, raw) if component == "base"
                                  else run_edge(edge, raw, tokens))
                    stream.synchronize()
                    if component == "base":
                        with lock:
                            latest[0] = output
                    else:
                        finish_robot_command(output, inference)
                    call_id += 1
        except Exception as exc:
            errors.append(exc)
            stop.set()
            ready.abort()
            gate.set()

    threads = [threading.Thread(target=worker, args=(component,)) for component in components]
    for thread in threads:
        thread.start()
    try:
        ready.wait()
        gate.set()
        if mode == "async":
            stop.wait(DURATION)
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
        raise RuntimeError(f"{mode} worker failed") from errors[0]
    return latest[0]


def benchmark():
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires a CUDA GPU.")
    base, edge, inference, device = define_model()

    def image(name):
        with Image.open(Path("inference") / name) as source:
            return source.convert("RGB")

    raw = {
        "base_image": image("past.png"), "current_image": image("cur.png"),
        "goal_image": image("goal.png"),
        "goal_pose": [10.0, -100.0, np.cos(-90 / 180 * 3.1415), np.sin(-90 / 180 * 3.1415)],
        "language": "move toward blue trash bin", "modality_id": 4,
    }
    seed = run_base(base, raw)
    torch.cuda.synchronize(device)
    torch.cuda.cudart().cudaProfilerStart()
    try:
        seed = run_workers("isolated", ("base",), base, edge, raw, inference, seed, device)
        run_workers("isolated", ("edge",), base, edge, raw, inference, seed, device)
        run_workers("async", ("base", "edge"), base, edge, raw, inference, seed, device)
    finally:
        torch.cuda.synchronize(device)
        torch.cuda.cudart().cudaProfilerStop()


if __name__ == "__main__":
    benchmark()
