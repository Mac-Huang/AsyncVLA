"""
1. Define the components: In this case, we use modality 8 which is a combination of current image, pose and text
    a. Data Pre-processing Component (CPU): Image Processing + Pose + Prompt Tokenizer
    b. Vision Projection Component (on GPU): A. SigLIP B. Eff. Net
    c. Goal Pose Projection Component (on GPU): Base only
    d. Deep Learning Component (on GPU): A. Llama 2 B. Transformer
    e. Connection Component (on GPU):   
        A. Token projection 
        B. Action Projection; Such split is because different designs have different projection methods, and we want to benchmark the projection time separately.
        C. Trajectory Conversion (Transformer -> Action Head)
    f. Robot Trajectory (on CPU): Action Head -> Control Command
2. Build isolated / concurrent runs of components: make main function looks easy and readable
    a. Isolated run: 
        Run each component one by one on designated hardware
        Run each component for a fixed number of times, and record the time spent on each component
    b. Concurrent run: Run all components concurrently on designated hardwares
        Need to figure out how to record the time spent on each component
        Use CUDA Graph to optimize the concurrent run (can it record time via CUDA Graph interface?)
        Run the whole pipline for a fixed duration, and record the time spent on each component
3. Record the time spent on each component and the end-to-end time:
    I wanna test isolated run and concurrent run, and see if the concurrent run has higher latency than the isolated run,
    and figure out which component is the bottleneck in the concurrent run
"""

import functools
import math
from pathlib import Path
import torch
import run_asyncvla_v0 as original
import numpy as np
import threading
import time
import json

from PIL import Image

from inference.components_timing import measure_call, save_results, _summary

def load_data():
    image_dir = Path(__file__).resolve().parent

    def read_image(filename):
        with Image.open(image_dir / filename) as img:
            return img.convert("RGB")

    # In this script, we are using modality 8 which is a combination of current image, pose and text
    # But the data loading function is designed to be flexible and can be easily modified to load different modalities
    yaw = -90.0 / 180.0 * 3.1415

    return {
        "base_image": read_image("past.png"),
        "current_image": read_image("cur.png"),
        "goal_image": read_image("goal.png"),
        "language": "move toward blue trash bin",
        "goal_pose": [
            10.0,
            -100.0,
            math.cos(yaw),
            math.sin(yaw),
        ],
        "modality_id": 8,
    }

def load_models():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")

    # use original inference configuration
    cfg = original.InferenceConfig()

    # reuse original inputs even though most are not able to use
    (
        vla, _action_head, pose_projector, shead, action_proj,
        device, num_patches, action_tokenizer, processor,
    ) = original.define_model(cfg)

    for module in (vla, pose_projector, shead, action_proj):
        module.eval()

    return {
        "vla": vla,
        "pose_projector": pose_projector,
        "shead": shead,
        "action_proj": action_proj,
        "action_predictor": shead.action_predictor,
        "device": device,
        "num_patches": num_patches,
        "action_tokenizer": action_tokenizer,
        "processor": processor,
    }

def base_preprocess_inputs(raw, models):
    """Data Pre-processing Component (CPU): Image Pre-processing + Pose + Prompt Tokenizer"""
    processor = models["processor"]
    action_tokenizer = models["action_tokenizer"]

    # Align to the figure inputs: from left to right
    # Image pre-processing
    image_transform = processor.image_processor.apply_transform # give proccess func a shorter name
    base_image = raw["base_image"].resize((224, 224), Image.BILINEAR)
    goal_image = raw["goal_image"].resize((224, 224), Image.BILINEAR)

    pixel_values = torch.cat([image_transform(base_image), image_transform(goal_image)], dim=0).unsqueeze(0).to(torch.bfloat16)

    # Goal Pose
    goal_pose = torch.tensor([raw["goal_pose"]], dtype=torch.bfloat16)

    # Prompt
    prompt = original.PurePromptBuilder("openvla")
    prompt.add_turn("human", f"What action should the robot take to {raw['language']}?")

    # Actions bit (purple hidden bits in figure)
    actions = np.zeros((original.NUM_ACTIONS_CHUNK, original.ACTION_DIM))
    action_tokens = (action_tokenizer(actions[0]) + "".join(action_tokenizer(actions[1:])))

    ## Combine Prompt and Action tokens for later merging with Vision
    prompt.add_turn("gpt", action_tokens)
    input_ids = torch.tensor(processor.tokenizer(prompt.get_prompt(), add_special_tokens=True,).input_ids, dtype=torch.long)
    ## To locate the action token
    labels = input_ids.clone()
    labels[:-(len(action_tokens) + 1)] = -100 # ignore the tokens (other hidden states) except last N action tokens, handing over to edge later
    max_length = processor.tokenizer.model_max_length
    input_ids = input_ids[:max_length].unsqueeze(0)
    labels = labels[:max_length].unsqueeze(0)

    return {
        "input_ids": input_ids,
        "attention_mask": input_ids.ne(processor.tokenizer.pad_token_id),
        "labels": labels,
        "pixel_values": pixel_values,
        "goal_pose": goal_pose
    }

def base_move_inputs_to_gpu(inputs, models, modality_id):
    """PCIe transfer on server"""

    # TODO: need to check shared memory on Jetson, whether it's free copy or what

    batch = {}

    # Transfer input_ids (tokens), pixel_values and goal_posed (non-tokens)
    for name, tensor in inputs.items():
        batch[name] = tensor.to(models["device"], non_blocking=True)

    batch["modality_id"] = torch.tensor([modality_id], dtype=torch.bfloat16, device=models["device"])

    return batch

@torch.no_grad()
def base_vision_projection(batch, models):
    """SigLIP + DINOv2"""

    # original repo implementation
    with torch.autocast("cuda", dtype=torch.bfloat16): # for mixing data with different precision
        return models["vla"]._process_vision_features(batch["pixel_values"], use_film=False)

@torch.no_grad()
def base_pose_projection(batch, vision_tokens, models):
    """Project the goal pose and append one token to the image embeddings."""    
    vla = models["vla"]

    with torch.autocast("cuda", dtype=torch.bfloat16):
        vision_pose_tokens = vla._process_proprio_features(vision_tokens, batch["goal_pose"], models["pose_projector"])

    return vision_pose_tokens

@torch.no_grad()
def base_prepare_llama_inputs(batch, vision_pose_embeddings, models):
    vla = models["vla"]

    with torch.autocast("cuda", dtype=torch.bfloat16):
        # Convert prompt token IDs into embedding vectors
        prompt_embeddings = vla.get_input_embeddings()(batch["input_ids"])

        action_mask = vla._process_action_masks(batch["labels"])
        # AND 0s to zero the placeholder/action embedding
        prompt_embeddings = (prompt_embeddings * ~action_mask.unsqueeze(-1))

        # Prepare the complete embeddings for base Llama model
        embeddings, attention_mask = vla._build_multimodal_attention_MMN(
            input_embeddings=prompt_embeddings,
            projected_patch_embeddings=vision_pose_embeddings,
            attention_mask=batch['attention_mask'],
            attention_mask_label=None,
            modality_id=batch["modality_id"]
        )

        # Also need to pass the label to find the action tokens after inference
        labels = vla._build_multimodal_labels(batch["labels"], vision_pose_embeddings)

        return {
            "inputs_embeds": embeddings,
            "attention_mask": attention_mask,
            "labels": labels
        }

@torch.no_grad()
def base_select_action_tokens(last_hidden_states, batch, models):
    target_ids = batch["labels"][:, 1:]

    action_mask = (original.get_current_action_mask(target_ids) | original.get_next_actions_mask(target_ids))

    aligned_hidden_states = last_hidden_states[:, models["num_patches"]:-1]

    batch_size = batch["input_ids"].shape[0]
    action_count = original.NUM_ACTIONS_CHUNK * original.ACTION_DIM
    hidden_size = last_hidden_states.shape[-1]

    action_hidden_states = aligned_hidden_states[action_mask].reshape(batch_size, action_count, hidden_size)

    return action_hidden_states.to(torch.bfloat16)

@torch.no_grad()
def base_llama_execution(llama_inputs, models):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = models["vla"].language_model(
            input_ids=None,
            inputs_embeds=llama_inputs["inputs_embeds"],
            attention_mask=llama_inputs["attention_mask"],
            labels=llama_inputs["labels"],

            # Disable the KV Cache for measurement to fake each time is a new image, pose and prompt
            past_key_values=None,
            use_cache=False,

            output_hidden_states=True,
            output_attentions=False,
            return_dict=True,
        )

        # Return every sequence position from the final Llama layer.
        # [batch, total_sequence_length, Llama_hidden_size].
        return output.hidden_states[-1]

@torch.no_grad()
def base_token_projection(action_hidden_states, batch, models):

    return models["action_proj"].predict_action(action_hidden_states, batch["modality_id"])

def edge_preprocess_inputs(raw):

    def prepare_image(image):
        image = image.resize((224, 224), Image.BILINEAR)

        tensor = original.to_tensor(image)
        tensor = original.TF.resize(tensor, (96, 96))

        # add batch dimention: [3, 96, 96] -> [1, 3, 96, 96]
        tensor = tensor.unsqueeze(0)

        return original.transform(tensor).to(torch.bfloat16)

    return {"current_image": prepare_image(raw["current_image"]), "past_image": prepare_image(raw["base_image"])}

def edge_move_inputs_to_gpu(inputs, models):
    # Both images already use BF16 from CPU preprocessing.
    return {
        name: tensor.to(models["device"], non_blocking=True)
        for name, tensor in inputs.items()
    }

@torch.no_grad()
def edge_vision_projection(edge_batch, models):
    shead = models["shead"]
    current = edge_batch["current_image"]
    past = edge_batch["past_image"]

    ## Two Paths in original repo: One for current image and the other one for current + past image
    # Current Image                                                         # [B, 3, 96, 96]
    current_encoding = shead.obs_encoder.extract_features(current)          # [B, 1280, 3, 3]
    current_encoding = shead.obs_encoder._avg_pooling(current_encoding)     # [B, 1280, 1, 1]
    if shead.obs_encoder._global_params.include_top:
        current_encoding = current_encoding.flatten(start_dim=1)            # [B, 1280]
        current_encoding = shead.obs_encoder._dropout(current_encoding)

    current_encoding = shead.compress_obs_enc(current_encoding)             # [B, 1024]

    # Paired Images
    paired_image = torch.cat((current, past), dim=1)                        # [B, 6, 96, 96]
    paired_encoding = shead.cat_encoder.extract_features(paired_image)      # [B, 1280, 3, 3]
    paired_encoding = shead.cat_encoder._avg_pooling(paired_encoding)       # [B, 1280, 1, 1]
    if shead.cat_encoder._global_params.include_top:
        paired_encoding = paired_encoding.flatten(start_dim=1)              # [B, 1280]
        paired_encoding = shead.cat_encoder._dropout(paired_encoding)

    paired_encoding = shead.compress_cat_enc(paired_encoding)               # [B, 1024]

    return torch.stack((current_encoding, paired_encoding), dim=1)

@torch.no_grad()
def edge_prepare_transformer_inputs(base_features, edge_vision_tokens):
    # [batch, 8 + 2, 1024]
    return torch.cat((base_features, edge_vision_tokens), dim=1)

@torch.no_grad()
def edge_transformer_execution(edge_inputs, models):
    output = models["shead"].decoder(edge_inputs)

    # [batch, 8 + 2, 1024] -> [batch, 1, 1024]
    selected_token = output[:, -2: -1, :]   # Select the current-image token at index 8 (the ninth token).

    return selected_token.reshape(selected_token.shape[0], -1)

@torch.no_grad()
def edge_action_projection(edge_hidden_states, task_id, models):
    # Edge_adapter_v0 appends the numeric task ID to the feature/output hidden dimention
    # [batch, 1024] -> [batch, 1025]
    predictor_inputs = torch.cat((edge_hidden_states, task_id.unsqueeze(1)), dim=1)

    action_deltas = models["action_predictor"](predictor_inputs)

    # [batch, 8, 4] 8 actions and 4 hidden dim
    return action_deltas.reshape(edge_hidden_states.shape[0], original.NUM_ACTIONS_CHUNK, original.ACTION_DIM)

def edge_trajectory_conversion(action_deltas):
    return original.delta_to_pose(action_deltas)

def edge_move_trajectory_to_cpu(trajectory_gpu):
    return trajectory_gpu.cpu()

def edge_controller_commands(trajectory_cpu, metric_waypoint_spacing=0.1):
    """Same logic from original repo, but fixed bug `NameError: name 'clip_angle' is not defined`
    This function wrote by Codex"""

    # Convert the CPU trajectory to float32 for NumPy.
    waypoints = trajectory_cpu.float().numpy()

    # Select waypoint index 4 and convert its position to meters.
    waypoint = waypoints[0, 4].copy()
    waypoint[:2] *= metric_waypoint_spacing
    dx, dy, hx, hy = waypoint

    # Preserve the original controller's fixed timestep and zero threshold.
    eps, dt = 1e-8, 1 / 3

    if np.abs(dx) < eps and np.abs(dy) < eps:
        linear = 0.0

        # Wrap the desired heading into [-pi, pi].
        angle = np.arctan2(hy, hx)
        angular = np.arctan2(np.sin(angle), np.cos(angle)) / dt
    elif np.abs(dx) < eps:
        linear = 0.0
        angular = np.sign(dy) * np.pi / (2 * dt)
    else:
        linear = dx / dt
        angular = np.arctan(dy / dx) / dt

    # Preserve the original initial command limits.
    linear = np.clip(linear, 0.0, 0.5)
    angular = np.clip(angular, -1.0, 1.0)

    # Apply the original final limits while preserving turning curvature.
    maxv, maxw = 0.3, 0.3

    if np.abs(linear) <= maxv:
        if np.abs(angular) <= maxw:
            return float(linear), float(angular)

        radius = linear / angular
        return (
            float(maxw * np.sign(linear) * np.abs(radius)),
            float(maxw * np.sign(angular)),
        )

    # Preserve the original special case for nearly straight motion.
    if np.abs(angular) <= 0.001:
        return float(maxv * np.sign(linear)), 0.0

    radius = linear / angular
    if np.abs(radius) >= maxv / maxw:
        return (
            float(maxv * np.sign(linear)),
            float(maxv * np.sign(angular) / np.abs(radius)),
        )

    return (
        float(maxw * np.sign(linear) * np.abs(radius)),
        float(maxw * np.sign(angular)),
    )

@torch.no_grad()
def run_base_once(raw, models, stage):
    ## Base Module
    # Data Pre-processing Component (CPU): Image Processing + Pose + Prompt Tokenizer
    inputs = stage("base_preprocess", lambda: base_preprocess_inputs(raw, models), wait_for_gpu=False)

    # Data Transfer: copy inputs to GPU
    batch = stage("base_h2d", lambda: base_move_inputs_to_gpu(inputs, models, raw["modality_id"]))

    ## So far, all data for base all pre-proccessed and transferrd to GPU memory
    ## Now it has Vision/Pose Tensors + Prompt/Action Tokens; next project vision/pose to token and build the final input to VLM
    
    # Base Vision Projection Component (on GPU): SigLIP + DINOv2
    base_vision_tokens = stage("base_vision", lambda: base_vision_projection(batch, models))

    # Base Goal Pose Projection Component + Base Vision Embedding (instead of tokens because of repos api design)
    vision_pose_embeddings = stage("base_pose", lambda: base_pose_projection(batch, base_vision_tokens, models))

    # Combine all base llama input tokens
    llama_inputs = stage("base_input_assembly", lambda: base_prepare_llama_inputs(batch, vision_pose_embeddings, models))

    # Run the llama
    last_hidden_states = stage("base_llama", lambda: base_llama_execution(llama_inputs, models))
    action_hidden_states = stage("base_action_selection", lambda: base_select_action_tokens(last_hidden_states, batch, models))

    ## After get the tokens from Base, they need to be projected to Edge input
    # Connection Component
    base_features = stage("base_token_projection", lambda: base_token_projection(action_hidden_states, batch, models))

    return base_features, batch["modality_id"]

@torch.no_grad()
def run_edge_once(raw, models, base_features, task_id, stage):

    ## Edge Module
    # Data Pre-processing Component (on CPU): Image Preprocessing
    edge_cpu_inputs = stage("edge_preprocess", lambda: edge_preprocess_inputs(raw), wait_for_gpu=False)

    # Data Transfer: copy inputs to GPU
    edge_batch = stage("edge_h2d", lambda: edge_move_inputs_to_gpu(edge_cpu_inputs, models))

    # Edge Vision Projection Component (On GPU): Efficient Net
    edge_vision_tokens = stage("edge_vision", lambda: edge_vision_projection(edge_batch, models))

    # Combine edge input tokens
    edge_tokens = stage("edge_input_assembly", lambda: edge_prepare_transformer_inputs(base_features, edge_vision_tokens))

    # Edge Transformer
    edge_hidden_states = stage("edge_transformer", lambda: edge_transformer_execution(edge_tokens, models))

    ## Robot Control Module (I assume this stage can be overlapping with Edge/Base, as well as CPU preprocessing;
    ## and there are CPU/memory bandwidth contention potentially
    # Action Head
    action_deltas = stage("edge_action_projection", lambda: edge_action_projection(edge_hidden_states, task_id, models))

    # Trajectory Conversion
    trajectory_gpu = stage("edge_trajectory", lambda: edge_trajectory_conversion(action_deltas))

    # Transfer completed waypoints to CPU (blocking the CPU controller in next step)
    trajectory_cpu = stage("edge_d2h", lambda: edge_move_trajectory_to_cpu(trajectory_gpu), wait_for_gpu=False)

    # Calculate velocity on CPU
    command = stage("controller", lambda: edge_controller_commands(trajectory_cpu), wait_for_gpu=False)

    return command

def run_sequencial(raw, models, timings):
    """Run once and measure each component alone"""

    def stage(name, function, wait_for_gpu=True):
        if timings is None:
            return function()

        output, record = measure_call(name, function, models["device"], wait_for_gpu=wait_for_gpu)
        timings.append(record)

        return output

    base_features, task_id = run_base_once(raw, models, stage)
    return run_edge_once(raw, models, base_features, task_id, stage)

def run_concurrent(raw, models, timings, duration=30.0, warmup=10):
    """Run two workers: one for base, and one for edge"""
    device = models["device"]
    # for preparation and line up after warming up
    stop = threading.Event()            # stop signal for worker
    gate = threading.Event()            # start signal for worker
    ready = threading.Barrier(3)        # meeting point for 3 threads after warming up
    lock = threading.Lock()
    warmup_lock = threading.Lock()  # Setup only. Measured execution stays concurrent.
    errors = []
    samples = {"base": {}, "edge": {}}  # timing for components (Vision/LLM/...)
    calls = {"base": [], "edge": []}    # timing for calls (Base/Edge)

    def direct(name, function, wait_for_gpu=True):
        """Replace stage() function to directly run the components without time recording for warmups"""
        return function()

    # Run base once to get a result for edge
    features, task_id = run_base_once(raw, models, direct)
    torch.cuda.synchronize(device)
    latest = [{     # get first base and update it with lock later
        "features": features,
        "task_id": task_id,
        "base_image": raw["base_image"],
        "call_id": -1,
        "ready_ns": time.perf_counter_ns(),
    }]

    # acquire() and publish() are for concurrently updating the latest
    # data from base for edge
    def acquire():
        with lock:
            return latest[0]

    def publish(packet):
        with lock:
            previous = latest[0]
            latest[0] = packet
        del previous

    # Create both streams before any worker can enter automatic graph capture.
    streams = {name: torch.cuda.Stream(device=device) for name in ("base", "edge")}

    def worker(component):
        try:
            stream = streams[component]
            with torch.cuda.device(device), torch.cuda.stream(stream), torch.no_grad():

                def stage(name, function, wait_for_gpu=True):
                    # Measure the host call without adding a GPU wait.
                    # Operations inside function() may still block.
                    start_ns = time.perf_counter_ns()
                    output = function()
                    elapsed_ms = (time.perf_counter_ns() - start_ns) / 1e6
                    samples[component].setdefault(name, []).append(elapsed_ms)

                    return output

                def edge_call(packet, timer):
                    packet["features"].record_stream(stream)
                    packet["task_id"].record_stream(stream)
                    edge_raw = dict(raw, base_image=packet["base_image"])
                    return run_edge_once(edge_raw, models, packet["features"], packet["task_id"], timer)

                # Compilation is lazy. Warm up on each job's own CUDA stream.
                # Keep capture separate from the other job's GPU work during setup.
                with warmup_lock:
                    for _ in range(warmup):
                        if component == "base":
                            output = run_base_once(raw, models, direct)
                        else:
                            output = edge_call(acquire(), direct)
                        stream.synchronize()
                        del output

                ready.wait()
                gate.wait()

                # Concurrent Run
                call_id = 0
                while not stop.is_set():
                    start_ns = time.perf_counter_ns()
                    if component == "base":
                        output_features, output_task = run_base_once(raw, models, stage)
                        stream.synchronize()
                        publish({
                            "features": output_features,
                            "task_id": output_task,
                            "base_image": raw["base_image"],
                            "call_id": call_id,
                            "ready_ns": time.perf_counter_ns(),
                        })
                    else:
                        packet = acquire()
                        acquire_ns = time.perf_counter_ns()

                        # Reuse the latest completed base result while base run again
                        command = edge_call(packet, stage)

                    finish_ns = time.perf_counter_ns()
                    calls[component].append({
                        "call_id": call_id,
                        "wall_ms": (finish_ns - start_ns) / 1e6,
                    })
                    call_id += 1
        except Exception as error:
            with lock:
                errors.append(error)
            stop.set()      # stop it
            ready.abort()   # abort the ready list
            gate.set()      # release the workers still working

    # Create CPU Threading
    threads = [
        threading.Thread(target=worker, args=("base",), name="base",),
        threading.Thread(target=worker, args=("edge",), name="edge",),
    ]

    started = []
    try:
        for thread in threads:
            thread.start()
            started.append(thread)
        ready.wait()        # wait for both workers finish warmup
        gate.set()          # allow to start measurement
        stop.wait(duration) # wait for running until duration time up

    except threading.BrokenBarrierError:
        pass

    finally:
        stop.set()          # stop it
        # Release warmup waiters if startup stopped before the gate opened.
        if not gate.is_set():
            ready.abort()
        gate.set()
        for thread in started:
            thread.join()

    if errors:
        raise RuntimeError("A concurrent worker failed.") from errors[0]

    # GPU component samples describe host calls. Base/Edge calls include completion.
    records = []
    for component in ("base", "edge"):
        for name, values in samples[component].items():
            records.append({
                "component": name,
                "host_call_ms": _summary(values)
            })

        if calls[component]:
            records.append({
                "component": f"{component}_call",
                "wall_ms": _summary([call["wall_ms"] for call in calls[component]])
            })

    timings.extend(records)


def compile_gpu_components():
    """Enable compilation and automatic CUDA Graphs for GPU compute components."""
    names = (
        "base_vision_projection",
        "base_pose_projection",
        "base_prepare_llama_inputs",
        "base_llama_execution",
        "base_select_action_tokens",
        "base_token_projection",
        "edge_vision_projection",
        "edge_prepare_transformer_inputs",
        "edge_transformer_execution",
        "edge_action_projection",
        "edge_trajectory_conversion",
    )

    def wrap(function):
        compiled = torch.compile(
            function,
            mode="reduce-overhead",
            dynamic=False,  # This benchmark repeats fixed input shapes.
            # Allow unsupported Python sections to remain eager.
            # fullgraph=True would fail on the current Base branches and masks.
            fullgraph=False,
        )

        @functools.wraps(function)
        def call(*args, **kwargs):
            output = compiled(*args, **kwargs)
            # Clone outside the compiled region. Later calls may reuse graph storage.
            # In particular, Edge must retain the previous Base features safely.
            if isinstance(output, dict):
                return {
                    key: value.clone() if isinstance(value, torch.Tensor) else value
                    for key, value in output.items()
                }
            return output.clone()

        return call

    # Install once before the seed Base call and worker warmup.
    # CPU preprocessing, CPU control, and transfer functions remain unchanged.
    for name in names:
        globals()[name] = wrap(globals()[name])
    return list(names)


def main():
    # load images, pose and language prompt as well as modality
    raw = load_data()
    # get implemented modules from original repo
    models = load_models()

    # Use the same setting for isolated and concurrent comparisons.
    use_compile = True
    compiled_components = compile_gpu_components() if use_compile else []

    timings = []
    mode = "concurrent"
    if mode == "isolated":
        # run each components alone
        run_sequencial(raw, models, timings=timings)
    elif mode == "concurrent":
        run_concurrent(raw, models, timings=timings, duration=30.0, warmup=10)
    
    # Compilation requests automatic graphs. GPU capture coverage is not verified here.
    result_mode = f"{mode}_compiled" if use_compile else mode
    output = save_results(
        timings, models["device"], mode=result_mode,
        compiled_components=compiled_components,
    )
    print(output.resolve())



if __name__ == "__main__":
    main()











# def main():
#     if flag == 'isolated':
#         for component in components:
#             run_isolated(component)
#     elif flag == 'concurrent':
#         run_concurrent(components)

# def run_isolated(component):
#     for i in range(repeats):
#         start_time = time.time()
#         # this is a placeholder for the actual function calls to run each component
#         # current is too naive
#         if component == 'data_preprocessing':
#             run_data_preprocessing(raw_current_image, raw_past_image, raw_text)
#         elif component == 'vision_projection_base':
#             run_vision_projection_base()
#         elif component == 'vision_projection_edge':
#             run_vision_projection_edge()
#         elif component == 'deep_learning_base':
#             run_deep_learning_base()
#         elif component == 'deep_learning_edge':
#             run_deep_learning_edge()
#         elif component == 'connection':
#             run_connection()
#         end_time = time.time()
#         record_time(component, end_time - start_time)

# def run_concurrent(components):
#     """All components running currently on designated hardwares, and the data stream is pipelined,
#     where there shouldn't have a situation where all components are waiting for data to be processed"""

#     # All the time recording could be a problem cuz they are running concurrently, and the time spent on each component should be overlapped with other components.

#     e2e_start_time = time.time()
#     # Fetch the raw data
#     raw_current_image, raw_past_image, raw_text = load_data()

#     # Data Pre-processing Component (CPU): Image Processing (SigLIP + DINOv2 / Eff. Net) + Tokenizer
#     preprocessing_start_time = time.time()
#     # This should be run with multiprocessing or threading on CPU and all data shouldn't be finished at the same time i assume
#     # Once one data is preprocessed, it can be sent to the GPU for the next components
#     current_image, past_image, text_token = run_data_preprocessing(raw_current_image, raw_past_image, raw_text)
#     preprocessing_end_time = time.time() # should be the last data finished preprocessed time, not the first one
#     record_time('data_preprocessing', preprocessing_end_time - preprocessing_start_time)

#     # Run the vision projection and deep learning components concurrently on GPU with CUDA Graph
#     # This is a placeholder for the actual function calls to run each component concurrently
#     # And I assume most interfaces have been implemented in repo

#     # Base Side: 
#     # Vision Projection Component (on GPU): A. SigLIP + DINOv2
#     vision_projection_start_time = time.time()
#     # Deep Learning Component (on GPU): A. Llama 2
#     vision_token_base = run_vision_projection_base(current_image)
#     vision_projection_end_time = time.time()
#     record_time('vision_projection_base', vision_projection_end_time - vision_projection_start_time)

#     llama_token = vision_token_base + text_token + placeholder_token 

#     deep_learning_base_start_time = time.time()
#     hidden_token = run_deep_learning_base(llama_token)
#     deep_learning_base_end_time = time.time()
#     record_time('deep_learning_base', deep_learning_base_end_time - deep_learning_base_start_time)

#     # Connection Component (on GPU): Token projection
#     connection_start_time = time.time()
#     connection_token = run_connection(hidden_token)
#     connection_end_time = time.time()
#     record_time('connection', connection_end_time - connection_start_time)

#     # Edge Side:
#     # Vision Projection Component (on GPU): B. Eff. Net
#     vision_projection_edge_start_time = time.time()
#     # Deep Learning Component (on GPU): B. Transformer
#     vision_token_edge = run_vision_projection_edge(current_image,past_image)
#     vision_projection_edge_end_time = time.time()
#     record_time('vision_projection_edge', vision_projection_edge_end_time - vision_projection_edge_start_time)

#     transformer_token = vision_token_edge + projected_token

#     run_deep_learning_edge_start_time = time.time()
#     run_deep_learning_edge(transformer_token)
#     run_deep_learning_edge_end_time = time.time()
#     record_time('deep_learning_edge', run_deep_learning_edge_end_time - run_deep_learning_edge_start_time)

#     e2e_end_time = time.time()
#     record_time('end_to_end', e2e_end_time - e2e_start_time)