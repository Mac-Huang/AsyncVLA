# ===============================================================
# AsyncVLA Inference
# ===============================================================
# 
# Sample inference code for AsyncVLA
# if you want to control the robot, you need to update the current state such as pose and image in "run_asyncvla"
#
# ---------------------------
# Paths and System Setup
# ---------------------------
import sys, os
sys.path.extend([
    "../Learning-to-Drive-Anywhere-with-MBRA/train/", '../lerobot'
])

import time, math, json
import threading
import argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from edge_timing import (
    EdgeStageTimer, execution_context, observe_benchmark,
    cpu_snapshot, cpu_counter_delta,
)
import yaml
from typing import Optional, Tuple, Type, Dict
from dataclasses import dataclass

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.nn.parallel import DistributedDataParallel as DDP
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_tensor
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import utm

# ---------------------------
# Custom Imports
# ---------------------------
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.models.projectors import ProprioProjector
from prismatic.models.action_heads import L1RegressionActionHead_idcat, L1RegressionDistHead
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction_MMNv1
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.training.train_utils import get_current_action_mask, get_next_actions_mask
from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK, POSE_DIM, ACTION_PROPRIO_NORMALIZATION_TYPE
from prismatic.models.small_head import Edge_adapter, Proj_Actiontokens

from transformers import AutoConfig, AutoProcessor, AutoModelForVision2Seq, AutoImageProcessor

transform = transforms.Compose([
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ===============================================================
# Utility Functions
# ===============================================================
def remove_ddp_in_checkpoint(state_dict: dict) -> dict:
    return {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}

def load_checkpoint(module_name: str, path: str, step: int, device: str = "cpu") -> dict:
    if not os.path.exists(os.path.join(path, f"{module_name}--{step}_checkpoint.pt")) and module_name == "pose_projector":
        module_name = "proprio_projector"
    checkpoint_path = os.path.join(path, f"{module_name}--{step}_checkpoint.pt")
    print(f"Loading checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device)
    return remove_ddp_in_checkpoint(state_dict)

def count_parameters(module: nn.Module, name: str) -> None:
    num_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    print(f"# trainable params in {name}: {num_params}")

def init_module(
    module_class: Type[nn.Module],
    module_name: str,
    cfg: "InferenceConfig",
    device_id: int,
    module_args: dict,
    to_bf16: bool = False,
) -> DDP:
    module = module_class(**module_args)
    count_parameters(module, module_name)

    if cfg.resume:
        state_dict = load_checkpoint(module_name, cfg.vla_path, cfg.resume_step)
        module.load_state_dict(state_dict)

    if to_bf16:
        module = module.to(torch.bfloat16)
    module = module.to(device_id)
    return module

def delta_to_pose(delta):
    """
    Fully differentiable, no inplace ops.
    delta: [N, T, 4]
    """
    dx = delta[..., 0]
    dy = delta[..., 1]
    dtheta = torch.atan2(delta[..., 3], delta[..., 2])

    N, T = dx.shape

    # Allocate list, then stack (safe)
    poses = []

    # Initial pose
    x = dx[:, 0]
    y = dy[:, 0]
    theta = dtheta[:, 0]

    poses.append(torch.stack([x, y, torch.cos(theta), torch.sin(theta)], dim=-1))

    # Loop for t >= 1
    for t in range(1, T):
        ct = torch.cos(theta)
        st = torch.sin(theta)

        dx_w = ct * dx[:, t] - st * dy[:, t]
        dy_w = st * dx[:, t] + ct * dy[:, t]

        # NEW tensors — NOT inplace updates
        x = x + dx_w
        y = y + dy_w
        theta = theta + dtheta[:, t]

        poses.append(torch.stack([x, y, torch.cos(theta), torch.sin(theta)], dim=-1))

    # Stack at the end
    return torch.stack(poses, dim=1)

# ===============================================================
# Inference Class
# ===============================================================
class Inference:
    def __init__(self, save_dir, lan_inst_prompt, goal_utm, goal_compass, goal_image_PIL, action_tokenizer, processor):
        self.tick_rate = 3
        self.lan_inst_prompt = lan_inst_prompt
        self.goal_utm = goal_utm
        self.goal_compass = goal_compass
        self.goal_image_PIL = goal_image_PIL
        self.action_tokenizer = action_tokenizer
        self.processor = processor
        self.count_id = 0
        self.linear, self.angular = 0.0, 0.0
        self.datastore_path_image = save_dir
    # ----------------------------
    # Static Utility Methods
    # ----------------------------
    @staticmethod
    def calculate_relative_position(x_a, y_a, x_b, y_b):
        return x_b - x_a, y_b - y_a

    @staticmethod
    def rotate_to_local_frame(delta_x, delta_y, heading_a_rad):
        rel_x = delta_x * math.cos(heading_a_rad) + delta_y * math.sin(heading_a_rad)
        rel_y = -delta_x * math.sin(heading_a_rad) + delta_y * math.cos(heading_a_rad)
        return rel_x, rel_y

    # ----------------------------
    # Main Loop
    # ----------------------------
    def run(self):
        loop_time = 1 / self.tick_rate
        start_time = time.time()
        while True:
            # I prob don't need to set the frequency here for the baseline
            # instead just let it run non stop
            if time.time() - start_time > loop_time:
                self.tick()
                start_time = time.time()
                break

    def tick(self):
        self.run_asyncvla()

    # ----------------------------
    # AsyncVLA Inference
    # ----------------------------
    def run_asyncvla(self):
        thres_dist = 30.0
        metric_waypoint_spacing = 0.1

        # Load current GPS & heading
        current_lat = 37.87371258374039
        current_lon = -122.26729417226024
        current_compass = 270.0
        cur_utm = utm.from_latlon(current_lat, current_lon)
        cur_compass = -float(current_compass) / 180.0 * math.pi  # inverted compass

        # Local goal position
        delta_x, delta_y = self.calculate_relative_position(
            cur_utm[0], cur_utm[1], self.goal_utm[0], self.goal_utm[1]
        )
        relative_x, relative_y = self.rotate_to_local_frame(delta_x, delta_y, cur_compass)
        radius = np.sqrt(relative_x**2 + relative_y**2)
        if radius > thres_dist:
            relative_x *= thres_dist / radius
            relative_y *= thres_dist / radius
        """
        goal_pose_loc_norm = np.array([
            relative_y / metric_waypoint_spacing,
            -relative_x / metric_waypoint_spacing,
            np.cos(self.goal_compass - cur_compass),
            np.sin(self.goal_compass - cur_compass)
        ])
        """
        yaw_ang = -90.0
        goal_pose_loc_norm = np.array([
            1.0 / metric_waypoint_spacing,
            -10.0 / metric_waypoint_spacing,
            np.cos(yaw_ang/180.0*3.1415),
            np.sin(yaw_ang/180.0*3.1415)
        ])
        print("Goal pose on the robot coordinate for 1st image: [x, y], yaw=", goal_pose_loc_norm[0:2]*metric_waypoint_spacing, yaw_ang)
        
        # Load current image
        current_image_path = "./inference/past.png"
        current_image_PIL = Image.open(current_image_path).convert("RGB").resize((224, 224), Image.BILINEAR)
        next_image_path = "./inference/cur.png"
        next_image_PIL = Image.open(next_image_path).convert("RGB").resize((224, 224), Image.BILINEAR)
        
        # Language instruction
        lan_inst = self.lan_inst_prompt if lan_prompt else "xxxx"
        """
        p_image=TF.resize(to_tensor(current_image_PIL), (96, 96)),    
        c_image=TF.resize(to_tensor(current_image_PIL), (96, 96)), 

        # Prepare batch
        batch = self.data_transformer_asyncvla(
            current_image_PIL, p_image, c_image, lan_inst, self.goal_image_PIL, goal_pose_loc_norm,
            prompt_builder=PurePromptBuilder,
            action_tokenizer=self.action_tokenizer,
            processor=self.processor
        )
        """
        # Run forward pass
        actions_list, modality_id = self.run_forward_pass(
            vla=vla.eval(),
            action_head=action_head.eval(),
            action_proj=action_proj.eval(),
            shead=shead.eval(),
            noisy_action_projector=None,
            pose_projector=pose_projector.eval(),
            current_image_PIL=current_image_PIL,
            lan_inst=lan_inst,
            goal_pose_loc_norm=goal_pose_loc_norm,
            metric_waypoint_spacing=metric_waypoint_spacing,
            action_tokenizer=self.action_tokenizer,
            device_id=device_id,
            use_l1_regression=True,
            use_diffusion=False,
            use_film=False,
            num_patches=NUM_PATCHES,
            compute_diffusion_l1=False,
            num_diffusion_steps_train=None,
            mode="train",
            idrun=self.count_id,
        )
        self.count_id += 1

        # Save behavior
        self.save_robot_behavior(
            current_image_PIL, next_image_PIL, goal_pose_loc_norm, actions_list, metric_waypoint_spacing, modality_id.cpu().numpy()
        )

    # ----------------------------
    # Save Robot Behavior Visualization
    # ----------------------------
    def save_robot_behavior(self, cur_img, goal_img, goal_pose, actions_list, metric_waypoint_spacing, mask_number):
        fig = plt.figure(figsize=(34, 16), dpi=80)
        gs = fig.add_gridspec(2, 2)
        ax_ob = fig.add_subplot(gs[0, 0])
        ax_goal = fig.add_subplot(gs[1, 0])
        ax_graph_pos = fig.add_subplot(gs[:, 1])

        ax_ob.imshow(np.array(cur_img).astype(np.uint8))
        ax_goal.imshow(np.array(goal_img).astype(np.uint8))

        x_seq_1 = actions_list[0][0, :, 0] #generated trajectory is on the robot coordinate. X is front and Y is left. 
        y_seq_inv_1 = -actions_list[0][0, :, 1]           
        x_seq_2 = actions_list[1][0, :, 0] #generated trajectory is on the robot coordinate. X is front and Y is left. 
        y_seq_inv_2 = -actions_list[1][0, :, 1]        
                
        ax_graph_pos.plot(np.insert(y_seq_inv_1, 0, 0.0), np.insert(x_seq_1, 0, 0.0), linewidth=4.0, markersize=12, marker='o', color='blue', label="1st image for edge adapter")
        ax_graph_pos.plot(np.insert(y_seq_inv_2, 0, 0.0), np.insert(x_seq_2, 0, 0.0), linewidth=4.0, markersize=12, marker='o', color='red', label="2nd image for edge adapter")
        
        # Mask annotation
        mask_type = int(mask_number[0])
        mask_texts = [
            "satellite only", "pose and satellite", "satellite and image", "all",
            "pose only", "pose and image", "image only", "language only", "language and pose"
        ]
        if mask_type < len(mask_texts):
            ax_graph_pos.annotate(mask_texts[mask_type], xy=(1.0, 0.0), xytext=(-20, 20), fontsize=18, textcoords='offset points')

        ax_ob.set_title("1st egocentric image", fontsize=18)
        ax_goal.set_title("2nd Egocentric image", fontsize=18)
        ax_graph_pos.tick_params(axis='x', labelsize=15) 
        ax_graph_pos.tick_params(axis='y', labelsize=15) 
        
        if int(mask_number[0]) == 1 or int(mask_number[0]) == 3 or int(mask_number[0]) == 4 or int(mask_number[0]) == 5 or int(mask_number[0]) == 8:
            ax_graph_pos.plot(-goal_pose[1], goal_pose[0], marker = '*', color='red', markersize=15)  
        else:                           
            ax_graph_pos.set_xlim(-3.0, 3.0)
            ax_graph_pos.set_ylim(-0.1, 10.0)
        ax_graph_pos.set_xlim(-6.0, 6.0)
        ax_graph_pos.set_ylim(-0.1, 12.0)
        ax_graph_pos.legend()                
        ax_graph_pos.set_title("Normalized generated 2D trajectories from AsyncVLA", fontsize=18)
        
        save_path = os.path.join(self.datastore_path_image, "visualization_asyncvla.jpg")
        plt.savefig(save_path)

    # ----------------------------
    # Custom Collator
    # ----------------------------
    def collator_custom(self, instances, model_max_length, pad_token_id, padding_side="right", pixel_values_dtype=torch.float32):
        IGNORE_INDEX = -100
        input_ids = pad_sequence([inst["input_ids"] for inst in instances], batch_first=True, padding_value=pad_token_id)
        labels = pad_sequence([inst["labels"] for inst in instances], batch_first=True, padding_value=IGNORE_INDEX)
        input_ids, labels = input_ids[:, :model_max_length], labels[:, :model_max_length]
        attention_mask = input_ids.ne(pad_token_id)

        pixel_values = [inst["pixel_values_current"] for inst in instances]
        if "dataset_name" in instances[0]:
            dataset_names = [inst["dataset_name"] for inst in instances]
        else:
            dataset_names = None

        if isinstance(pixel_values[0], torch.Tensor):
            if "pixel_values_goal" in instances[0]:
                pixel_values_goal = [inst["pixel_values_goal"] for inst in instances]
                pixel_values = torch.cat((torch.stack(pixel_values), torch.stack(pixel_values_goal)), dim=1)
            else:
                pixel_values = torch.stack(pixel_values)
        else:
            raise ValueError(f"Unsupported `pixel_values` type: {type(pixel_values)}")

        actions = torch.stack([torch.from_numpy(np.copy(inst["actions"])) for inst in instances])
        goal_pose = torch.stack([torch.from_numpy(np.copy(inst["goal_pose"])) for inst in instances])
        
        output = dict(
            pixel_values=pixel_values.to(),
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            actions=actions,
            goal_pose=goal_pose,
        )
        if dataset_names is not None:
            output["dataset_names"] = dataset_names
        return output

    # ----------------------------
    # Transform Data to Dataset Format
    # ----------------------------
    def transform_datatype(self, inst_obj, actions, goal_pose_cos_sin,
                           current_image_PIL, goal_image_PIL, prompt_builder, action_tokenizer,
                           base_tokenizer, image_transform, predict_stop_token=True):
        IGNORE_INDEX = -100
        current_action = actions[0]
        future_actions = actions[1:]
        future_actions_string = ''.join(action_tokenizer(future_actions))
        current_action_string = action_tokenizer(current_action)
        action_chunk_string = current_action_string + future_actions_string
        action_chunk_len = len(action_chunk_string)

        if inst_obj == "xxxx":
            conversation = [
                {"from": "human", "value": "No language instruction"},
                {"from": "gpt", "value": action_chunk_string},
            ]
        else:
            conversation = [
                {"from": "human", "value": f"What action should the robot take to {inst_obj}?"},
                {"from": "gpt", "value": action_chunk_string},
            ]

        prompt_builder = prompt_builder("openvla")
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Tokenize
        input_ids = torch.tensor(base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids)
        labels = input_ids.clone()
        labels[:-(action_chunk_len + 1)] = IGNORE_INDEX
        if not predict_stop_token:
            labels[-1] = IGNORE_INDEX

        pixel_values_current = image_transform(current_image_PIL)
        pixel_values_goal = image_transform(goal_image_PIL)
        dataset_name = "lelan"

        return dict(
            pixel_values_current=pixel_values_current,
            pixel_values_goal=pixel_values_goal,
            input_ids=input_ids,
            labels=labels,
            dataset_name=dataset_name,
            actions=torch.as_tensor(actions),
            goal_pose=goal_pose_cos_sin,
            img_PIL=current_image_PIL,
            inst=inst_obj,
        )

    # ----------------------------
    # Data Transformer for AsyncVLA
    # ----------------------------
    def data_transformer_asyncvla(self, current_image_PIL, lan_inst, goal_image_PIL, goal_pose_loc_norm, action_tokenizer, processor):
        actions = np.random.rand(8, 4)  # dummy actions
        goal_pose_cos_sin = goal_pose_loc_norm

        batch_data = self.transform_datatype(
            lan_inst, actions, goal_pose_cos_sin,
            current_image_PIL, goal_image_PIL,
            prompt_builder=PurePromptBuilder,
            action_tokenizer=action_tokenizer,
            base_tokenizer=processor.tokenizer,
            image_transform=processor.image_processor.apply_transform,
        )

        batch = self.collator_custom(
            # batch size is one cuz we only pass one example; i can't think of a reason to pass more than one example at a time for inference
            instances=[batch_data],
            model_max_length=processor.tokenizer.model_max_length,
            pad_token_id=processor.tokenizer.pad_token_id,
            padding_side="right"
        )
        return batch

    def pd_controller(self, actions, metric_waypoint_spacing):
        waypoints = actions.float().cpu().numpy()

        # Select waypoint
        waypoint_select = 4
        chosen_waypoint = waypoints[0][waypoint_select].copy()
        chosen_waypoint[:2] *= metric_waypoint_spacing
        dx, dy, hx, hy = chosen_waypoint

        # PD controller
        EPS = 1e-8
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

        # Velocity limitation
        maxv, maxw = 0.3, 0.3
        if np.abs(linear_vel_value) <= maxv:
            if np.abs(angular_vel_value) <= maxw:
                linear_vel_value_limit = linear_vel_value
                angular_vel_value_limit = angular_vel_value
            else:
                rd = linear_vel_value / angular_vel_value
                linear_vel_value_limit = maxw * np.sign(linear_vel_value) * np.abs(rd)
                angular_vel_value_limit = maxw * np.sign(angular_vel_value)
        else:
            if np.abs(angular_vel_value) <= 0.001:
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
                    
        return linear_vel_value_limit, angular_vel_value_limit                
        
    # ----------------------------
    # Run Forward Pass
    # ----------------------------
    def run_forward_pass(self, vla, action_head, action_proj, shead, noisy_action_projector, pose_projector,
                         current_image_PIL, lan_inst, goal_pose_loc_norm, metric_waypoint_spacing, action_tokenizer, device_id, use_l1_regression, use_diffusion,
                         use_film, num_patches, compute_diffusion_l1=False,
                         num_diffusion_steps_train=None, mode="vali", idrun=0) -> Tuple[torch.Tensor, Dict[str, float]]:
        
        batch = self.data_transformer_asyncvla(
            current_image_PIL, lan_inst, self.goal_image_PIL, goal_pose_loc_norm,
            action_tokenizer=self.action_tokenizer,
            processor=self.processor
        )

        metrics = {}
        noise, noisy_actions, diffusion_timestep_embeddings = None, None, None

        # Determine modality
        if satellite and not lan_prompt and not pose_goal and not image_goal:
            modality_id = torch.as_tensor([0], dtype=torch.float32)
        elif satellite and not lan_prompt and pose_goal and not image_goal:
            modality_id = torch.as_tensor([1], dtype=torch.float32)
        elif satellite and not lan_prompt and not pose_goal and image_goal:
            modality_id = torch.as_tensor([2], dtype=torch.float32)
        elif satellite and not lan_prompt and pose_goal and image_goal:
            modality_id = torch.as_tensor([3], dtype=torch.float32)
        elif not satellite and not lan_prompt and pose_goal and not image_goal:
            modality_id = torch.as_tensor([4], dtype=torch.float32)
        elif not satellite and not lan_prompt and pose_goal and image_goal:
            modality_id = torch.as_tensor([5], dtype=torch.float32)
        elif not satellite and not lan_prompt and not pose_goal and image_goal:
            modality_id = torch.as_tensor([6], dtype=torch.float32)
        elif not satellite and lan_prompt and not pose_goal and not image_goal:
            modality_id = torch.as_tensor([7], dtype=torch.float32)
        elif not satellite and lan_prompt and pose_goal and not image_goal:
            modality_id = torch.as_tensor([8], dtype=torch.float32)

        #For Edge adapter
        #img_cur = transform(c_image).to(device_id).to(torch.bfloat16)
        #img_past = transform(p_image).to(device_id).to(torch.bfloat16)        
        #img_cur = transform(batch["c_image"]).to(device_id).to(torch.bfloat16)
        #img_past = transform(batch["p_image"]).to(device_id).to(torch.bfloat16)

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            output: CausalLMOutputWithPast = vla(
                input_ids=batch["input_ids"].to(device_id),
                attention_mask=batch["attention_mask"].to(device_id),
                pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
                modality_id=modality_id.to(torch.bfloat16).to(device_id),
                labels=batch["labels"].to(device_id),
                output_hidden_states=True,
                proprio=batch["goal_pose"].to(torch.bfloat16).to(device_id),
                proprio_projector=pose_projector,
                noisy_actions=noisy_actions if use_diffusion else None,
                noisy_action_projector=noisy_action_projector if use_diffusion else None,
                diffusion_timestep_embeddings=diffusion_timestep_embeddings if use_diffusion else None,
                use_film=use_film,
            )

        # Prepare data for metrics
        ground_truth_token_ids = batch["labels"][:, 1:].to(device_id)
        current_action_mask = get_current_action_mask(ground_truth_token_ids)
        next_actions_mask = get_next_actions_mask(ground_truth_token_ids)
         
        # Get last layer hidden states
        last_hidden_states = output.hidden_states[-1]  # (B, seq_len, D)
        
        # Get hidden states for text portion of prompt+response (after the vision patches)
        text_hidden_states = last_hidden_states[:, num_patches:-1]              
        # Get hidden states for action portion of response
        batch_size = batch["input_ids"].shape[0]
        actions_hidden_states = (
            text_hidden_states[current_action_mask | next_actions_mask]
            .reshape(batch_size, NUM_ACTIONS_CHUNK * ACTION_DIM, -1)
            .to(torch.bfloat16)
        )  # (B, act_chunk_len, D)

        with torch.no_grad():
            projected_actions = action_proj.predict_action(actions_hidden_states.detach(), modality_id.to(torch.bfloat16).to(device_id))

        # TODO: Split right here for baseline run
        # Loop the VLM baseline/Edge adapter to simulate the streaming

        past_image_path = "./inference/past.png" 
        past_image_PIL = Image.open(past_image_path).convert("RGB").resize((224, 224), Image.BILINEAR)
        p_image=TF.resize(to_tensor(past_image_PIL), (96, 96)).unsqueeze(0)   
        img_past = transform(p_image).to(device_id).to(torch.bfloat16)    
        
        predicted_actions_list = []

        # example used 2 edge call and we can loop 2 calls
        # maybe just use odd for current and even for past image
        # Each shead call runs both EfficientNet encoders.
        # The baseline EdgeAdaptor wrapper also times image preparation.
        for i in range(2):
            # Load current image        
            if i==0:
                current_image_path = "./inference/past.png"                               
            elif i==1:
                current_image_path = "./inference/cur.png"                               
            current_image_PIL = Image.open(current_image_path).convert("RGB").resize((224, 224), Image.BILINEAR)                 
            c_image=TF.resize(to_tensor(current_image_PIL), (96, 96)).unsqueeze(0) 
            img_cur = transform(c_image).to(device_id).to(torch.bfloat16)        
            # Predict action
            with torch.no_grad():
                predicted_dactions = shead(img_cur, img_past, projected_actions)
                predicted_actions = delta_to_pose(predicted_dactions) 
            linear_vel, angular_vel = self.pd_controller(predicted_actions.cpu(), metric_waypoint_spacing)  
            print("action pose chunk", predicted_actions)                
            print("linear velocity", linear_vel)
            print("angular velocity", angular_vel)                                            
            predicted_actions_list.append(predicted_actions.cpu().float())

        # Return both the loss tensor (with gradients) and the metrics dictionary (with detached values)
        return predicted_actions_list, modality_id
          
# ===============================================================
# Split BaseVLM and Edge Adapter for Baseline SLO Measurement
# ===============================================================

def clip_angle(angle):
    # The original pd_controller refers to this helper but does not define it.
    return np.arctan2(np.sin(angle), np.cos(angle))


def resize_rgb(image):
    return image.convert("RGB").resize((224, 224), Image.BILINEAR)


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
        return self.forward_prepared(*self.prepare_inputs(raw))

    @torch.no_grad()
    def forward_prepared(self, batch, modality):
        """Run fresh vision and language inference from device-resident inputs."""
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
    """Raw current/past images and base tokens -> action-head deltas."""

    def __init__(self, shead, device, profile=False):
        super().__init__()
        self.shead = shead
        self.device = device
        self.stage_timer = EdgeStageTimer(shead, device) if profile else None

    def prepare_image(self, image, name):
        if self.stage_timer is None:
            small = TF.resize(to_tensor(resize_rgb(image)), (96, 96)).unsqueeze(0)
            return transform(small).to(device=self.device, dtype=torch.bfloat16)

        timer = self.stage_timer
        image = timer.cpu_call(name + ".resize224", lambda: resize_rgb(image))
        tensor = timer.cpu_call(name + ".to_tensor", lambda: to_tensor(image))
        small = timer.cpu_call(
            name + ".resize96",
            lambda: TF.resize(tensor, (96, 96)).unsqueeze(0),
        )
        normalized = timer.cpu_call(name + ".normalize", lambda: transform(small))
        return timer.transfer(
            name + ".h2d_bf16",
            lambda: normalized.to(device=self.device, dtype=torch.bfloat16),
        )

    @torch.no_grad()
    def forward(self, current_image, past_image, base_tokens):
        # Both EfficientNets execute inside shead on every call.
        timer = self.stage_timer
        if timer is not None:
            timer.begin()
        try:
            img_cur = self.prepare_image(current_image, "current")
            img_past = self.prepare_image(past_image, "past")
            if timer is None:
                return self.shead(img_cur, img_past, base_tokens)
            return timer.cpu_call(
                "shead_dispatch",
                lambda: self.shead(img_cur, img_past, base_tokens),
            )
        except Exception:
            if timer is not None:
                timer.abort()
            raise


def run_base(base, raw):
    return base(raw)


def run_edge(edge, raw, base_tokens):
    return edge(raw["current_image"], raw["base_image"], base_tokens)


def finish_robot_command(delta_actions, inference, profile=None):
    if profile is None:
        trajectory = delta_to_pose(delta_actions)
        return inference.pd_controller(trajectory.cpu(), metric_waypoint_spacing=0.1)
    start = time.perf_counter_ns()
    trajectory = delta_to_pose(delta_actions)
    after_pose = time.perf_counter_ns()
    trajectory_cpu = trajectory.cpu()
    after_copy = time.perf_counter_ns()
    command = inference.pd_controller(trajectory_cpu, metric_waypoint_spacing=0.1)
    finish = time.perf_counter_ns()
    profile.update({
        "delta_to_pose_dispatch_ms": (after_pose - start) / 1e6,
        "trajectory_to_cpu_wall_ms": (after_copy - after_pose) / 1e6,
        "pd_controller_cpu_ms": (finish - after_copy) / 1e6,
    })
    return command


def measure_call(fn, device, record_wait=False):
    # Measure execution latency for run_base/run_edge.
    torch.cuda.synchronize(device)
    start_ns = time.perf_counter_ns()
    result = fn()
    dispatched_ns = time.perf_counter_ns() if record_wait else None
    torch.cuda.synchronize(device)
    end_ns = time.perf_counter_ns()
    record = {
        "start_ns": start_ns, "end_ns": end_ns,
        "latency_ms": (end_ns - start_ns) / 1e6,
    }
    if record_wait:
        record["completion_wait_ms"] = (end_ns - dispatched_ns) / 1e6
    return result, record


def measure_e2e(base, edge, raw, inference, device):
    # Four boundaries produce three consecutive timing records.
    torch.cuda.synchronize(device)
    t0 = time.perf_counter_ns()
    tokens = run_base(base, raw)
    torch.cuda.synchronize(device)
    t1 = time.perf_counter_ns()
    deltas = run_edge(edge, raw, tokens)
    torch.cuda.synchronize(device)
    t2 = time.perf_counter_ns()
    command = finish_robot_command(deltas, inference)
    torch.cuda.synchronize(device)
    t3 = time.perf_counter_ns()
    return command, {
        "t0_ns": t0, "t1_ns": t1, "t2_ns": t2, "t3_ns": t3,
        "base_ms": (t1 - t0) / 1e6,
        "edge_ms": (t2 - t1) / 1e6,
        "framework_ms": (t3 - t2) / 1e6,
        "e2e_ms": (t3 - t0) / 1e6,
        "linear_velocity": float(command[0]),
        "angular_velocity": float(command[1]),
    }


def call_timing(call_id, start_ns, finish_ns, origin_ns):
    return {
        "call_id": call_id,
        "start_ns": start_ns,
        "finish_ns": finish_ns,
        "start_ms": (start_ns - origin_ns) / 1e6,
        "finish_ms": (finish_ns - origin_ns) / 1e6,
        "latency_ms": (finish_ns - start_ns) / 1e6,
    }


def add_edge_profile(edge, record, wait_ms, command_profile):
    if edge.stage_timer is not None:
        profile = edge.stage_timer.collect()
        profile["completion_wait_ms"] = wait_ms
        record["edge_profile"] = profile
        record["command_profile"] = command_profile


def run_isolated_benchmark(
    base, edge, raw, inference, device, repeats=100, warmup=10,
    fixed_tokens=None,
):
    if repeats <= 0 or warmup < 0:
        raise ValueError("repeats must be positive and warmup must be nonnegative.")

    base_records, edge_records = [], []
    profiled = edge.stage_timer is not None
    with torch.cuda.device(device), torch.no_grad():
        for _ in range(warmup):
            run_base(base, raw)
        torch.cuda.synchronize(device)
        origin_ns = time.perf_counter_ns()

        # Only BaseVLM executes during this phase.
        for call_id in range(repeats):
            tokens, timing = measure_call(lambda: run_base(base, raw), device)
            base_records.append(call_timing(
                call_id, timing["start_ns"], timing["end_ns"], origin_ns,
            ))

        edge_tokens = tokens if fixed_tokens is None else fixed_tokens
        for _ in range(warmup):
            deltas = run_edge(edge, raw, edge_tokens)
            finish_robot_command(deltas, inference)
        torch.cuda.synchronize(device)
        context = execution_context(device) if profiled else None

        # BaseVLM stays idle throughout this phase.
        for call_id in range(repeats):
            deltas, timing = measure_call(
                lambda: run_edge(edge, raw, edge_tokens), device,
                record_wait=profiled,
            )
            command_profile = {} if profiled else None
            command = finish_robot_command(deltas, inference, command_profile)
            torch.cuda.synchronize(device)
            command_finish_ns = time.perf_counter_ns()
            record = call_timing(
                call_id, timing["start_ns"], timing["end_ns"], origin_ns,
            )
            record.update({
                "base_id": base_records[-1]["call_id"] if fixed_tokens is None else -1,
                "command_finish_ns": command_finish_ns,
                "command_finish_ms": (command_finish_ns - origin_ns) / 1e6,
                "framework_ms": (command_finish_ns - timing["end_ns"]) / 1e6,
                "edge_to_command_ms": (
                    command_finish_ns - timing["start_ns"]
                ) / 1e6,
                "linear_velocity": float(command[0]),
                "angular_velocity": float(command[1]),
            })
            add_edge_profile(
                edge, record, timing.get("completion_wait_ms", 0.0),
                command_profile,
            )
            edge_records.append(record)

    result = {
        "origin_ns": origin_ns,
        "base_calls": base_records,
        "edge_calls": edge_records,
    }
    if context is not None:
        result["execution_context"] = {"main": context}
        result["edge_cpu_cgroup_delta"] = cpu_counter_delta(
            context["cpu_cgroup"], cpu_snapshot(),
        )
    if fixed_tokens is not None:
        result["token_policy"] = "fixed_fixture"
    return result


def run_async_benchmark(
    base, edge, raw, inference, device, duration_s=30.0, warmup=10,
    base_enabled=True, fixed_tokens=None, cpu_threads=None,
):
    if duration_s <= 0 or warmup < 0:
        raise ValueError("duration_s must be positive and warmup must be nonnegative.")

    # Prepare tokens for warmup, or use the shared diagnostic fixture.
    with torch.cuda.device(device), torch.no_grad():
        seed_tokens = run_base(base, raw) if fixed_tokens is None else fixed_tokens
        torch.cuda.synchronize(device)

    base_stream = torch.cuda.Stream(device=device)
    edge_stream = torch.cuda.Stream(device=device)
    condition = threading.Condition()
    stop = threading.Event()
    start_gate = threading.Event()
    ready = threading.Barrier(3 if base_enabled else 2)
    latest = None
    if not base_enabled or fixed_tokens is not None:
        latest = {
            "base_id": -1, "tokens": seed_tokens,
            "past_image": raw["base_image"], "base_start_ns": None,
        }
    origin_ns = 0
    base_records, edge_records, errors = [], [], []
    contexts = {}
    profiled = edge.stage_timer is not None

    def fail(exc):
        with condition:
            errors.append(exc)
            stop.set()
            condition.notify_all()
        ready.abort()
        start_gate.set()

    def base_worker():
        nonlocal latest
        try:
            if cpu_threads is not None:
                torch.set_num_threads(cpu_threads)
            with torch.cuda.device(device), torch.cuda.stream(base_stream), torch.no_grad():
                for _ in range(warmup):
                    if stop.is_set():
                        return
                    run_base(base, raw)
                    base_stream.synchronize()
                if profiled:
                    contexts["base"] = execution_context(device)

                ready.wait()
                start_gate.wait()
                call_id = 0
                while not stop.is_set():
                    base_input = raw.copy()
                    start_ns = time.perf_counter_ns()
                    tokens = run_base(base, base_input)
                    base_stream.synchronize()
                    finish_ns = time.perf_counter_ns()

                    # Normal async execution publishes completed results.
                    # Controlled diagnostics keep the edge fixture unchanged.
                    if fixed_tokens is None:
                        packet = {
                            "base_id": call_id,
                            "tokens": tokens,
                            "past_image": base_input["base_image"],
                            "base_start_ns": start_ns,
                        }
                        with condition:
                            latest = packet
                            condition.notify_all()

                    base_records.append(
                        call_timing(call_id, start_ns, finish_ns, origin_ns)
                    )
                    call_id += 1
        except Exception as exc:
            fail(exc)

    def edge_worker():
        try:
            if cpu_threads is not None:
                torch.set_num_threads(cpu_threads)
            with torch.cuda.device(device), torch.cuda.stream(edge_stream), torch.no_grad():
                seed_tokens.record_stream(edge_stream)
                for _ in range(warmup):
                    if stop.is_set():
                        return
                    deltas = run_edge(edge, raw, seed_tokens)
                    finish_robot_command(deltas, inference)
                    edge_stream.synchronize()
                if profiled:
                    contexts["edge"] = execution_context(device)

                ready.wait()
                start_gate.wait()
                call_id = 0
                while not stop.is_set():
                    with condition:
                        condition.wait_for(
                            lambda: latest is not None or stop.is_set()
                        )
                        if stop.is_set():
                            break
                        packet = latest

                    edge_input = raw.copy()
                    edge_input["base_image"] = packet["past_image"]
                    packet["tokens"].record_stream(edge_stream)

                    start_ns = time.perf_counter_ns()
                    deltas = run_edge(edge, edge_input, packet["tokens"])
                    dispatched_ns = time.perf_counter_ns() if profiled else None
                    edge_stream.synchronize()
                    finish_ns = time.perf_counter_ns()

                    command_profile = {} if profiled else None
                    command = finish_robot_command(deltas, inference, command_profile)
                    edge_stream.synchronize()
                    command_finish_ns = time.perf_counter_ns()

                    record = call_timing(
                        call_id, start_ns, finish_ns, origin_ns
                    )
                    record.update({
                        "base_id": packet["base_id"],
                        "command_finish_ns": command_finish_ns,
                        "command_finish_ms": (command_finish_ns - origin_ns) / 1e6,
                        "framework_ms": (command_finish_ns - finish_ns) / 1e6,
                        "edge_to_command_ms": (command_finish_ns - start_ns) / 1e6,
                        "linear_velocity": float(command[0]),
                        "angular_velocity": float(command[1]),
                    })
                    if packet["base_start_ns"] is not None:
                        record["base_start_to_command_ms"] = (
                            command_finish_ns - packet["base_start_ns"]
                        ) / 1e6
                    add_edge_profile(
                        edge, record,
                        (finish_ns - dispatched_ns) / 1e6 if profiled else 0.0,
                        command_profile,
                    )
                    edge_records.append(record)
                    call_id += 1
                if profiled:
                    contexts["edge"]["cpu_cgroup_delta"] = cpu_counter_delta(
                        contexts["edge"]["cpu_cgroup"], cpu_snapshot(),
                    )
        except Exception as exc:
            fail(exc)

    workers = []
    if base_enabled:
        workers.append(threading.Thread(target=base_worker, name="base"))
    workers.append(threading.Thread(target=edge_worker, name="edge"))
    for worker in workers:
        worker.start()

    try:
        ready.wait()
        origin_ns = time.perf_counter_ns()
        start_gate.set()
        # This bounds experiment duration. It does not pace either worker.
        stop.wait(duration_s)
    except threading.BrokenBarrierError:
        pass
    finally:
        stop.set()
        start_gate.set()
        with condition:
            condition.notify_all()
        for worker in workers:
            worker.join()

    if errors:
        raise RuntimeError("An asynchronous benchmark worker failed.") from errors[0]

    result = {
        "origin_ns": origin_ns,
        "base_calls": base_records,
        "edge_calls": edge_records,
    }
    if profiled:
        result["execution_context"] = contexts
    if fixed_tokens is not None or not base_enabled:
        result["token_policy"] = "fixed_fixture"
        result["base_enabled"] = base_enabled
    return result


def save_benchmark_results(name, output_path, metadata, records):
    with open(output_path, "w") as output_file:
        json.dump({"metadata": metadata, **records}, output_file, indent=2)

    for component, samples in (
        ("base", records["base_calls"]),
        ("edge", records["edge_calls"]),
    ):
        print(f"{name} {component}: {len(samples)} completed calls")
        if samples:
            report_samples(f"{name} {component}", samples, "latency_ms")

    if records["edge_calls"]:
        for field in (
            "framework_ms", "edge_to_command_ms", "base_start_to_command_ms",
        ):
            if field in records["edge_calls"][0]:
                report_samples(
                    f"{name} {field}", records["edge_calls"], field,
                )
        profiles = [
            record["edge_profile"] for record in records["edge_calls"]
            if "edge_profile" in record
        ]
        if profiles:
            for field in (
                "cpu_prepare_ms", "cpu_prepare_thread_ms", "h2d_bf16_wall_ms",
                "shead_dispatch_ms", "completion_wait_ms",
            ):
                report_samples(f"{name} {field}", profiles, field)
            for group in ("cpu_wall_ms", "cpu_thread_ms", "cuda_span_ms"):
                for stage in profiles[0][group]:
                    report_samples(
                        f"{name} {group}.{stage}",
                        [profile[group] for profile in profiles], stage,
                    )
            if records.get("edge_cpu_cgroup_delta"):
                print(f"{name} edge CPU cgroup counters: "
                      f"{records['edge_cpu_cgroup_delta']}")
            elif "execution_context" in records:
                delta = records["execution_context"].get("edge", {}).get(
                    "cpu_cgroup_delta", {},
                )
                if delta:
                    print(f"{name} edge CPU cgroup counters: {delta}")
            command_profiles = [
                record["command_profile"] for record in records["edge_calls"]
                if "command_profile" in record
            ]
            for stage in command_profiles[0]:
                report_samples(
                    f"{name} command.{stage}", command_profiles, stage,
                )
    print(f"Saved {name} call timings to {output_path}")


def compare_edge_diagnostics(results):
    """Compare matching stage means across the three diagnostic conditions."""
    names = ("diag_main", "diag_worker", "diag_concurrent")
    if any(not results[name]["edge_calls"] for name in names):
        print("Stage comparison requires completed edge calls in all conditions.")
        return
    metrics = (
        ("edge latency", None, "latency_ms"),
        ("CPU image preparation", "edge_profile", "cpu_prepare_ms"),
        ("H2D and BF16 host duration", "edge_profile", "h2d_bf16_wall_ms"),
        ("network host dispatch", "edge_profile", "shead_dispatch_ms"),
        ("completion wait", "edge_profile", "completion_wait_ms"),
        ("pair encoder CUDA span", "cuda_span_ms", "pair_encoder"),
        ("current encoder CUDA span", "cuda_span_ms", "current_encoder"),
        ("transformer CUDA span", "cuda_span_ms", "transformer"),
        ("action head CUDA span", "cuda_span_ms", "action_head"),
    )
    print("Stage means (ms). CUDA spans overlap host durations.")
    print(f"{'stage':<30} {'main':>9} {'worker':>9} {'concurrent':>11}"
          f" {'main-worker':>13} {'worker-concurrent':>18}")
    for label, group, field in metrics:
        means = []
        for name in names:
            calls = results[name]["edge_calls"]
            if group is None:
                values = [call[field] for call in calls]
            elif group == "edge_profile":
                values = [call[group][field] for call in calls]
            else:
                values = [call["edge_profile"][group][field] for call in calls]
            means.append(float(np.mean(values)))
        main, worker, concurrent = means
        print(f"{label:<30} {main:9.2f} {worker:9.2f} {concurrent:11.2f}"
              f" {main - worker:13.2f} {worker - concurrent:18.2f}")


def report_samples(name, records, field):
    values = np.asarray([record[field] for record in records])
    print(f"{name}: mean={values.mean():.2f} ms, "
          f"p50={np.percentile(values, 50):.2f} ms, "
          f"p95={np.percentile(values, 95):.2f} ms")
                
# ===============================================================
# Inference Configuration
# ===============================================================
class InferenceConfig:
    resume: bool = True
    vla_path: str = "./AsyncVLA_release"
    resume_step: Optional[int] = 750000    
    use_l1_regression: bool = True
    use_diffusion: bool = False
    use_film: bool = False
    num_images_in_input: int = 2
    use_lora: bool = True
    lora_rank: int = 32
    lora_dropout: float = 0.0

def define_model(cfg: InferenceConfig) -> None:
    cfg.vla_path = cfg.vla_path.rstrip("/")
    print(f"Loading OpenVLA Model `{cfg.vla_path}`")

    # GPU setup
    device_id = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device_id)
    torch.cuda.empty_cache()

    print(
        "Detected constants:\n"
        f"\tNUM_ACTIONS_CHUNK: {NUM_ACTIONS_CHUNK}\n"
        f"\tACTION_DIM: {ACTION_DIM}\n"
        f"\tPOSE_DIM: {POSE_DIM}\n"
        f"\tACTION_PROPRIO_NORMALIZATION_TYPE: {ACTION_PROPRIO_NORMALIZATION_TYPE}"
    )

    # Register OpenVLA model to HF Auto Classes (not needed if the model is on HF Hub)
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction_MMNv1)
    
    # Load processor and VLA
    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(device_id) #            trust_remote_code=True,
    
    vla.vision_backbone.set_num_images_in_input(cfg.num_images_in_input)
    vla.to(dtype=torch.bfloat16, device=device_id)

    # If applicable, instantiate proprio projector
    pose_projector = init_module(
        ProprioProjector,
        "pose_projector",
        cfg,
        device_id,
        {"llm_dim": vla.llm_dim, "proprio_dim": POSE_DIM},
    )

    action_head = init_module(
        L1RegressionActionHead_idcat,
        "action_head",
        cfg,
        device_id,
        {"input_dim": vla.llm_dim, "hidden_dim": vla.llm_dim, "action_dim": ACTION_DIM},
        to_bf16=True,
    )

    action_proj = init_module(
        Proj_Actiontokens,
        "action_proj",
        cfg,
        device_id,
        {"input_dim": vla.llm_dim, "hidden_dim": vla.llm_dim, "action_dim": 1024},
        to_bf16=True,
    )
    
    #defining edge adapter
    with open("./config_nav/dataset_config.yaml", "r") as f:        
        config = yaml.safe_load(f)      
    
    shead = Edge_adapter(
        obs_encoding_size=config["obs_encoding_size"],
        mha_num_attention_heads=config["mha_num_attention_heads"],
        mha_num_attention_layers=config["mha_num_attention_layers"],
        mha_ff_dim_factor=config["mha_ff_dim_factor"],
        )
    
    if cfg.resume:
        checkpoint_path_shead = os.path.join(cfg.vla_path, f"shead--{cfg.resume_step}_checkpoint.pt")
        if not os.path.isfile(checkpoint_path_shead):
            raise FileNotFoundError(checkpoint_path_shead)
        print("Loading shead model from ", checkpoint_path_shead)
        latest_checkpoint_shead = torch.load(checkpoint_path_shead, map_location="cpu")
        state_dict = latest_checkpoint_shead
        if any(k.startswith("module.") for k in state_dict.keys()):
            print("Detected DDP-style checkpoint, removing 'module.' prefix...")
            state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

        shead.load_state_dict(state_dict, strict=True)
    
    shead.to(torch.bfloat16).to(device=device_id)
    
    # Get number of vision patches
    NUM_PATCHES = vla.vision_backbone.get_num_patches() * vla.vision_backbone.get_num_images_in_input()    
    NUM_PATCHES += 1 #for goal pose

    # Create Action Tokenizer
    action_tokenizer = ActionTokenizer(processor.tokenizer)

    return vla, action_head, pose_projector, shead, action_proj, device_id, NUM_PATCHES, action_tokenizer, processor

# ===============================================================
# Main Entry
# ===============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Measure isolated and asynchronous AsyncVLA execution."
    )
    parser.add_argument(
        "--mode",
        choices=("all", "isolated", "async", "edge-worker", "diagnose"),
        default="all",
        help="Select isolated, concurrent, or controlled edge diagnostics.",
    )
    parser.add_argument("--profile-edge", action="store_true",
                        help="Record CPU stages, CUDA spans, and encoder counts.")
    parser.add_argument("--gpu-telemetry", action="store_true",
                        help="Sample GPU clocks and operating state every 250 ms.")
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--duration", type=float, default=30.0,
                        help="Worker measurement window in seconds, not a frequency.")
    parser.add_argument("--cpu-threads", type=int, default=None,
                        help="Optional CPU thread control for a separate diagnostic run.")
    parser.add_argument("--reverse-order", action="store_true",
                        help="Reverse the three conditions in diagnose mode.")
    args = parser.parse_args()
    if (args.repeats <= 0 or args.warmup < 0
            or not math.isfinite(args.duration) or args.duration <= 0):
        parser.error("Require repeats > 0, warmup >= 0, and duration > 0.")
    if args.reverse_order and args.mode != "diagnose":
        parser.error("--reverse-order requires --mode diagnose.")
    if args.cpu_threads is not None:
        if args.cpu_threads <= 0:
            parser.error("--cpu-threads must be positive.")
        torch.set_num_threads(args.cpu_threads)


    # # select modality
    # pose_goal = True
    # satellite = False
    # image_goal = False
    # lan_prompt = False

    # # Goal definitions
    # lan_inst_prompt = "move toward blue trash bin"
    # goal_lat, goal_lon, goal_compass = 37.8738930785863, -122.26746181032362, 0.0
    # goal_utm = utm.from_latlon(goal_lat, goal_lon)
    # goal_compass = -float(goal_compass) / 180.0 * math.pi
    # goal_image_PIL = Image.open("./inference/goal.png").convert("RGB").resize((224, 224), Image.BILINEAR)

    # check CUDA
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please check your GPU setup.")

    # Define models (VLA, action_head, pose_projector, processor, etc.)
    cfg = InferenceConfig()
    vla, action_head, pose_projector, shead, action_proj, device_id, NUM_PATCHES, action_tokenizer, processor = define_model(cfg)

    # Set the inputs for both planner/action systems
    # action module will resize it to 96x96 later
    def load_rgb(path):
        with Image.open(path) as image:
            return image.convert("RGB")

    raw = {
        "base_image": load_rgb("./inference/past.png"),
        "current_image": load_rgb("./inference/cur.png"),
        "goal_image": load_rgb("./inference/goal.png"),
        # like demo, the goal pose (gps) is hardcoded
        "goal_pose": [10.0, -100.0,
                      np.cos(-90.0 / 180.0 * 3.1415),
                      np.sin(-90.0 / 180.0 * 3.1415)],
        "language": "move toward blue trash bin",
        # demo use 4 to match the original pose-only demo.
        # it doesn't support language + image or language + pose + image yet
        "modality_id": 4, # 8 can be language + pose
    }

    # Run inference
    inference = Inference(
        save_dir="./inference",
        lan_inst_prompt=raw["language"],
        goal_utm=(0.0, 0.0),  # Placeholder for goal UTM (world position) coordinates (not used in this example)
        goal_compass=0.0,  # Placeholder for goal compass (not used in this example)
        goal_image_PIL=raw["goal_image"],
        action_tokenizer=action_tokenizer,
        processor=processor,
    )
    # inference.run()

    # Reuse the same modules for isolated and asynchronous experiments.
    base = BaseVLM(vla, pose_projector, action_proj, inference, device_id, NUM_PATCHES).eval()
    profile_edge = args.profile_edge or args.mode == "diagnose"
    edge = EdgeAdaptor(shead, device_id, profile=profile_edge).eval()

    metadata = {
        "gpu": torch.cuda.get_device_name(device_id),
        "batch_size": 1,
        "dtype": "bfloat16",
        "language_instruction": raw["language"],
        "goal_pose": raw["goal_pose"],
    }
    experiments = []
    if args.mode == "diagnose":
        # Hold the exact edge inputs fixed across all three conditions.
        fixture = run_base(base, raw)
        torch.cuda.synchronize(device_id)
        experiments = [
            ("diag_main", lambda: run_isolated_benchmark(
                base, edge, raw, inference, device_id,
                repeats=args.repeats, warmup=args.warmup, fixed_tokens=fixture,
            )),
            ("diag_worker", lambda: run_async_benchmark(
                base, edge, raw, inference, device_id,
                duration_s=args.duration, warmup=args.warmup,
                cpu_threads=args.cpu_threads,
                base_enabled=False, fixed_tokens=fixture,
            )),
            ("diag_concurrent", lambda: run_async_benchmark(
                base, edge, raw, inference, device_id,
                duration_s=args.duration, warmup=args.warmup,
                cpu_threads=args.cpu_threads,
                base_enabled=True, fixed_tokens=fixture,
            )),
        ]
        if args.reverse_order:
            experiments.reverse()
    else:
        suffix = "_profile" if profile_edge else ""
        if args.mode in ("all", "isolated"):
            experiments.append(("isolated" + suffix, lambda: run_isolated_benchmark(
                base, edge, raw, inference, device_id,
                repeats=args.repeats, warmup=args.warmup,
            )))
        if args.mode == "edge-worker":
            experiments.append(("edge_worker" + suffix, lambda: run_async_benchmark(
                base, edge, raw, inference, device_id,
                duration_s=args.duration, warmup=args.warmup,
                cpu_threads=args.cpu_threads, base_enabled=False,
            )))
        if args.mode in ("all", "async"):
            experiments.append(("async" + suffix, lambda: run_async_benchmark(
                base, edge, raw, inference, device_id,
                duration_s=args.duration, warmup=args.warmup,
                cpu_threads=args.cpu_threads,
            )))

    diagnostic_results = {}
    try:
        for name, run_experiment in experiments:
            if profile_edge or args.gpu_telemetry:
                records = observe_benchmark(
                    run_experiment, device_id, telemetry=args.gpu_telemetry,
                )
            else:
                records = run_experiment()
            if args.mode == "diagnose":
                diagnostic_results[name] = records
            suffix = "_reverse" if args.mode == "diagnose" and args.reverse_order else ""
            save_benchmark_results(
                name, f"./inference/baseline_{name}{suffix}.json", metadata, records,
            )
        if args.mode == "diagnose":
            compare_edge_diagnostics(diagnostic_results)
    finally:
        if edge.stage_timer is not None:
            edge.stage_timer.close()
