import os
import time
import json
import traceback
import random
import torchvision.transforms as T
import torchvision.transforms.functional as F
from PIL import Image
import webdataset as wds
from .data_utils import pil_img2rgb
from .distributed_iterable_dataset import DistributedIterableDataset
import io
import re
from itertools import groupby
import torch
import cv2
import itertools
import h5py
import numpy as np
np.bool = np.bool_
if not hasattr(np, 'complex'):
    np.complex = complex
if not hasattr(np, 'float'):
    np.float = float
if not hasattr(np, 'int'):
    np.int = int
import imgaug.augmenters as iaa
import string
import fnmatch

from .rlds.dataset import make_interleaved_dataset, make_single_dataset
from .rlds.oxe import OXE_NAMED_MIXTURES, get_oxe_dataset_kwargs_and_weights
from pathlib import Path
from typing import Any, Dict, Tuple, Type
from .rlds.utils.data_utils import NormalizationType


class HDF5IterableDataset(DistributedIterableDataset):
    def __init__(
        self, dataset_name, transform, vit_transform, tokenizer, data_dir_list, num_used_data,
        img_aug, use_subtask=False, use_goal_image=False,
        local_rank=0, world_size=1, num_workers=8, data_status=None,
        shuffle_lines=False, shuffle_seed=0,
    ):
        """
        data_dir_list: list of directories containing .hdf5 files
        num_used_data: list of number of .hdf5 files to use from each directory. Use -1 or None to use all files.
        """
        super().__init__(dataset_name, local_rank, world_size, num_workers)
        self.transform = transform
        self.vit_transform = vit_transform
        self.tokenizer = tokenizer
        self.data_status = data_status
        self.img_aug = img_aug
        self.use_subtask = use_subtask
        self.use_goal_image = use_goal_image

        self.CHUNK_SIZE = 16
        self.IMG_HISORY_SIZE = 3

        self.data_paths = self._get_hdf5_files_from_dirs(data_dir_list, num_used_data)
        if not self.data_paths:
            raise ValueError("No .hdf5 files found in the given directories. Please check your paths and configuration.")

        self.set_epoch()

    def _get_hdf5_files_from_dirs(self, data_dir_list, num_used_data):
        final_hdf5_list = []
        for dir_path, num_hdf5s in zip(data_dir_list, num_used_data):
            if not os.path.isdir(dir_path):
                print(f"Directory {dir_path} does not exist. Skipping.")
                continue

            dir_path = Path(dir_path)
            try:
                hdf5_files_in_dir = sorted([str(path) for path in dir_path.rglob('*.hdf5')])
            except Exception as e:
                print(f"Error listing files in directory {dir_path}: {e}")
                continue

            if not hdf5_files_in_dir:
                print(f"No .hdf5 files found in directory {dir_path}. Skipping.")
                continue

            if num_hdf5s is None or num_hdf5s < 0:
                selected_files = hdf5_files_in_dir
            else:
                if len(hdf5_files_in_dir) < num_hdf5s:
                    print(f"Only {len(hdf5_files_in_dir)} .hdf5 files found in directory {dir_path}, "
                          f"but {num_hdf5s} were requested. Using all available files.")
                selected_files = hdf5_files_in_dir[:num_hdf5s]

            final_hdf5_list.extend(selected_files)

        return final_hdf5_list

    def _get_episode_len(self, file_path):
        try:
            with h5py.File(file_path, "r") as f:
                qpos = f["observations"]["qpos"]
                length = qpos.shape[0]
                return length
        except Exception as e:
            print(f"Error reading length from {file_path}: {e}")
            return 0

    def parse_hdf5_file(self, file_path):
        try:
            with h5py.File(file_path, "r") as f:
                qpos = f["observations"]["qpos"][:]       # (T, 14)
                left_arm_dim = f["observations"]["left_arm_dim"][:]
                right_arm_dim = f["observations"]["right_arm_dim"][:]
                num_steps = qpos.shape[0]

                EPS = 1e-2
                qpos_delta = np.abs(qpos - qpos[0:1])
                indices = np.where(np.any(qpos_delta > EPS, axis=1))[0]
                if len(indices) > 0:
                    first_idx = indices[0]
                else:
                    print(f"[SKIP] No valid movement in file: {file_path}")
                    return False, None

                step_id = np.random.randint(first_idx - 1, num_steps)

                # ---- Load instruction & subtask json ----
                dir_path = os.path.dirname(file_path)
                instr_file_path = None
                subtask_file_path = None
                for filename in os.listdir(dir_path):
                    if fnmatch.fnmatch(filename, "episode*_subtask.json"):
                        subtask_file_path = os.path.join(dir_path, filename)
                    elif filename.endswith(".json"):
                        instr_file_path = os.path.join(dir_path, filename)

                if instr_file_path is None or subtask_file_path is None:
                    print(f"[SKIP] Missing JSON files: {file_path}")
                    return False, None

                with open(instr_file_path, 'r') as f_instr:
                    instruction_dict = json.load(f_instr)
                with open(subtask_file_path, 'r', encoding='utf-8') as f_subtask:
                    subtask_dict = json.load(f_subtask)

                instruction_type = 'seen'
                instruction = instruction_dict[instruction_type]
                if isinstance(instruction, list):
                    instruction = np.random.choice(instruction)

                # ---- fps & current time ----
                fps = subtask_dict.get("fps", 10)
                current_time_sec = step_id / fps

                # ---- Check whether reasoning exists / is valid ----
                reasoning_data = subtask_dict.get("reasoning")
                if not reasoning_data:
                    print(f"[SKIP] No reasoning data: {file_path}")
                    return False, None

                # ---- Keep original keys; only use float for sorting/comparison to avoid str(float) precision pitfalls ----
                reason_keys = sorted(reasoning_data.keys(), key=lambda t: float(t))
                reason_times_float = [float(t) for t in reason_keys]

                if len(reason_keys) == 0:
                    print(f"[SKIP] Empty reasoning keys: {file_path}")
                    return False, None

                # ---- Find the index of the current subtask ----
                current_subtask_idx = 0
                for idx, t in enumerate(reason_times_float):
                    if t <= current_time_sec:
                        current_subtask_idx = idx
                    else:
                        break

                # ---- Build thought ----
                thought = ""
                if self.use_subtask:
                    # Look up by original key; do not convert float->str
                    matched_key = reason_keys[current_subtask_idx]
                    reason_data = reasoning_data[matched_key]

                    plan = "\n".join(reason_data.get("plan", []))
                    subtask = reason_data.get("subtask", "")
                    left_primitives = f["Primitives"]["left"]["text"][step_id]
                    right_primitives = f["Primitives"]["right"]["text"][step_id]
                    movement = f"[LEFT]: {left_primitives}\n[RIGHT]: {right_primitives}\n"

                    thought = (
                        f"\n"
                        f"<|plan|>\n{plan}\n<|\\ plan|>\n\n"
                        f"<|subtask|>\n{subtask}\n<|\\ subtask|>\n\n"
                        f"<|movement|>\n{movement}\n<|\\ movement|>\n\n"
                    )

                # ---- Compute goal frame idx ----
                goal_frame_idx = None
                if self.use_goal_image:
                    if current_subtask_idx < len(reason_keys) - 1:
                        next_key = reason_keys[current_subtask_idx + 1]
                        next_ts = float(next_key)   # Only used for numeric comparison; do not convert back to str
                        goal_frame_idx = int(next_ts * fps) - 1
                    else:
                        goal_frame_idx = num_steps - 1
                    goal_frame_idx = max(0, min(goal_frame_idx, num_steps - 1))

                # ---- state & action ----
                qpos = qpos / np.array([[1] * (left_arm_dim[0] + 1 + right_arm_dim[0] + 1)])
                target_qpos = f["action"][step_id:step_id + self.CHUNK_SIZE] / np.array(
                    [[1] * (left_arm_dim[0] + 1 + right_arm_dim[0] + 1)])

                state = qpos[step_id:step_id + 1]
                state_std = np.std(qpos, axis=0)
                state_mean = np.mean(qpos, axis=0)
                state_norm = np.sqrt(np.mean(qpos**2, axis=0))
                actions = target_qpos
                if actions.shape[0] < self.CHUNK_SIZE:
                    actions = np.concatenate(
                        [actions, np.tile(actions[-1:], (self.CHUNK_SIZE - actions.shape[0], 1))],
                        axis=0,
                    )

                # ---- Parse images ----
                def parse_img(key):
                    imgs = []
                    for i in range(max(step_id - self.IMG_HISORY_SIZE + 1, 0), step_id + 1):
                        img_bits = f["observations"]["images"][key][i]
                        img = cv2.imdecode(np.frombuffer(img_bits, np.uint8), cv2.IMREAD_COLOR)
                        imgs.append(img)

                    if self.use_goal_image and goal_frame_idx is not None:
                        goal_bits = f["observations"]["images"][key][goal_frame_idx]
                        goal_image = cv2.imdecode(np.frombuffer(goal_bits, np.uint8), cv2.IMREAD_COLOR)
                        imgs.append(goal_image)

                    imgs = np.stack(imgs)
                    target_len = self.IMG_HISORY_SIZE + (1 if self.use_goal_image else 0)
                    if imgs.shape[0] < target_len:
                        imgs = np.concatenate(
                            [np.tile(imgs[:1], (target_len - imgs.shape[0], 1, 1, 1)), imgs],
                            axis=0,
                        )
                    return imgs

                cam_high = parse_img("cam_high")
                valid_len = min(step_id - (first_idx - 1) + 1, self.IMG_HISORY_SIZE)
                cam_high_mask = np.array([False] * (self.IMG_HISORY_SIZE - valid_len) + [True] * valid_len)

                cam_left_wrist = parse_img("cam_left_wrist")
                cam_left_wrist_mask = cam_high_mask.copy()
                cam_right_wrist = parse_img("cam_right_wrist")
                cam_right_wrist_mask = cam_high_mask.copy()

                meta = {
                    "dataset_name": self.dataset_name,
                    "#steps": num_steps,
                    "step_id": step_id,
                    "instruction": instruction,
                }

                return True, {
                    "meta": meta,
                    "state": state,
                    "state_std": state_std,
                    "state_mean": state_mean,
                    "state_norm": state_norm,
                    "actions": actions,
                    "thought": thought,
                    "cam_high": cam_high,
                    "cam_high_mask": cam_high_mask,
                    "cam_left_wrist": cam_left_wrist,
                    "cam_left_wrist_mask": cam_left_wrist_mask,
                    "cam_right_wrist": cam_right_wrist,
                    "cam_right_wrist_mask": cam_right_wrist_mask,
                }

        except Exception as e:
            print(f"[ERROR] Failed to parse {file_path}: {str(e)}")
            return False, None

    def _build_dataset(self, hdf5_files_list):
        data_list = []
        for hdf5_file in hdf5_files_list:
            valid, data = self.parse_hdf5_file(hdf5_file)
            if valid:
                data_list.append(data)
        random.shuffle(data_list)
        return data_list

    def change_format(self, data, images):
        elements = []
        num_images = len(images)
        for idx in range(num_images - 1):
            elements.append({'type': 'image', 'has_loss': 0, 'image': images[idx]})
        elements.append({
            'type': 'text',
            'has_loss': 0,
            'text': f"You are an embodied agent. Your task is to: {data['meta']['instruction'].lower().rstrip('.')}. Plan and act.",
            'special_token_loss': 0,
        })
        if self.use_subtask:
            elements.append({
                'type': 'text',
                'has_loss': 1,
                'text': data['thought'],
                'special_token_loss': 0,
            })
        if self.use_goal_image:
            elements.append({
                'type': 'goal_image',
                'has_loss': 1,
                'image': images[-1],
            })
        elements.append({
            'type': 'action',
            'has_loss': 1,
            'action': data['actions'],
        })
        return elements

    def sample_random_params(self):
        enable_aug = random.random() < 0.8
        params = {
            "enable_aug": enable_aug,
            "brightness": random.uniform(0.6, 1.4),
            "contrast": random.uniform(0.6, 1.4),
            "saturation": random.uniform(0.0, 1.5),
            "hue": random.uniform(-0.1, 0.1),
            "bg_color": np.random.uniform(0.0, 1.0, (3,)),
            "add_strength": random.uniform(0.0, 0.3),
        }
        return params

    def apply_domain_randomization(self, pil_imgs, params):
        if not params["enable_aug"]:
            return pil_imgs

        seq = iaa.Sequential(
            [
                iaa.Sometimes(0.5, iaa.CoarseDropout(
                    (0.02, 0.10), size_percent=(0.02, 0.1), per_channel=False
                )),
                iaa.OneOf([
                    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
                    iaa.AdditiveLaplaceNoise(scale=(0.0, 0.05 * 255), per_channel=0.5),
                    iaa.AdditivePoissonNoise(lam=(0.0, 0.05 * 255), per_channel=0.5),
                ]),
                iaa.SomeOf((0, 1), [
                    iaa.OneOf([
                        iaa.GaussianBlur((0, 3.0)),
                        iaa.AverageBlur(k=(2, 7)),
                    ]),
                    iaa.MotionBlur(k=(3, 36)),
                ]),
                iaa.Sometimes(0.3, iaa.OneOf([
                    iaa.FrequencyNoiseAlpha(
                        exponent=(-4, -2),
                        first=iaa.Multiply([0.5, 1.5], per_channel=True),
                        second=iaa.ContrastNormalization((0.5, 2.0))
                    ),
                    iaa.SimplexNoiseAlpha(
                        iaa.OneOf([
                            iaa.EdgeDetect(alpha=(0.5, 1.0)),
                            iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
                        ])
                    ),
                ])),
            ],
            random_order=True,
        )

        jittered_imgs = []
        for img in pil_imgs:
            img = F.adjust_brightness(img, params["brightness"])
            img = F.adjust_contrast(img, params["contrast"])
            img = F.adjust_saturation(img, params["saturation"])
            img = F.adjust_hue(img, params["hue"])
            jittered_imgs.append(np.array(img))

        batch_arr = np.stack(jittered_imgs).astype(np.float32) / 255.0
        color = params["bg_color"]
        strength = params["add_strength"]
        batch_arr = batch_arr * (1 - strength) + color * strength
        images_uint8 = (batch_arr * 255).astype(np.uint8)

        combined_img = np.concatenate(list(images_uint8), axis=2)
        combined_img = combined_img[None, ...]
        seq_det = seq.to_deterministic()
        aug_result = seq_det(images=combined_img)
        aug_result = aug_result[0]
        out_imgs_np = np.split(aug_result, len(pil_imgs), axis=2)
        out_imgs = [Image.fromarray(img) for img in out_imgs_np]
        return out_imgs

    def __iter__(self):
        hdf5_files_for_this_worker, worker_id = self.get_data_paths_per_worker()
        if not hdf5_files_for_this_worker:
            print(f"Rank-{self.local_rank} worker-{worker_id} dataset-{self.dataset_name}: "
                  f"No hdf5 files for this worker.")
            return

        print(f"Worker-{worker_id}: Calculating episode lengths for weighted sampling...")
        file_lengths = []
        valid_files = []

        for f_path in hdf5_files_for_this_worker:
            l = self._get_episode_len(f_path)
            if l > 0:
                file_lengths.append(l)
                valid_files.append(f_path)

        if not valid_files:
            print(f"Worker-{worker_id}: No valid files found after length check.")
            return

        total_len = sum(file_lengths)
        sample_weights = np.array(file_lengths) / total_len

        SAMPLE_BATCH_SIZE = 150
        transform_stride = self.transform.stride

        print(f"Rank-{self.local_rank} worker-{worker_id} dataset-{self.dataset_name}: "
              f"hdf5 files for this worker: {len(hdf5_files_for_this_worker)}")

        while True:
            sampled_files_batch = np.random.choice(
                valid_files,
                size=SAMPLE_BATCH_SIZE,
                p=sample_weights,
                replace=True
            )
            for data_item in self._build_dataset(sampled_files_batch):
                num_tokens = 0
                image_tensor_list = []
                text_ids_list = []
                action_list = []
                sequence_plan = []

                try:
                    raw_images = None
                    if 'cam_high' in data_item:
                        if type(data_item['cam_high']) == np.ndarray:
                            raw_images = [
                                pil_img2rgb(Image.fromarray(image)) for image in data_item['cam_high']
                            ]
                        else:
                            raw_images = [
                                pil_img2rgb(Image.fromarray(data_item['cam_high']))
                            ]
                except:
                    traceback.print_exc()
                    continue

                if self.img_aug:
                    random_params = self.sample_random_params()
                    raw_images = self.apply_domain_randomization(raw_images, random_params)

                elements = self.change_format(data_item, raw_images)

                for item in elements:
                    if item['type'] == 'text':
                        text_data = item['text']
                        text_ids = self.tokenizer.encode(text_data)
                        if len(text_ids) > 0:
                            text_ids_list.append(text_ids)
                            num_tokens += len(text_ids)
                            sequence_plan.append({
                                'type': 'text',
                                'enable_cfg': 0,
                                'loss': item['has_loss'],
                                'special_token_loss': item['special_token_loss'],
                                'special_token_label': None,
                            })
                    elif item['type'] == 'image':
                        if self.use_goal_image:
                            sequence_plan.append({
                                'type': 'vae_image',
                                'enable_cfg': 0,
                                'loss': 0,
                                'special_token_loss': 0,
                                'special_token_label': None,
                            })
                            image_tensor = self.transform(item['image'])
                            h, w = image_tensor.shape[1:]
                            num_tokens += w * h // self.transform.stride ** 2
                            image_tensor_list.append(image_tensor.clone())

                        sequence_plan.append({
                            'type': 'vit_image',
                            'enable_cfg': 0,
                            'loss': 0,
                            'special_token_loss': 0,
                            'special_token_label': None,
                        })
                        vit_image_tensor = self.vit_transform(item['image'])
                        h, w = vit_image_tensor.shape[1:]
                        num_tokens += w * h // self.vit_transform.stride ** 2
                        image_tensor_list.append(vit_image_tensor)

                    elif item['type'] == 'goal_image':
                        # vae image with loss
                        sequence_plan.append({
                            'type': 'vae_image',
                            'enable_cfg': 0,
                            'loss': 1,
                            'special_token_loss': 0,
                            'special_token_label': None,
                        })
                        image_tensor = self.transform(item['image'])
                        h, w = image_tensor.shape[1:]
                        num_tokens += w * h // self.transform.stride ** 2
                        image_tensor_list.append(image_tensor)

                        # vae image without loss (cfg)
                        sequence_plan.append({
                            'type': 'vae_image',
                            'enable_cfg': 0,
                            'loss': 0,
                            'special_token_loss': 0,
                            'special_token_label': None,
                        })
                        image_tensor = self.transform(item['image'])
                        h, w = image_tensor.shape[1:]
                        num_tokens += w * h // self.transform.stride ** 2
                        image_tensor_list.append(image_tensor.clone())

                        # vit image
                        sequence_plan.append({
                            'type': 'vit_image',
                            'enable_cfg': 0,
                            'loss': 0,
                            'special_token_loss': 0,
                            'special_token_label': None,
                        })
                        vit_image_tensor = self.vit_transform(item['image'])
                        h, w = vit_image_tensor.shape[1:]
                        num_tokens += w * h // self.vit_transform.stride ** 2
                        image_tensor_list.append(vit_image_tensor)

                    elif item['type'] == 'action':
                        action_list.append(item['action'])
                        num_tokens += len(item['action'])
                        sequence_plan.append({
                            'type': 'action',
                            'enable_cfg': 0,
                            'loss': item['has_loss'],
                            'special_token_loss': 0,
                            'special_token_label': None,
                        })

                has_loss = [item['loss'] for item in sequence_plan]
                if sum(has_loss) == 0:
                    print(f'No loss defined, skipped.')
                    continue

                yield dict(
                    image_tensor_list=image_tensor_list,
                    text_ids_list=text_ids_list,
                    action_list=action_list,
                    sequence_plan=sequence_plan,
                    num_tokens=num_tokens,
                    data_indexes={
                        "worker_id": worker_id,
                        "dataset_name": self.dataset_name,
                    }
                )

            print(f"{self.dataset_name} repeat in rank-{self.local_rank} worker-{worker_id}")