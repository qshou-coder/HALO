import os
import time
import json
import traceback
import random
import torchvision.transforms as T
import torchvision.transforms.functional as F
from PIL import Image
_orig_Random_seed = random.Random.seed

def _patched_Random_seed(self, a=None, version=2):
    if isinstance(a, tuple):
        a = hash(a)  # Workaround for Python 3.11 not accepting tuples as seed
    return _orig_Random_seed(self, a, version)

random.Random.seed = _patched_Random_seed
random.seed = lambda a=None, version=2: _patched_Random_seed(random._inst, a, version)
import webdataset as wds
from .data_utils import pil_img2rgb
from .distributed_iterable_dataset import DistributedIterableDataset
import io
import re
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
from .action_tokenizer import ActionTokenizer
from .rlds.utils.data_utils import NormalizationType
# --- llava-next ---
class SftWebDatasetIterableDataset(DistributedIterableDataset):
    def __init__(
        self, dataset_name, transform, tokenizer, data_dir_list, num_used_data, 
        local_rank=0, world_size=1, num_workers=8, data_status=None,
    ):
        """
        data_dir_list: list of directories containing .tar files
        num_used_data: list of number of .tar files to use from each directory. Use -1 or None to use all files.
        """
        super().__init__(dataset_name, local_rank, world_size, num_workers)
        self.transform = transform
        self.tokenizer = tokenizer
        self.data_status = data_status

        self.data_paths = self._get_tar_files_from_dirs(data_dir_list, num_used_data)
        if not self.data_paths:
            raise ValueError("No .tar files found in the given directories. Please check your paths and configuration.")
        
        # self.dataset = self._build_pipeline(final_tar_path_list)
        self.set_epoch()

    def _get_tar_files_from_dirs(self, data_dir_list, num_used_data):
        final_tar_list = []
        for dir_path, num_tars in zip(data_dir_list, num_used_data):
            if not os.path.isdir(dir_path):
                print(f"Directory {dir_path} does not exist. Skipping.")
                continue

            try:
                tar_files_in_dir = sorted([f for f in os.listdir(dir_path) if f.endswith('.tar')])
            except Exception as e:
                print(f"Error listing files in directory {dir_path}: {e}")
                continue

            if not tar_files_in_dir:
                print(f"No .tar files found in directory {dir_path}. Skipping.")
                continue

            if num_tars is None or num_tars < 0:
                selected_files = tar_files_in_dir
            else:
                if len(tar_files_in_dir) < num_tars:
                    print(f"Only {len(tar_files_in_dir)} .tar files found in directory {dir_path}, but {num_tars} were requested. Using all available files.")
                selected_files = tar_files_in_dir[:num_tars]

            full_path_files = [os.path.join(dir_path, f) for f in selected_files]
            final_tar_list.extend(full_path_files)

        return final_tar_list
    
    def _to_tuple(sample):
        keys = ["__key__", "__url__", "json", "__local_path__"]
        if "jpg" in sample:
            keys.append("jpg")

        return tuple(sample[k] for k in keys)
    
    def _build_pipeline(self, tar_path_list):
        pipeline = wds.WebDataset(
            tar_path_list,
            shardshuffle=False,
            nodesplitter=None,
            workersplitter=None,
        )

        print("Message: Enable sample-level shuffle by default...")
        pipeline = pipeline.shuffle(1000)
        pipeline = pipeline.decode("pil", handler=wds.handlers.warn_and_continue)
        # pipeline = cycle(pipeline)
        # pipeline = pipeline.to_tuple("__key__", "jpg", "txt")
        # {'__key__': '000000470005', '__url__': '${HALO_DATA_DIR}/Webdataset/LLaVA-NeXT-Data/shard-000047.tar', 'json': {'__key__': '000000470005', 'conversations': [...]}, '__local_path__': '${HALO_DATA_DIR}/Webdataset/LLaVA-NeXT-Data/shard-000047.tar', 'jpg': <PIL.Image.Image image mode=RGB size=640x480 at 0x7FE6274D0940>}
        # pipeline = pipeline.map(self._to_tuple)
        return pipeline

    # def change_format(self, data, num_images):
    #     elements = []
    #     key, image, text = data

    #     # Add image to elements
    #     elements.append({'type': 'image',})

    #     # Add text to elements
    #     if "?" in text:
    #         question, sep, answer = text.partition("?")
    #         question = question + sep
    #     else:
    #         match = re.search(r'\d+', text)
    #         if match:
    #             question = text[:match.start()] + '?'
    #             answer = text[match.start():]

    #     elements.append({
    #         'type': 'text',
    #         'has_loss': 0,
    #         'text': question,
    #     })
    #     elements.append({
    #         'type': 'text',
    #         'has_loss': 1,
    #         'text': answer,
    #     })

    #     return elements

    def change_format(self, data, num_images):
        data_term = data['json']
        elements = []
        for conversation in data_term['conversations']:
            if conversation['from'] == 'human':
                if '<image>' not in conversation['value']:
                    elements.append({
                        'type': 'text',
                        'has_loss': 0,
                        'text': conversation['value'],
                    })
                else:
                    text_list = conversation['value'].split('<image>')
                    for idx, text in enumerate(text_list):
                        if text.strip() != '':
                            elements.append({
                                'type': 'text',
                                'has_loss': 0,
                                'text': text.strip(),
                            })
                        if (idx != len(text_list) - 1) and (idx < num_images):
                            elements.append({'type': 'image',})
            elif conversation['from'] == 'gpt':
                elements.append({
                    'type': 'text',
                    'has_loss': 1,
                    'text': conversation['value'],
                })
        return elements

    def __iter__(self):
        tar_files_for_this_worker, worker_id = self.get_data_paths_per_worker()
        if not tar_files_for_this_worker:
            print(f"Rank-{self.local_rank} worker-{worker_id} dataset-{self.dataset_name}: "
                  f"No tar files for this worker.")
            return
        transform_stride = self.transform.stride

        # if self.data_status is not None:
        #     row_start_id = self.data_status[worker_id] + 1
        # else:
        #     row_start_id = 0

        # print(
        #     f"rank-{self.local_rank} worker-{worker_id} dataset-{self.dataset_name}: "
        #     f"resuming data at row#{row_start_id}"
        # )
        print(f"Rank-{self.local_rank} worker-{worker_id} dataset-{self.dataset_name}: "
              f"tar files for this worker: {len(tar_files_for_this_worker)}")
        # pipeline_for_this_worker = self._build_pipeline(tar_files_for_this_worker)

        while True:
            for data_item in self._build_pipeline(tar_files_for_this_worker):
                # key, image, text = data_item
                key = data_item["__key__"]
                url = data_item["__url__"]
                json = data_item["json"]
                local_path = data_item["__local_path__"]

                image = None
                if "jpg" in data_item:
                    image = data_item["jpg"]

                num_tokens = 0
                image_tensor_list = []
                text_ids_list = []
                sequence_plan = []

                try:
                    raw_images = None
                    if image:
                        raw_images = [pil_img2rgb(image)]

                        for raw_image in raw_images:
                            image_tensor = self.transform(raw_image, img_num=len(raw_images))
                            image_tensor_list.append(image_tensor)
                            height, width = image_tensor.shape[1:]
                            num_tokens += width * height // transform_stride ** 2
                    else:
                        raw_images = []
                except:
                    traceback.print_exc()
                    continue

                elements = self.change_format(data_item, len(image_tensor_list))
                for item in elements:
                    if item['type'] == 'text':
                        text_data = item['text']
                        text_ids = self.tokenizer.encode(text_data)
                        
                        if len(text_ids) > 0:
                            text_ids_list.append(text_ids)
                            num_tokens += len(text_ids)
                            current_plan = {
                                'type': 'text',
                                'enable_cfg': 0,
                                'loss': item['has_loss'],
                                'special_token_loss': 0,
                                'special_token_label': None,
                            }
                            sequence_plan.append(current_plan)
                    elif item['type'] == 'image':
                        current_plan = {
                            'type': 'vit_image',
                            'enable_cfg': 0,
                            'loss': 0,
                            'special_token_loss': 0,
                            'special_token_label': None,
                        }
                        sequence_plan.append(current_plan)

                # # Test
                # print(f"vqa length:{num_tokens}")
                has_loss = [item['loss'] for item in sequence_plan]
                if sum(has_loss) == 0:
                    print(f'No loss defined, skipped.')
                    continue

                # with open("${HALO_OUTPUT_DIR}/test_wbdata_key.txt", "a") as f:
                #     f.write(f"local_rank{self.local_rank} worker{worker_id} {key}\n")
            
                yield dict(
                    image_tensor_list=image_tensor_list,
                    text_ids_list=text_ids_list,
                    sequence_plan=sequence_plan,
                    num_tokens=num_tokens,
                    data_indexes={
                        "data_indexes": key,
                        "worker_id": worker_id,
                        "dataset_name": self.dataset_name,
                    }
                )

            # row_start_id = 0
            # print(f"{self.dataset_name} repeat in rank-{self.local_rank} worker-{worker_id}")

# # --- llava_video ---
class RLDSDataset(DistributedIterableDataset):
    def __init__(
        self, 
        dataset_name,   # 'open_x_embodiment'
        transform,
        tokenizer,
        data_dir_list,  # ['${HALO_DATA_DIR}/OpenVLA_Dataset']
        data_root_dir: Path, # '${HALO_DATA_DIR}/OpenVLA_Dataset'
        data_mix: str, # 'bridge_rt_1'
        resize_resolution: Tuple[int, int], # [256, 256]
        local_rank: int = 0, 
        world_size: int = 1, 
        num_workers: int = 8, 
        data_status = None,
        shuffle_buffer_size: int = 256_000,
        train: bool = True,
        image_aug: bool = False,
    ) -> None:
        """Lightweight wrapper around RLDS TFDS Pipeline for use with PyTorch Data Loaders."""
        self.dataset_name = dataset_name
        self.data_root_dir, self.data_mix = data_root_dir, data_mix
        self.transform = transform
        self.tokenizer = tokenizer
        self.data_status = data_status
        self.local_rank, self.world_size, self.num_workers = local_rank, world_size, num_workers

        # Configure RLDS Dataset(s), configure the dataset mixture according to OXE_NAMED_MIXTURES
        if self.data_mix in OXE_NAMED_MIXTURES:
            mixture_spec = OXE_NAMED_MIXTURES[self.data_mix]
        else:
            # Assume that passed "mixture" name is actually a single dataset -- create single-dataset "mix"
            mixture_spec = [(self.data_mix, 1.0)]

        # fmt: off
        per_dataset_kwargs, weights = get_oxe_dataset_kwargs_and_weights(
            self.data_root_dir,
            mixture_spec,
            load_camera_views=("primary",),
            load_depth=False,
            load_proprio=False,
            load_language=True,
            action_proprio_normalization_type=NormalizationType.BOUNDS_Q99, # Action normalization, 1% & 99% quantiles
        )
        rlds_config = dict(
            traj_transform_kwargs=dict(
                window_size=3,                                      # If we wanted to feed / predict more than one step, 3
                future_action_window_size=0,                        # For action chunking, 8
                skip_unlabeled=True,                                # Skip trajectories without language labels
                goal_relabeling_strategy="uniform",                 # Goals are currently unused
            ),
            frame_transform_kwargs=dict(
                resize_size=resize_resolution,
                num_parallel_calls=16,                          # For CPU-intensive ops (decoding, resizing, etc.)
            ),
            dataset_kwargs_list=per_dataset_kwargs,
            shuffle_buffer_size=shuffle_buffer_size,
            sample_weights=weights,
            balance_weights=True,
            traj_transform_threads=len(mixture_spec),
            traj_read_threads=len(mixture_spec),
            train=train,
        )

        # If applicable, enable image augmentations
        if image_aug:
            rlds_config["frame_transform_kwargs"].update({"image_augment_kwargs" : dict(
                random_resized_crop=dict(scale=[0.9, 0.9], ratio=[1.0, 1.0]),
                random_brightness=[0.2],
                random_contrast=[0.8, 1.2],
                random_saturation=[0.8, 1.2],
                random_hue=[0.05],
                augment_order=[
                    "random_resized_crop",
                    "random_brightness",
                    "random_contrast",
                    "random_saturation",
                    "random_hue",
                ],
            )}),
        # fmt: on

        # Initialize RLDS Dataset, we've already set the rlds_config, therefore no need to pay attention to
        self.dataset, self.dataset_length, self.dataset_statistics = self.make_dataset(rlds_config)

    def batch_transform(self, rlds_batch):
        action_tokenizer = ActionTokenizer()

        ### --- Single Turn ---
        # dataset_name, action = rlds_batch["dataset_name"], rlds_batch["action"][0]
        # img = Image.fromarray(rlds_batch["observation"]["image_primary"][0])
        # lang = rlds_batch["task"]["language_instruction"].decode().lower()

        # # Test
        # # timestamp = int(time.time() * 1000) 
        # # img.save(f"test/output_image_{timestamp}.jpg")
        # # with open("test/output_lang.txt", "a") as f:
        # #     f.write(lang + "\n")

        # output_term = {}
        # # Construct a llava type single turn conversation
        # # Requires a multi-turn version
        # conversation = [
        #     {"from": "human", "value": f"<image>\nWhat action should the robot take to {lang}?"},
        #     {"from": "gpt", "value": action_tokenizer(action)},
        # ]
        # image = img
        # output_term["image"] = image
        # output_term["conversations"] = conversation


        ### --- Multi Turn, refer: RoboOmni ---
        dataset_name, action = rlds_batch["dataset_name"], rlds_batch["action"]
        img = rlds_batch["observation"]["image_primary"]
        lang = rlds_batch["task"]["language_instruction"].decode().lower()
        # Construct a llava type multi turn conversation
        output_term = {}
        conversation = []
        image_list = []

        for idx in range(len(action)):
            conversation.append({
                "from": "human",
                "value": f"<image>\nWhat action should the robot take to {lang}?",
            })
            conversation.append({
                "from": "gpt",
                "value": action_tokenizer(action[idx]),
            })
            image_list.append(Image.fromarray(img[idx]))
        output_term["image"] = image_list
        output_term["conversations"] = conversation

        return output_term
        # {"image": image, "conversation": conversation(llava_type)}

    def make_dataset(self, rlds_config):
        return make_interleaved_dataset(**rlds_config) # Interleaved dataset, epoch length, dataset statistics
        # Important: dataset = {"observation": {}, "task": {}, "action": [], "dataset_name": [], "absolute_action_mask": [](Optional)}

    def change_format(self, data, num_images):
        # data = {"image": image, "conversation": conversation(llava_type)}
        elements = []
        for conversation in data['conversations']:
            if conversation['from'] == 'human':
                if '<image>' not in conversation['value']:
                    elements.append({
                        'type': 'text',
                        'has_loss': 0,
                        'text': conversation['value'],
                    })
                else:
                    text_list = conversation['value'].split('<image>')
                    for idx, text in enumerate(text_list):
                        if text.strip() != '':
                            elements.append({
                                'type': 'text',
                                'has_loss': 0,
                                'text': text.strip(),
                            })
                        if (idx != len(text_list) - 1) and (idx < num_images):
                            elements.append({'type': 'image',})
            elif conversation['from'] == 'gpt':
                elements.append({
                    'type': 'text',
                    'has_loss': 1,
                    'text': conversation['value'],
                })
        return elements

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        # Caculate global worker id and worker number
        global_worker_id = self.local_rank * num_workers + worker_id
        num_total_workers = self.world_size * num_workers

        # Create raw iterator
        iterator = self.dataset.as_numpy_iterator()
        # Set up sharded iterator
        sharded_iterator = itertools.islice(iterator, global_worker_id, None, num_total_workers)

        transform_stride = self.transform.stride
        for rlds_batch in sharded_iterator:
            # yield self.batch_transform(rlds_batch)
            data = self.batch_transform(rlds_batch)
            image = data['image']

            num_tokens = 0
            image_tensor_list = []
            text_ids_list = []
            sequence_plan = []

            try:
                raw_images = None
                if image:
                    if type(image) == list:
                        raw_images = [
                            pil_img2rgb(item) for item in image
                        ]
                    else:
                        raw_images = [
                            pil_img2rgb(image)
                        ]
                    for raw_image in raw_images:
                        image_tensor = self.transform(raw_image, img_num=len(raw_images))
                        image_tensor_list.append(image_tensor)
                        height, width = image_tensor.shape[1:]
                        num_tokens += width * height // transform_stride ** 2
                else:
                    raw_images = []
            except:
                traceback.print_exc()
                continue

            elements = self.change_format(data, len(image_tensor_list))
            for item in elements:
                if item['type'] == 'text':
                    text_data = item['text']
                    text_ids = self.tokenizer.encode(text_data)
                        
                    if len(text_ids) > 0:
                        text_ids_list.append(text_ids)
                        num_tokens += len(text_ids)
                        current_plan = {
                            'type': 'text',
                            'enable_cfg': 0,
                            'loss': item['has_loss'],
                            'special_token_loss': 0,
                            'special_token_label': None,
                        }
                        sequence_plan.append(current_plan)
                elif item['type'] == 'image':
                    current_plan = {
                        'type': 'vit_image',
                        'enable_cfg': 0,
                        'loss': 0,
                        'special_token_loss': 0,
                        'special_token_label': None,
                    }
                    sequence_plan.append(current_plan)

            has_loss = [item['loss'] for item in sequence_plan]
            if sum(has_loss) == 0:
                print(f'No loss defined, skipped.')
                continue

            # with open("${HALO_OUTPUT_DIR}/test_wbdata_key.txt", "a") as f:
            #     f.write(f"local_rank{self.local_rank} worker{worker_id} {key}\n")
            
            yield dict(
                image_tensor_list=image_tensor_list,
                text_ids_list=text_ids_list,
                sequence_plan=sequence_plan,
                num_tokens=num_tokens,
                data_indexes={
                    "worker_id": worker_id,
                    "dataset_name": self.dataset_name,
                }
            )

    def __len__(self) -> int:
        # Effective Dataset Length = Number of samples until each dataset has completed at least one epoch
        return self.dataset_length 

    # === Explicitly Unused ===
    def __getitem__(self, idx: int) -> None:
        raise NotImplementedError("IterableDataset does not implement map-style __getitem__; see __iter__ instead!")
    

# --- Open-X-Embodiement with continous flow matching ---
class OpenXActionFlowDataset(DistributedIterableDataset):
    def __init__(
        self, 
        dataset_name,   # 'open_x_embodiment'
        transform,
        # vit_transform,
        tokenizer,
        data_dir_list,  # ['${HALO_DATA_DIR}/OpenVLA_Dataset']
        data_root_dir: Path, # '${HALO_DATA_DIR}/OpenVLA_Dataset'
        data_mix: str, # 'bridge_rt_1'
        resize_resolution: Tuple[int, int], # [256, 256]
        local_rank: int = 0, 
        world_size: int = 1, 
        num_workers: int = 8, 
        data_status = None,
        shuffle_buffer_size: int = 256_000,
        train: bool = True,
        image_aug: bool = False,
    ) -> None:
        """Lightweight wrapper around RLDS TFDS Pipeline for use with PyTorch Data Loaders."""
        self.dataset_name = dataset_name
        self.data_root_dir, self.data_mix = data_root_dir, data_mix
        self.transform = transform
        # self.vit_transform = vit_transform
        self.tokenizer = tokenizer
        self.data_status = data_status
        self.local_rank, self.world_size, self.num_workers = local_rank, world_size, num_workers

        # Configure RLDS Dataset(s), configure the dataset mixture according to OXE_NAMED_MIXTURES
        if self.data_mix in OXE_NAMED_MIXTURES:
            mixture_spec = OXE_NAMED_MIXTURES[self.data_mix]
        else:
            # Assume that passed "mixture" name is actually a single dataset -- create single-dataset "mix"
            mixture_spec = [(self.data_mix, 1.0)]

        # fmt: off
        per_dataset_kwargs, weights = get_oxe_dataset_kwargs_and_weights(
            self.data_root_dir,
            mixture_spec,
            load_camera_views=("primary",),
            load_depth=False,
            load_proprio=False,
            load_language=True,
            action_proprio_normalization_type=NormalizationType.BOUNDS_Q99, # Action normalization, 1% & 99% quantiles
        )
        rlds_config = dict(
            traj_transform_kwargs=dict(
                window_size=3,                                      # If we wanted to feed / predict more than one step, 3
                future_action_window_size=16,                        # For action chunking, chunk_size - 1; we set 16 as chunk size
                skip_unlabeled=True,                                # Skip trajectories without language labels
                goal_relabeling_strategy="uniform",                 # Goals are currently unused
            ),
            frame_transform_kwargs=dict(
                resize_size=resize_resolution,
                num_parallel_calls=16,                          # For CPU-intensive ops (decoding, resizing, etc.)
            ),
            dataset_kwargs_list=per_dataset_kwargs,
            shuffle_buffer_size=shuffle_buffer_size,
            sample_weights=weights,
            balance_weights=True,
            traj_transform_threads=len(mixture_spec),
            traj_read_threads=len(mixture_spec),
            train=train,
        )

        # If applicable, enable image augmentations
        if image_aug:
            rlds_config["frame_transform_kwargs"].update({"image_augment_kwargs" : dict(
                random_resized_crop=dict(scale=[0.9, 0.9], ratio=[1.0, 1.0]),
                random_brightness=[0.2],
                random_contrast=[0.8, 1.2],
                random_saturation=[0.8, 1.2],
                random_hue=[0.05],
                augment_order=[
                    "random_resized_crop",
                    "random_brightness",
                    "random_contrast",
                    "random_saturation",
                    "random_hue",
                ],
            )}),
        # fmt: on

        # Initialize RLDS Dataset, we've already set the rlds_config, therefore no need to pay attention to
        self.dataset, self.dataset_length, self.dataset_statistics = self.make_dataset(rlds_config)

    def batch_transform(self, rlds_batch):
        dataset_name, action = rlds_batch["dataset_name"], rlds_batch["action"]
        img_list = rlds_batch["observation"]["image_primary"]
        lang = rlds_batch["task"]["language_instruction"].decode().strip().lower()
        goal_img = rlds_batch["goal_observation"]["image_primary"]
        # Construct a llava type multi turn conversation
        output_term = {}
        conversation = []
        image_list = []
        goal_image_list = []

        conversation.append({
            "from": "human",
            "value": f"{'<image>'*len(img_list)}\nWhat action should the robot take to {lang}?",
        })
        conversation.append({
            "from": "gpt",
            "value": action,
        })
        for img in img_list:
            image_list.append(Image.fromarray(img))
        output_term["image"] = image_list
        output_term["conversations"] = conversation
        output_term["action"] = action
        goal_image_list.append(Image.fromarray(goal_img))
        output_term["goal_image"] = goal_image_list

        return output_term
        # {"image": image, "conversation": conversation(llava_type)}

    def make_dataset(self, rlds_config):
        return make_interleaved_dataset(**rlds_config) # Interleaved dataset, epoch length, dataset statistics
        # Important: dataset = {"observation": {}, "task": {}, "action": [], "dataset_name": [], "absolute_action_mask": [](Optional)}

    def change_format(self, data):
        # data = {"image": image, "conversation": conversation(llava_type)}
        elements = []
        for conversation in data['conversations']:
            if conversation['from'] == 'human':
                if '<image>' not in conversation['value']:
                    elements.append({
                        'type': 'text',
                        'has_loss': 0,
                        'text': conversation['value'],
                    })
                else:
                    text_list = conversation['value'].split('<image>')
                    for idx, text in enumerate(text_list):
                        if text.strip() != '':
                            elements.append({
                                'type': 'text',
                                'has_loss': 0,
                                'text': text.strip(),
                            })
                        if (idx != len(text_list) - 1):
                            elements.append({'type': 'image',})
                    # elements.append({'type': 'goal_image',})
            elif conversation['from'] == 'gpt':
                elements.append({
                    'type': 'action',
                    'has_loss': 1,
                    'action': conversation['value'],
                })
        return elements

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        # Caculate global worker id and worker number
        global_worker_id = self.local_rank * num_workers + worker_id
        num_total_workers = self.world_size * num_workers

        # Create raw iterator
        iterator = self.dataset.as_numpy_iterator()
        # Set up sharded iterator
        sharded_iterator = itertools.islice(iterator, global_worker_id, None, num_total_workers)

        transform_stride = self.transform.stride
        for rlds_batch in sharded_iterator:
            # yield self.batch_transform(rlds_batch)
            data = self.batch_transform(rlds_batch)
            image = data['image']
            action = data['action']
            goal_image = data['goal_image']

            num_tokens = 0
            image_tensor_list = []
            text_ids_list = []
            action_list = []
            sequence_plan = []

            action_list.append(action) # modify

            # Process image
            try:
                raw_images = None
                if image:
                    if type(image) == list:
                        raw_images = [
                            pil_img2rgb(item) for item in image
                        ]
                    else:
                        raw_images = [
                            pil_img2rgb(image)
                        ]
            except:
                traceback.print_exc()
                continue
            
            if raw_images:
                for raw_image in raw_images:
                    image_tensor = self.transform(raw_image, img_num=len(raw_images))
                    image_tensor_list.append(image_tensor)
                    height, width = image_tensor.shape[1:]
                    num_tokens += width * height // transform_stride ** 2

            elements = self.change_format(data)
            for item in elements:
                if item['type'] == 'text':
                    text_data = item['text']
                    text_ids = self.tokenizer.encode(text_data)
                        
                    if len(text_ids) > 0:
                        text_ids_list.append(text_ids)
                        num_tokens += len(text_ids)
                        current_plan = {
                            'type': 'text',
                            'enable_cfg': 1,
                            'loss': item['has_loss'],
                            'special_token_loss': 0,
                            'special_token_label': None,
                        }
                        sequence_plan.append(current_plan)
                elif item['type'] == 'image':
                    sequence_plan.append(
                        {
                            'type': 'vit_image',
                            'enable_cfg': 1, 
                            'loss': 0,
                            'special_token_loss': 0,
                            'special_token_label': None,
                        },
                    )
                elif item['type'] == 'action':
                    num_tokens += item['action'].shape[0]
                    current_plan = {
                        'type': 'action',
                        'enable_cfg': 0,
                        'loss': item['has_loss'],
                        'special_token_loss': 0,
                        'special_token_label': None,
                    }
                    sequence_plan.append(current_plan)

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

    def __len__(self) -> int:
        # Effective Dataset Length = Number of samples until each dataset has completed at least one epoch
        return self.dataset_length 

    # === Explicitly Unused ===
    def __getitem__(self, idx: int) -> None:
        raise NotImplementedError("IterableDataset does not implement map-style __getitem__; see __iter__ instead!")
    

# --- RoboTwin Data (For few shot finetuning) ---
class SSv2VideoPredictionDataset(DistributedIterableDataset):
    def __init__(
        self, dataset_name, transform, vit_transform, tokenizer, frame_sampler, 
        jsonl_path_list, data_dir_list, num_used_data, 
        local_rank=0, world_size=1, num_workers=8, data_status=None, 
        shuffle_lines=False, shuffle_seed=0,
    ):
        """
        jsonl_path_list: list of jsonl file paths
        data_dir_list: list of image directories containing the images of each jsonl file
        num_used_data: list of number of sampled data points for each jsonl
        """
        super().__init__(dataset_name, local_rank, world_size, num_workers)
        self.transform = transform
        self.vit_transform = vit_transform
        self.tokenizer = tokenizer
        self.frame_sampler = frame_sampler
        self.data_status = data_status
        self.data_paths = self.get_data_paths(
            jsonl_path_list, 
            data_dir_list, 
            num_used_data, 
            shuffle_lines, 
            shuffle_seed,
        )
        self.lazy_set_epoch()

    def get_data_paths(
        self, 
        jsonl_path_list, 
        data_dir_list, 
        num_used_data, 
        shuffle_lines, 
        shuffle_seed,
    ):
        data_paths = []
        for jsonl_path, image_dir, num_data_point in zip(
            jsonl_path_list, data_dir_list, num_used_data
        ):
            with open(jsonl_path, 'r') as f:
                raw_data = json.load(f)
            if shuffle_lines:
                self.rng.seed(shuffle_seed)
                self.rng.shuffle(raw_data)
            raw_data = raw_data[:num_data_point]
            data_paths.extend([(json_data, os.path.join(image_dir, f"{json_data['id']}.webm")) for json_data in raw_data])
        return data_paths
    
    def extract_frames_simple(self, video_path, num_frames):
        """
        Using cv2.VideoCapture to extract frames from a video file.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            cap.release()
            return []

        # Core: compute sample positions and extract frames
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        frames = []

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)

        cap.release()
        return frames
    
    def __iter__(self):
        data_paths_per_worker, worker_id = self.get_data_paths_per_worker()
        if self.data_status is not None:
            row_start_id = self.data_status[worker_id] + 1
        else:
            row_start_id = 0
        transform_stride = self.transform.stride

        print(
            f"rank-{self.local_rank} worker-{worker_id} dataset-{self.dataset_name}: "
            f"resuming data at row#{row_start_id}"
        )

        while True:
            random.shuffle(data_paths_per_worker)
            data_paths_per_worker_ = data_paths_per_worker[row_start_id:]
            for row_idx, (data, video_path) in enumerate(data_paths_per_worker_, start=row_start_id):
                prompt = f"Imagine the next scene when the person's hand is {data['label']}"
                
                num_tokens = 0
                image_tensor_list = []
                text_ids_list = []
                sequence_plan = []

                try:
                    raw_images = self.extract_frames_simple(video_path, 16) # Using Opencv
                    raw_images = [Image.fromarray(image) for image in raw_images]
                    special_tokens = '<image>' * len(raw_images)
                except Exception as e:
                    traceback.print_exc()
                    continue
                
                if len(raw_images) < 16:
                    continue
                start_point = random.choice(range(len(raw_images)-5))
                elements = [
                    {'type': 'image', 'has_loss': 0, 'image': raw_images[start_point]}, 
                    {'type': 'image', 'has_loss': 0, 'image': raw_images[start_point+1]}, 
                    {'type': 'image', 'has_loss': 0, 'image': raw_images[start_point+2]}, 
                    {
                        'type': 'text',
                        'has_loss': 0,
                        'text': prompt,
                    },
                    {'type': 'image', 'has_loss': 1, 'image': raw_images[start_point+5]}, 
                ]

                for item in elements:
                    if item['type'] == 'text':
                        text_data = item['text']
                        text_ids = self.tokenizer.encode(text_data)
                        
                        if len(text_ids) > 0:
                            text_ids_list.append(text_ids)
                            num_tokens += len(text_ids)
                            current_plan = {
                                'type': 'text',
                                'enable_cfg': 0,
                                'loss': item['has_loss'],
                                'special_token_loss': 0,
                                'special_token_label': None,
                            }
                            sequence_plan.append(current_plan)
                    elif item['type'] == 'image':
                        if item['has_loss']:
                            sequence_plan.append(
                                {
                                    'type': 'vae_image', 
                                    'enable_cfg': 0, 
                                    'loss': 1, 
                                    'special_token_loss': 0,
                                    'special_token_label': None,
                                }
                            )

                            image_tensor = self.transform(item['image'])
                            height, width = image_tensor.shape[1:]
                            num_tokens += width * height // self.transform.stride ** 2
                            image_tensor_list.append(image_tensor)

                        
                        sequence_plan.append(
                            {
                                'type': 'vae_image', 
                                'enable_cfg': 0, 
                                'loss': 0, 
                                'special_token_loss': 0,
                                'special_token_label': None,
                            }
                        )
                        image_tensor = self.transform(item['image'])
                        height, width = image_tensor.shape[1:]
                        num_tokens += width * height // self.transform.stride ** 2
                        image_tensor_list.append(image_tensor.clone())

                        sequence_plan.append(
                            {
                                'type': 'vit_image',
                                'enable_cfg': 0, 
                                'loss': 0,
                                'special_token_loss': 0,
                                'special_token_label': None,
                            },
                        )
                        vit_image_tensor = self.vit_transform(item['image'])
                        height, width = vit_image_tensor.shape[1:]
                        num_tokens += width * height // self.vit_transform.stride ** 2
                        image_tensor_list.append(vit_image_tensor)

                has_loss = [item['loss'] for item in sequence_plan]
                if sum(has_loss) == 0:
                    print(f'No loss defined, skipped.')
                    continue

                yield dict(
                    image_tensor_list=image_tensor_list,
                    text_ids_list=text_ids_list,
                    sequence_plan=sequence_plan,
                    num_tokens=num_tokens,
                    data_indexes={
                        "data_indexes": row_idx,
                        "worker_id": worker_id,
                        "dataset_name": self.dataset_name,
                    }
                )

            row_start_id = 0
            print(f"{self.dataset_name} repeat in rank-{self.local_rank} worker-{worker_id}")

# --- kinetics-400 ---
class OpenXInverseDataset(DistributedIterableDataset):
    def __init__(
        self, 
        dataset_name,   # 'open_x_inverse'
        transform,
        vit_transform,
        tokenizer,
        data_dir_list,  # ['${HALO_DATA_DIR}/OpenVLA_Dataset']
        data_root_dir: Path, # '${HALO_DATA_DIR}/OpenVLA_Dataset'
        data_mix: str, # 'oxe_magic_soup'
        resize_resolution: Tuple[int, int], # [256, 256]
        local_rank: int = 0, 
        world_size: int = 1, 
        num_workers: int = 8, 
        data_status = None,
        shuffle_buffer_size: int = 256_000,
        train: bool = True,
        image_aug: bool = False,
    ) -> None:
        """Lightweight wrapper around RLDS TFDS Pipeline for use with PyTorch Data Loaders."""
        self.dataset_name = dataset_name
        self.data_root_dir, self.data_mix = data_root_dir, data_mix
        self.transform = transform
        self.vit_transform = vit_transform
        self.tokenizer = tokenizer
        self.data_status = data_status
        self.local_rank, self.world_size, self.num_workers = local_rank, world_size, num_workers

        # Configure RLDS Dataset(s), configure the dataset mixture according to OXE_NAMED_MIXTURES
        if self.data_mix in OXE_NAMED_MIXTURES:
            mixture_spec = OXE_NAMED_MIXTURES[self.data_mix]
        else:
            # Assume that passed "mixture" name is actually a single dataset -- create single-dataset "mix"
            mixture_spec = [(self.data_mix, 1.0)]

        # fmt: off
        per_dataset_kwargs, weights = get_oxe_dataset_kwargs_and_weights(
            self.data_root_dir,
            mixture_spec,
            load_camera_views=("primary",),
            load_depth=False,
            load_proprio=False,
            load_language=True,
            action_proprio_normalization_type=NormalizationType.BOUNDS_Q99, # Action normalization, 1% & 99% quantiles
        )
        rlds_config = dict(
            traj_transform_kwargs=dict(
                window_size=3,                                      # We need 3 historical images
                future_action_window_size=1,                        # Actually, we don't need to predict the future actions，just set it to 1
                skip_unlabeled=True,                                # Skip trajectories without language labels
                goal_relabeling_strategy="uniform",                 # Goals are currently unused
            ),
            frame_transform_kwargs=dict(
                resize_size=resize_resolution,                  # (256, 256)
                num_parallel_calls=16,                          # For CPU-intensive ops (decoding, resizing, etc.)
            ),
            dataset_kwargs_list=per_dataset_kwargs,
            shuffle_buffer_size=shuffle_buffer_size,
            sample_weights=weights,
            balance_weights=True,
            traj_transform_threads=len(mixture_spec),
            traj_read_threads=len(mixture_spec),
            train=train,
        )

        # If applicable, enable image augmentations
        if image_aug:
            rlds_config["frame_transform_kwargs"].update({"image_augment_kwargs" : dict(
                random_resized_crop=dict(scale=[0.9, 0.9], ratio=[1.0, 1.0]),
                random_brightness=[0.2],
                random_contrast=[0.8, 1.2],
                random_saturation=[0.8, 1.2],
                random_hue=[0.05],
                augment_order=[
                    "random_resized_crop",
                    "random_brightness",
                    "random_contrast",
                    "random_saturation",
                    "random_hue",
                ],
            )}),
        # fmt: on

        # Initialize RLDS Dataset, we've already set the rlds_config, therefore no need to pay attention to
        self.dataset, self.dataset_length, self.dataset_statistics = self.make_dataset(rlds_config)

    def batch_transform(self, rlds_batch):
        dataset_name, action = rlds_batch["dataset_name"], rlds_batch["action"]
        img_list = rlds_batch["observation"]["image_primary"]
        lang = rlds_batch["task"]["language_instruction"].decode().rstrip(string.punctuation).lower()
        if lang == '':
            lang = 'interact with any object'
        lang = lang.replace('\n', ' ')
        goal_img = rlds_batch["goal_observation"]["image_primary"]
        # Construct a llava type multi turn conversation
        output_term = {}
        conversation = []
        image_list = []
        goal_image_list = []

        conversation.append({
            "from": "human",
            "value": f"{'<image>'*(len(img_list))}\nImagine the next scene when the robot takes action to {lang}.",
        })
        conversation.append({
            "from": "gpt",
            "value": action,
        })

        for img in img_list:
            image_list.append(Image.fromarray(img))
        output_term["image"] = image_list
        output_term["conversations"] = conversation
        output_term["action"] = action
        goal_image_list.append(Image.fromarray(goal_img))
        output_term["goal_image"] = goal_image_list

        return output_term
        # {"image": image, "conversation": conversation(llava_type)}

    def make_dataset(self, rlds_config):
        return make_interleaved_dataset(**rlds_config) # Interleaved dataset, epoch length, dataset statistics
        # Important: dataset = {"observation": {}, "task": {}, "action": [], "dataset_name": [], "absolute_action_mask": [](Optional)}

    def change_format(self, data):
        # data = {"image": image, "conversation": conversation(llava_type)}
        elements = []
        image = data['image']
        # Process image
        try:
            raw_images = None
            if image:
                if type(image) == list:
                    raw_images = [
                        pil_img2rgb(item) for item in image
                    ]
                else:
                    raw_images = [
                        pil_img2rgb(image)
                    ]
        except:
            traceback.print_exc()

        for conversation in data['conversations']:
            if conversation['from'] == 'human':
                if '<image>' not in conversation['value']:
                    elements.append({
                        'type': 'text',
                        'has_loss': 0,
                        'text': conversation['value'],
                    })
                else:
                    text_list = conversation['value'].split('<image>')
                    for idx, text in enumerate(text_list):
                        if text.strip() != '':
                            elements.append({
                                'type': 'text',
                                'has_loss': 0,
                                'text': text.strip(),
                            })
                        if (idx != len(text_list) - 1):
                            elements.append({'type': 'image', 'image': raw_images[idx]})
                    elements.append({'type': 'goal_image',})
            elif conversation['from'] == 'gpt':
                elements.append({
                    'type': 'action',
                    'has_loss': 0,
                    'text': conversation['value'],
                })
        return elements

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        # Caculate global worker id and worker number
        global_worker_id = self.local_rank * num_workers + worker_id
        num_total_workers = self.world_size * num_workers

        # Create raw iterator
        iterator = self.dataset.as_numpy_iterator()
        # Set up sharded iterator
        sharded_iterator = itertools.islice(iterator, global_worker_id, None, num_total_workers)

        transform_stride = self.transform.stride
        for rlds_batch in sharded_iterator:
            # yield self.batch_transform(rlds_batch)
            data = self.batch_transform(rlds_batch)
            action = data['action']
            goal_image = data['goal_image']

            num_tokens = 0
            image_tensor_list = []
            text_ids_list = []
            action_list = []
            sequence_plan = []

            action_list.append(action) # modify

            # Process goal image
            try:
                raw_goal_images = None
                if goal_image:
                    if type(goal_image) == list:
                        raw_goal_images = [
                            pil_img2rgb(item) for item in goal_image
                        ]
                    else:
                        raw_goal_images = [
                            pil_img2rgb(goal_image)
                        ]
            except:
                traceback.print_exc()
                continue

            elements = self.change_format(data)
            for item in elements:
                if item['type'] == 'text':
                    text_data = item['text']
                    text_ids = self.tokenizer.encode(text_data)
                        
                    if len(text_ids) > 0:
                        text_ids_list.append(text_ids)
                        num_tokens += len(text_ids)
                        current_plan = {
                            'type': 'text',
                            'enable_cfg': 1,
                            'loss': item['has_loss'],
                            'special_token_loss': 0,
                            'special_token_label': None,
                        }
                        sequence_plan.append(current_plan)
                elif item['type'] == 'image':
                    sequence_plan.append(
                        {
                            'type': 'vae_image', 
                            'enable_cfg': 1, 
                            'loss': 0, 
                            'special_token_loss': 0,
                            'special_token_label': None,
                        }
                    )
                    image_tensor = self.transform(item['image'])
                    height, width = image_tensor.shape[1:]
                    num_tokens += width * height // self.transform.stride ** 2
                    image_tensor_list.append(image_tensor.clone())

                    sequence_plan.append(
                        {
                            'type': 'vit_image',
                            'enable_cfg': 1, 
                            'loss': 0,
                            'special_token_loss': 0,
                            'special_token_label': None,
                        },
                    )
                    vit_image_tensor = self.vit_transform(item['image'])
                    height, width = vit_image_tensor.shape[1:]
                    num_tokens += width * height // self.vit_transform.stride ** 2
                    image_tensor_list.append(vit_image_tensor)
                elif item['type'] == 'goal_image':
                    sequence_plan.append(
                        {
                            'type': 'vae_image', 
                            'enable_cfg': 0, 
                            'loss': 1, 
                            'special_token_loss': 0,
                            'special_token_label': None,
                        }
                    )
                    image_tensor = self.transform(raw_goal_images[0])
                    height, width = image_tensor.shape[1:]
                    num_tokens += width * height // self.transform.stride ** 2
                    image_tensor_list.append(image_tensor)

            has_loss = [item['loss'] for item in sequence_plan]
            if sum(has_loss) == 0:
                print(f'No loss defined, skipped.')
                continue

            # with open("${HALO_OUTPUT_DIR}/test_wbdata_key.txt", "a") as f:
            #     f.write(f"local_rank{self.local_rank} worker{worker_id} {key}\n")
            
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

    def __len__(self) -> int:
        # Effective Dataset Length = Number of samples until each dataset has completed at least one epoch
        return self.dataset_length 

    # === Explicitly Unused ===
    def __getitem__(self, idx: int) -> None:
        raise NotImplementedError("IterableDataset does not implement map-style __getitem__; see __iter__ instead!")