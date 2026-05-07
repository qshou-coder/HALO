import os
import h5py
import numpy as np
import random
from pathlib import Path
from .data_utils import pil_img2rgb
from .distributed_iterable_dataset import DistributedIterableDataset
from PIL import Image
import traceback
import cv2
from itertools import groupby
import re

class RealTrajDataset(DistributedIterableDataset):
    def __init__(
        self, dataset_name, transform, vit_transform, tokenizer, data_dir_list, num_used_data, instruction,
        local_rank=0, world_size=1, num_workers=8, data_status=None,
        use_subtask=False, use_goal_image=False,
        shuffle_lines=False, shuffle_seed=0,
    ):
        """
        data_dir_list: list of directories containing .hdf5 files
        num_used_data: list of number of .hdf5 files to use from each directory. Use -1 or None to use all files.
        instruction: str, the instruction to use for all trajectories.
        """
        super().__init__(dataset_name, local_rank, world_size, num_workers)
        self.transform = transform
        self.vit_transform = vit_transform
        self.tokenizer = tokenizer
        self.data_status = data_status
        self.CHUNK_SIZE = 50
        self.IMG_HISORY_SIZE = 3
        self.instruction = instruction
        self.use_subtask = use_subtask
        self.use_goal_image = use_goal_image

        self.data_paths = self._get_hdf5_files_from_dirs(data_dir_list, num_used_data)
        if not self.data_paths:
            raise ValueError("No .hdf5 files found in the given directories. Please check your paths and configuration.")
        
        # self.dataset = self._build_pipeline(final_hdf5_path_list)
        self.set_epoch()

    def _get_hdf5_files_from_dirs(self, data_dir_list, num_used_data):
        final_hdf5_list = []
        for dir_path, num_hdf5s in zip(data_dir_list, num_used_data):
            if not os.path.isdir(dir_path):
                print(f"Directory {dir_path} does not exist. Skipping.")
                continue

            dir_path = Path(dir_path)
            try:
                # hdf5_files_in_dir = sorted([f for f in os.listdir(dir_path) if f.endswith('.hdf5')])
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
                    print(f"Only {len(hdf5_files_in_dir)} .hdf5 files found in directory {dir_path}, but {num_hdf5s} were requested. Using all available files.")
                selected_files = hdf5_files_in_dir[:num_hdf5s]

            full_path_files = [os.path.join(dir_path, f) for f in selected_files]
            final_hdf5_list.extend(full_path_files)

        return final_hdf5_list
    
    def parse_hdf5_file(self, file_path):
        """[Modify] Parse a hdf5 file to generate a training sample at
            a random timestep.

        Args:
            file_path (str): the path to the hdf5 file

        Returns:
            valid (bool): whether the episode is valid, which is useful for filtering.
                If False, this episode will be dropped.
            dict: a dictionary containing the training sample,
                {
                    "meta": {
                        "dataset_name": str,    # the name of your dataset.
                        "#steps": int,          # the number of steps in the episode,
                                                # also the total timesteps.
                        "instruction": str      # the language instruction for this episode.
                    },
                    "step_id": int,             # the index of the sampled step,
                                                # also the timestep t.
                    "state": ndarray,           # state[t], (1, STATE_DIM).
                    "state_std": ndarray,       # std(state[:]), (STATE_DIM,).
                    "state_mean": ndarray,      # mean(state[:]), (STATE_DIM,).
                    "state_norm": ndarray,      # norm(state[:]), (STATE_DIM,).
                    "actions": ndarray,         # action[t:t+CHUNK_SIZE], (CHUNK_SIZE, STATE_DIM).
                    "state_indicator", ndarray, # indicates the validness of each dim, (STATE_DIM,).
                    "cam_high": ndarray,        # external camera image, (IMG_HISORY_SIZE, H, W, 3)
                                                # or (IMG_HISORY_SIZE, 0, 0, 0) if unavailable.
                    "cam_high_mask": ndarray,   # indicates the validness of each timestep, (IMG_HISORY_SIZE,) boolean array.
                                                # For the first IMAGE_HISTORY_SIZE-1 timesteps, the mask should be False.
                    "cam_left_wrist": ndarray,  # left wrist camera image, (IMG_HISORY_SIZE, H, W, 3).
                                                # or (IMG_HISORY_SIZE, 0, 0, 0) if unavailable.
                    "cam_left_wrist_mask": ndarray,
                    "cam_right_wrist": ndarray, # right wrist camera image, (IMG_HISORY_SIZE, H, W, 3).
                                                # or (IMG_HISORY_SIZE, 0, 0, 0) if unavailable.
                                                # If only one wrist, make it right wrist, plz.
                    "cam_right_wrist_mask": ndarray
                } or None if the episode is invalid.
        """
        with h5py.File(file_path, "r") as f:
            qpos = f["observations"]["qpos"][:]
            # left_arm_dim = f["observations"]["left_arm_dim"][:] # (114,) all of them are 6
            # right_arm_dim = f["observations"]["right_arm_dim"][:] # (114,) all of them are 6
            num_steps = qpos.shape[0]
            action_dim = qpos.shape[1]

            # [Optional] We skip the first few still steps
            EPS = 1e-2
            # Get the idx of the first qpos whose delta exceeds the threshold
            qpos_delta = np.abs(qpos - qpos[0:1])
            indices = np.where(np.any(qpos_delta > EPS, axis=1))[0]
            if len(indices) > 0:
                first_idx = indices[0]
            else:
                raise ValueError("Found no qpos that exceeds the threshold.")

            # We randomly sample a timestep
            step_id = np.random.randint(first_idx - 1, num_steps)

            instruction = self.instruction

            # You can also use precomputed language embeddings (recommended)
            # instruction = "path/to/lang_embed.pt"
            # instructions_path = os.path.join(dir_path, "instructions")
            # instructions_names = []

            # for filename in os.listdir(instructions_path):
            #     # Check if filename ends with .pt
            #     if filename.endswith(".pt"):
            #         instructions_names.append(os.path.join(instructions_path, filename))
            # instruction = np.random.choice(instructions_names)
            # print(f"choose {instruction} file as instruction.")
            # Assemble the meta

            meta = {
                "dataset_name": self.dataset_name,
                "#steps": num_steps,
                "step_id": step_id,
                "instruction": instruction,
            }

            thought = ""
            if self.use_subtask: 
                left_primitives = f["Primitives"]["left"]["text"][step_id:step_id + self.CHUNK_SIZE]
                right_primitives = f["Primitives"]["right"]["text"][step_id:step_id + self.CHUNK_SIZE]
            
                # 2. Define an aggressive sanitizer function
                def bulletproof_clean(p):
                    if isinstance(p, bytes):
                        s = p.decode('utf-8', errors='ignore')
                    else:
                        s = str(p)
                    # Force-strip \x00
                    s = s.replace('\x00', '')
                    # Lowercase to remove case differences
                    s = s.lower()
                    # Regex: collapse any invisible whitespace (spaces, tabs, \xa0, etc.) into a single space
                    s = re.sub(r'\s+', ' ', s)
                    # Strip trailing/leading whitespace
                    return s.strip()

                # 3. Sanitize every frame and drop entirely-empty ("") bad frames
                # Important: an empty string between duplicates (e.g. ["move backward", "", "move backward"]) would prevent groupby from merging them
                left_strs = [bulletproof_clean(p) for p in left_primitives if bulletproof_clean(p)]
                right_strs = [bulletproof_clean(p) for p in right_primitives if bulletproof_clean(p)]

                # 4. After sanitization, all strings are clean and uniform; now dedupe adjacent values
                left_str = ', '.join([k for k, _ in groupby(left_strs)]) if left_strs else "none"
                right_str = ', '.join([k for k, _ in groupby(right_strs)]) if right_strs else "none"
            
                # 5. Assemble output
                thought = f"[LEFT]: {left_str}; [RIGHT]: {right_str}"
            # print(f"thought: {thought}")

            # Rescale gripper to [0, 1]
            # qpos = qpos / np.array([[1 for i in range(left_arm_dim[0] + 1 + right_arm_dim[0] + 1)]])
            # target_qpos = f["action"][step_id:step_id + self.CHUNK_SIZE] / np.array(
            #     [[1 for i in range(left_arm_dim[0] + 1 + right_arm_dim[0] + 1)]])
            target_qpos = f["action"][step_id:step_id + self.CHUNK_SIZE]

            # Parse the state and action
            state = qpos[step_id:step_id + 1]
            state_std = np.std(qpos, axis=0)
            state_mean = np.mean(qpos, axis=0)
            state_norm = np.sqrt(np.mean(qpos**2, axis=0))
            actions = target_qpos
            if actions.shape[0] < self.CHUNK_SIZE:
                # Pad the actions using the last action
                actions = np.concatenate(
                    [
                        actions,
                        np.tile(actions[-1:], (self.CHUNK_SIZE - actions.shape[0], 1)),
                    ],
                    axis=0,
                )

            # Fill the state/action into the unified vector

            # def fill_in_state(values):
            #     # Target indices corresponding to your state space
            #     # In this example: 6 joints + 1 gripper for each arm
            #     UNI_STATE_INDICES = (
            #         [STATE_VEC_IDX_MAPPING[f"left_arm_joint_{i}_pos"]
            #          for i in range(left_arm_dim[0])] + [STATE_VEC_IDX_MAPPING["left_gripper_open"]] +
            #         [STATE_VEC_IDX_MAPPING[f"right_arm_joint_{i}_pos"]
            #          for i in range(right_arm_dim[0])] + [STATE_VEC_IDX_MAPPING["right_gripper_open"]])
            #     uni_vec = np.zeros(values.shape[:-1] + (self.STATE_DIM, ))
            #     uni_vec[..., UNI_STATE_INDICES] = values
            #     return uni_vec

            # state = fill_in_state(state)
            # state_indicator = fill_in_state(np.ones_like(state_std))
            # state_std = fill_in_state(state_std)
            # state_mean = fill_in_state(state_mean)
            # state_norm = fill_in_state(state_norm)
            # # If action's format is different from state's,
            # # you may implement fill_in_action()
            # actions = fill_in_state(actions)

            # Parse the images
            def parse_img(key):
                imgs = []
                for i in range(max(step_id - self.IMG_HISORY_SIZE + 1, 0), step_id + 1):
                    img= f["observations"]["images"][key][i]
                    # img = cv2.imdecode(np.frombuffer(img_bits, np.uint8), cv2.IMREAD_COLOR)
                    imgs.append(img)
                # Using goal image predicting strategy
                if self.use_goal_image:
                    goal_image=f["observations"]["images"][key][min(step_id+self.CHUNK_SIZE, num_steps-1)]
                    # goal_image = cv2.imdecode(np.frombuffer(goal_image_bits, np.uint8), cv2.IMREAD_COLOR)
                    imgs.append(goal_image)
                imgs = np.stack(imgs)

                if self.use_goal_image:
                    if imgs.shape[0] < self.IMG_HISORY_SIZE + 1:
                        # Pad the images using the first image
                        imgs = np.concatenate(
                            [
                                np.tile(
                                    imgs[:1],
                                    (self.IMG_HISORY_SIZE + 1 - imgs.shape[0], 1, 1, 1),
                                ),
                                imgs,
                            ],
                            axis=0,
                        )
                else:
                    if imgs.shape[0] < self.IMG_HISORY_SIZE:
                        # Pad the images using the first image
                        imgs = np.concatenate(
                            [
                                np.tile(
                                    imgs[:1],
                                    (self.IMG_HISORY_SIZE - imgs.shape[0], 1, 1, 1),
                                ),
                                imgs,
                            ],
                            axis=0,
                        )
                return imgs

            # `cam_high` is the external camera image
            cam_high = parse_img("cam_high")
            # For step_id = first_idx - 1, the valid_len should be one
            valid_len = min(step_id - (first_idx - 1) + 1, self.IMG_HISORY_SIZE)
            cam_high_mask = np.array([False] * (self.IMG_HISORY_SIZE - valid_len) + [True] * valid_len)
            cam_left_wrist = parse_img("cam_left_wrist")
            cam_left_wrist_mask = cam_high_mask.copy()
            cam_right_wrist = parse_img("cam_right_wrist")
            cam_right_wrist_mask = cam_high_mask.copy()

            # Return the resulting sample
            # For unavailable images, return zero-shape arrays, i.e., (IMG_HISORY_SIZE, 0, 0, 0)
            # E.g., return np.zeros((self.IMG_HISORY_SIZE, 0, 0, 0)) for the key "cam_left_wrist",
            # if the left-wrist camera is unavailable on your robot
            return True, {
                "meta": meta,
                "state": state,
                "state_std": state_std,
                "state_mean": state_mean,
                "state_norm": state_norm,
                "actions": actions, # check
                "thought": thought, # check
                "cam_high": cam_high, # check
                "cam_high_mask": cam_high_mask, # check
                "cam_left_wrist": cam_left_wrist,
                "cam_left_wrist_mask": cam_left_wrist_mask,
                "cam_right_wrist": cam_right_wrist,
                "cam_right_wrist_mask": cam_right_wrist_mask,
            }

    def _build_dataset(self, hdf5_files_list):
        data_list = []
        chunk_size = 50

        for hdf5_file in hdf5_files_list:
            # with h5py.File(hdf5_file, 'r') as hf:
            #     action_list = hf['action'][:]
            #     image_list = hf['low_cam_image'][:]
            #     # We only apply THE FIRST INSTRUCTION in this simplist case
            #     instruction = hf['seen'][0]

            #     for idx in range(len(action_list)):
            #         object = {}

            #         if idx <= len(action_list) - chunk_size:
            #             object['action'] = action_list[idx:idx+chunk_size]
            #         else:
            #             object['action'] = action_list[idx:]

            #         if idx >= 2:
            #             object['image'] = image_list[idx-2:idx+1]
            #         else:
            #             object['image'] = [image_list[max(idx-2, 0)], image_list[max(idx-1, 0)], image_list[idx]]
            #         object['instruction'] = instruction

            #         data_list.append(object)
            valid, data = self.parse_hdf5_file(hdf5_file)
            if valid:
                data_list.append(data)
            
        random.shuffle(data_list)
        return data_list
    
    def change_format(self, data, images):
        ## --- Notice that data is of this format ---
        ## {action:..., image:..., instruction: ...}
        elements = []
        num_images = len(images)
        for idx in range(num_images - 1):
            elements.append({'type': 'image', 'has_loss': 0, 'image': images[idx]})
        elements.append({
                'type': 'text',
                'has_loss': 0,
                'text': f"You are an embodied agent. Your task is to: {data['meta']['instruction'][0].lower().rstrip('.')}. Plan and act.",
                'special_token_loss': 0,
        })
        # print(data["meta"]["instruction"][0])
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

    def __iter__(self):
        hdf5_files_for_this_worker, worker_id = self.get_data_paths_per_worker()
        if not hdf5_files_for_this_worker:
            print(f"Rank-{self.local_rank} worker-{worker_id} dataset-{self.dataset_name}: "
                  f"No hdf5 files for this worker.")
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
              f"hdf5 files for this worker: {len(hdf5_files_for_this_worker)}")

        # dataset_for_this_worker = self._build_dataset(hdf5_files_for_this_worker)
        while True:
            # _build_dataset returns a shuffled list, the structure is like: 
            # [{"actions", "cam_high", "cam_high_mask"}, ...]
            for data_item in self._build_dataset(hdf5_files_for_this_worker):
                num_tokens = 0
                image_tensor_list = []
                text_ids_list = []
                action_list = []
                sequence_plan = []

                try:
                    raw_images = None
                    if 'cam_high' in data_item:
                        if type(data_item['cam_high']) == np.ndarray:
                            raw_images = []
                            # cam_mask = data_item['cam_high_mask']
                            # for idx in range(len(cam_mask)):
                            #     if cam_mask[idx]:
                            #         raw_images.append(pil_img2rgb(Image.fromarray(data_item['cam_high'][idx])))
                            #         # raw_images[idx].save(f'output_1.jpg')
                            #     else:
                            #         height, width, _ = data_item['cam_high'][0].shape
                            #         random_image = np.uint8(np.random.normal(128, 50, (height, width, 3)).clip(0, 255))
                            #         raw_images.append(pil_img2rgb(Image.fromarray(random_image)))
                            #         # raw_images[idx].save(f'output.jpg')
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

                elements = self.change_format(data_item, raw_images)

                for item in elements:
                    if item['type'] == 'text':
                        text_data = item['text']
                        # text_ids_raw = self.tokenizer.encode(text_data)
                        text_ids = self.tokenizer.encode(text_data)

                        # if item.get('has_loss', 0) == 0:
                        #     assert len(text_ids_raw) <= self.padding_length
                        #     text_ids = [self.pad_token_id] * (self.padding_length - len(text_ids_raw)) + text_ids_raw
                        # else:
                        #     text_ids = text_ids_raw
                        
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
                        if self.use_goal_image:
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
                    elif item['type'] == 'action':
                        action_list.append(item['action'])
                        num_tokens += len(item['action'])

                        current_plan = {
                            'type': 'action',
                            'enable_cfg': 0,
                            'loss': item['has_loss'],
                            'special_token_loss': 0,
                            'special_token_label': None,
                        }
                        sequence_plan.append(current_plan)

                # # Test
                # print(f"robotwin length: {num_tokens}")
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

            row_start_id = 0
            print(f"{self.dataset_name} repeat in rank-{self.local_rank} worker-{worker_id}")