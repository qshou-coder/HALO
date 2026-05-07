# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from .unified_dataset import SftWebDatasetIterableDataset, RLDSDataset, OpenXActionFlowDataset, SSv2VideoPredictionDataset, OpenXInverseDataset
from .realworld_dataset import RealTrajDataset
from .robotwin_dataset import HDF5IterableDataset

DATASET_REGISTRY = {
    'llava_next_data': SftWebDatasetIterableDataset,
    'open_x_embodiment': OpenXActionFlowDataset,
    'RoboTwin': HDF5IterableDataset,
    'ssv2': SSv2VideoPredictionDataset,
    'open_x_inverse': OpenXInverseDataset,
    'real_traj': RealTrajDataset,
}


DATASET_INFO = {
    'llava_next_data':{
        'llava_next':{
            'data_dir': '${HALO_DATA_DIR}/Pretrain_Data/LLava-NeXT-Data', # path of the parquet files
            'num_files': 312, # number of data units to be sharded across all ranks and workers
            'num_total_samples': 779289, # number of total samples in the dataset
        }
    },
    'open_x_embodiment':{
        'oxe_data':{
            'data_dir': '${HALO_DATA_DIR}/OpenVLA_Dataset', # path of open x dataset
            'data_root_dir': '${HALO_DATA_DIR}/OpenVLA_Dataset',
        }
    },
    'ssv2': {
        'ssv2_data': {
            'data_dir': '${HALO_MODEL_DIR}/dataset/SSv2/sthv2/sthv2/videos',
            'jsonl_path': '${HALO_MODEL_DIR}/dataset/SSv2/sthv2/sthv2/label/train.json',
            'num_total_samples': 168913
        }
    },
    'open_x_inverse':{
        'oxe_inverse':{
            'data_dir': '${HALO_MODEL_DIR}/dataset/OpenVLA-Dataset',
            'data_root_dir': '${HALO_MODEL_DIR}/dataset/OpenVLA-Dataset',
        }
    },
    'RoboTwin':{
        'robotwin_data':{
            'data_dir': '${HALO_DATA_DIR}/unlabeled_robotwin_data',
        }
    },
    'real_traj':{
        'real_traj_data':{
            'data_dir': '${HALO_DATA_DIR}/demo_data/cot_data/table_bussing_v1',
        }
    },
}
