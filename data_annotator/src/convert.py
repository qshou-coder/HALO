#convert.py

import h5py
import json
import numpy as np
import argparse

def hdf5_dataset_to_serializable(data):
    """Convert numpy scalars to native Python types for JSON serialization."""
    if isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, (np.integer, np.floating, np.bool_)):
        return data.item()
    elif isinstance(data, bytes):
        return data.decode('utf-8')
    else:
        return data

def main():
    parser = argparse.ArgumentParser(description="Extract endpose data from HDF5 file and save as JSON.")
    parser.add_argument("--hdf5_path", required=True, help="Path to the input HDF5 file")
    parser.add_argument("--json_path", required=True, help="Path to the output JSON file")

    args = parser.parse_args()

    with h5py.File(args.hdf5_path, 'r') as f:
        endpose_group = f['endpose']
        output = {}

        for key in ['left_endpose', 'left_gripper', 'right_endpose', 'right_gripper']:
            if key in endpose_group:
                dataset = endpose_group[key]
                output[key] = hdf5_dataset_to_serializable(dataset[()])

    # Write JSON file
    with open(args.json_path, 'w', encoding='utf-8') as out_file:
        json.dump(output, out_file, indent=2, ensure_ascii=False)

    print(f"Successfully saved endpose data to {args.json_path}")

if __name__ == "__main__":
    main()