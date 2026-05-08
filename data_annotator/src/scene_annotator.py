import os
import json
import logging
from typing import List, Dict, Any, Optional
from src.qwen3vl_client import Qwen3VLClient
import argparse
import h5py
import numpy as np

class SceneAnnotator:
    def __init__(
        self,
        qwen_client: Qwen3VLClient,
        image_dir: str,
        instruction_path: Optional[str] = None,
        frame_prefix: str = "frame_",
        image_ext: str = ".jpg",
    ):
        self.qwen_client = qwen_client
        self.image_dir = image_dir
        self.frame_prefix = frame_prefix
        self.image_ext = image_ext
        
        self.instruction = ""

        if instruction_path and os.path.exists(instruction_path):
            try:
                with open(instruction_path, 'r', encoding='utf-8') as f:
                    example_data = json.load(f)
                if example_data.get("seen") and isinstance(example_data["seen"], list):
                    self.instruction = example_data["seen"][0]
                    logging.info(f"Loaded instruction: {self.instruction[:80]}...")
            except Exception as e:
                logging.error(f"Failed to load instruction file: {e}")

    def _get_image_path(self, frame_id: int) -> str:
        return os.path.join(self.image_dir, f"{self.frame_prefix}{frame_id:05d}{self.image_ext}")

    def _build_global_prompt(self, instruction: str = "") -> str:
        prompt = (
            "**Role:** You are an expert roboticist specializing in scene perception.  "
            "**Objective:** Provide a structured analysis of the visual scene for a robot arm. "
        )
        prompt += (f"**Primary Goal:** The robot arm's task is to {instruction}.  \n"
                   "Please analyze the provided image and generate a detailed description. ")
        
        prompt += (
            "The description should include: 1) The overall scene layout, 2) The position and orientation of objects, "
            "3) The robot's current state and posture, 4) The important attributes and current state of each key object, and 5) Any relevant spatial relationships between objects.  \n"
            "1-2 sentences per aspect are sufficient.  \n"
        )

        return prompt
    
    def annotate_scene(
        self,
        frame_id: int,
    ) -> Dict[str, Any]:
        
        image_path = self._get_image_path(frame_id)
        if not os.path.exists(image_path):
            logging.error(f"Image not found: {image_path}")
            return {}

        prompt_instruction = self.instruction
        prompt = self._build_global_prompt(prompt_instruction)

        response = self.qwen_client.annotate_clip(
            prompt=prompt,
            image_paths=[image_path]
        )

        logging.info(f"Raw response: {response}")

        return response 
    
def setup_logging(log_level: str = "INFO"):
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

def main():
    parser = argparse.ArgumentParser(description="Annotate scenes using Qwen3-VL and save to HDF5 with 'scene_description' field.")
    
    # Required I/O
    parser.add_argument("--annotation_json", type=str, required=True, help="Path to input annotation JSON")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing extracted frames (e.g., frame_00000.jpg)")
    parser.add_argument("--output_hdf5", type=str, required=True, help="Path to save HDF5 with 'scene_description' field")

    # Qwen3-VL server config
    parser.add_argument("--qwen_api_base", type=str, default="http://localhost:8000/v1", help="Qwen3-VL API base URL")
    parser.add_argument("--qwen_model_name", type=str, default="qwen3-vl", help="Qwen3-VL model name")

    parser.add_argument("--instruction_file", type=str, default=None, help="JSON file with instructions")
    
    # Image naming format
    parser.add_argument("--frame_prefix", type=str, default="frame_", help="Frame filename prefix")
    parser.add_argument("--image_ext", type=str, default=".jpg", help="Image file extension")
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )
    args = parser.parse_args()
    
    setup_logging(args.log_level)
    # Initialize Qwen client
    qwen_client = Qwen3VLClient(
        api_base=args.qwen_api_base,
        model_name=args.qwen_model_name
    )

    # Initialize SceneAnnotator
    annotator = SceneAnnotator(
        qwen_client=qwen_client,
        image_dir=args.image_dir,
        instruction_path=args.instruction_file,
        frame_prefix=args.frame_prefix,
        image_ext=args.image_ext
    )

    # Read raw annotations JSON
    with open(args.annotation_json, 'r', encoding='utf-8') as f:
        annotations: List[Dict[str, Any]] = json.load(f)

    if not isinstance(annotations, list):
        raise ValueError("Input JSON must be a list of annotation objects.")

    # Generate scene_description
    scene_description = annotator.annotate_scene(frame_id=0)
    first_ann = annotations[0] if annotations else {}
    total_subtasks = first_ann.get("total_subtasks", [])
    total_subtasks_str = json.dumps(total_subtasks, ensure_ascii=False)

    # === Per-frame fields ===
    per_frame_keys = {
        "frame_id",
        "current_subtask",
        "current_subtask_index",
        "remaining_subtasks",
        "moving_instruction"
    }

    # Write HDF5
    os.makedirs(os.path.dirname(os.path.abspath(args.output_hdf5)), exist_ok=True)
    with h5py.File(args.output_hdf5, 'w') as h5f:
        num_frames = len(annotations)

        # --- Global scalar datasets ---
        h5f.create_dataset(
            "scene_description",
            data=scene_description,
            dtype=h5py.string_dtype(encoding='utf-8')
        )
        h5f.create_dataset(
            "total_subtasks",
            data=total_subtasks_str,
            dtype=h5py.string_dtype(encoding='utf-8')
        )

        # --- Per-frame datasets ---
        for key in per_frame_keys:
            values = []
            for ann in annotations:
                val = ann.get(key, "")
                if isinstance(val, (list, dict)):
                    val = json.dumps(val, ensure_ascii=False)
                elif val is None:
                    val = ""
                else:
                    val = str(val)
                values.append(val)

            dt = h5py.string_dtype(encoding='utf-8')
            h5f.create_dataset(
                key,
                data=np.array(values, dtype=object),
                dtype=dt,
                compression="gzip"
            )

    logging.info(f"Saved HDF5 to {args.output_hdf5}")


if __name__ == "__main__":
    main()