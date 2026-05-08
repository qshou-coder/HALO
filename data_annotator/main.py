# main.py

import argparse
import json
import os
import logging
from pathlib import Path
from src.postprocessor import save_annotations_to_hdf5
from src.scene_annotator import SceneAnnotator

# from src.subtask_annotator_v2 import SubtaskAnnotator
from src.subtask_annotator_v3 import TwoStageSubtaskAnnotator as SubtaskAnnotator
from src.qwen3vl_client import Qwen3VLClient

def setup_logging(log_level: str = "INFO"):
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

def parse_args():
    parser = argparse.ArgumentParser(
        description="Annotate high-level subtasks for robot action segments using Qwen3-VL."
    )

    # Required I/O
    parser.add_argument(
        "--annotation_json",
        type=str,
        required=True,
        help="Path to input annotation JSON "
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="Directory containing extracted frames (e.g., frame_00000.jpg)"
    )
    parser.add_argument(
        "--output_json",
        type=str,
        required=True,
        help="Path to save enhanced JSON with 'subtask' field"
    )

    # Qwen3-VL server config
    parser.add_argument(
        "--qwen_api_base",
        type=str,
        default="http://localhost:8000/v1",
        help="Qwen3-VL API base URL (default: http://localhost:8000/v1)"
    )
    parser.add_argument(
        "--qwen_model_name",
        type=str,
        default="qwen3-vl",
        help="Qwen3-VL model name (default: qwen3-vl)"
    )

   
    parser.add_argument(
        "--instruction_file",
        type=str,
        default=None,
        help="Optional JSON file with 'seen' list (e.g., for context extraction)"
    )

    # Image naming format
    parser.add_argument(
        "--frame_prefix",
        type=str,
        default="frame_",
        help="Frame filename prefix (default: 'frame_')"
    )
    parser.add_argument(
        "--image_ext",
        type=str,
        default=".jpg",
        help="Image file extension (default: '.jpg')"
    )

    # Segmentation control
    parser.add_argument(
        "--max_frames_per_call",
        type=int,
        default=999,
        help="Max number of frames per Qwen call "
    )

    parser.add_argument(
        "--trajectory_image_path",
        type=str,
        default="None",
        help="trajectory_image_path"
    )

    # Logging
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )

    return parser.parse_args()

def main():
    args = parse_args()
    setup_logging(args.log_level)

    # Validate inputs
    if not os.path.exists(args.annotation_json):
        raise FileNotFoundError(f"Annotation JSON not found: {args.annotation_json}")
    if not os.path.exists(args.image_dir):
        raise FileNotFoundError(f"Image directory not found: {args.image_dir}")
    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)

    # Load raw annotations
    logging.info(f"Loading annotations from {args.annotation_json}")
    with open(args.annotation_json, "r", encoding="utf-8") as f:
        raw_annotations = json.load(f)

    # Initialize Qwen client
    logging.info(f"Initializing Qwen3-VL client: {args.qwen_model_name} @ {args.qwen_api_base}")
    qwen_client = Qwen3VLClient(
        api_base=args.qwen_api_base,
        model_name=args.qwen_model_name
    )

    # Initialize SubtaskAnnotator
    subtask_annotator = SubtaskAnnotator(
        qwen_client=qwen_client,
        image_dir=args.image_dir,
        instruction_path=args.instruction_file,
        frame_prefix=args.frame_prefix,
        image_ext=args.image_ext,
        max_frames_per_call=args.max_frames_per_call,
        trajectory_image_path=args.trajectory_image_path
    )
    scene_annotator = SceneAnnotator(
        qwen_client=qwen_client,
        image_dir=args.image_dir,
        instruction_path=args.instruction_file,
        frame_prefix=args.frame_prefix,
        image_ext=args.image_ext
    )
    # Run annotation
    logging.info("Starting subtask annotation...")
    enhanced_annotations = subtask_annotator.annotate_subtasks(raw_annotations)
    scene_description = scene_annotator.annotate_scene(frame_id=0)


    output_data = {
        "scene_description": scene_description,
        "frame_annotations": enhanced_annotations
    }

    # Save JSON
    logging.info(f"Saving enhanced annotations to {args.output_json}")
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    # Save HDF5
    # hdf5_path = os.path.splitext(args.output_json)[0] + ".hdf5"
    # logging.info(f"Saving HDF5 data to {hdf5_path}")
    # save_annotations_to_hdf5(enhanced_annotations, scene_description, hdf5_path)
    
    logging.info("✅ Subtask annotation completed successfully.")
    
if __name__ == "__main__":
    main()