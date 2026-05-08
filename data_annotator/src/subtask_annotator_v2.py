import os
import json
import logging
from typing import List, Dict, Any, Optional
from src.postprocessor import safe_parse_json, convert_to_subtask_sequence
from src.qwen3vl_client import Qwen3VLClient


class SubtaskAnnotator:
    def __init__(
        self,
        qwen_client: Qwen3VLClient,
        image_dir: str,
        instruction_path: Optional[str] = None,
        frame_prefix: str = "frame_",
        image_ext: str = ".jpg",
        max_frames_per_call: int = 999,  
        trajectory_image_path: Optional[str] = None,  
    ):
        self.qwen_client = qwen_client
        self.image_dir = image_dir
        self.frame_prefix = frame_prefix
        self.image_ext = image_ext
        self.max_frames_per_call = max_frames_per_call

        # === Resolve trajectory image path ===
        self.trajectory_path = None
        if trajectory_image_path is not None:
            if os.path.exists(trajectory_image_path):
                self.trajectory_path = trajectory_image_path
                logging.info(f"Using explicit trajectory image: {self.trajectory_path}")
            else:
                logging.warning(f"Explicit trajectory_image_path provided but not found: {trajectory_image_path}")
        else:
            # Try default path: {image_dir}/trajectory{image_ext}
            default_traj = os.path.join(image_dir, f"trajectory{image_ext}")
            if os.path.exists(default_traj):
                self.trajectory_path = default_traj
                logging.info(f"Using default trajectory image: {self.trajectory_path}")
            else:
                logging.info("No trajectory image found or provided. Skipping.")

        # === Load instructions ===
        self.default_context = ""
        if instruction_path and os.path.exists(instruction_path):
            try:
                with open(instruction_path, 'r', encoding='utf-8') as f:
                    example_data = json.load(f)
                if example_data.get("seen") and isinstance(example_data["seen"], list):
                    self.default_context = " ,or , ".join(example_data["seen"])
                    # self.default_context = example_data["seen"][0]
                    logging.info(f"Loaded default context: {self.default_context[:80]}...")
            except Exception as e:
                logging.error(f"Failed to load example file: {e}")

    def _get_image_path(self, frame_id: int) -> str:
        return os.path.join(self.image_dir, f"{self.frame_prefix}{frame_id:05d}{self.image_ext}")

    def _build_prompt(self, raw_annotations: List[Dict], has_trajectory: bool, context: str = "") -> str:
        prompt = (
            "You are an expert in robotic task analysis. "
            "Your task is to infer the HIGH-LEVEL SUBTASK (i.e., the GOAL or PURPOSE) for each video frame. "
            "**All subtasks MUST strictly align with the overall task context and goal.**\n\n"
        )

        if context.strip():
            prompt += f"Overall task context: {context}\n\n"

        total_images = len(raw_annotations) + (1 if has_trajectory else 0)
        prompt += f"There are {total_images} input images in total:\n"
        for i in range(len(raw_annotations)):
            prompt += f"- Image {i+1}: Frame {raw_annotations[i]['frame']}\n"
        if has_trajectory:
            prompt += f"- Image {total_images}: Global motion trajectories of both arms over the entire task\n\n"
        else:
            prompt += "\n"

        prompt += "Per-frame arm actions (low-level, DO NOT copy these!):\n"
        for ann in raw_annotations:
            frame_id = ann["frame"]
            left = ann.get("left_arm", ["idle"])[0]
            right = ann.get("right_arm", ["idle"])[0]
            left_desc = ann.get("left_arm_discription", "No action.")
            right_desc = ann.get("right_arm_discription", "No action.")
            prompt += (
                f"Frame {frame_id}:\n"
                f"- Left arm action: {left}\n"
                f"- Left description: {left_desc}\n"
                f"- Right arm action: {right}\n"
                f"- Right description: {right_desc}\n\n"
            )

        prompt += (
            "### INSTRUCTIONS:\n"
            "1. For EACH frame above, assign ONE high-level subtask that reflects the ROBOT'S INTENT.\n"
            "2. **NEVER** output low-level motions (e.g., 'reach', 'grasp', 'move', 'idle', 'adjust').\n"
            "3. Describe WHAT is being achieved: e.g., 'Pick up cup', 'Place box on shelf'.\n"
            "4. If the robot is idle or performing non-goal actions, REPEAT the last meaningful subtask.\n"
            "5. For bimanual cooperation, output ONE unified goal (e.g., 'Insert peg into hole').\n"
            "6. Use the GLOBAL TRAJECTORY IMAGE (if provided) to understand the full task flow and ensure consistency.\n"
            "7. Keep phrases concise (3–8 words), imperative, active voice: [ACTION] [OBJECT] [LOCATION].\n"
            "8. **Output exactly one subtask per input frame, in the same order.**\n\n"
            "### OUTPUT FORMAT:\n"
            "Return ONLY a JSON list of strings. NOTHING ELSE. NO EXPLANATIONS.\n"
            f"List length MUST be exactly {len(raw_annotations)}.\n"
        )
        return prompt

    def annotate_subtasks(
        self,
        raw_annotations: List[Dict[str, Any]],
        context: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if not raw_annotations:
            return []

        raw_annotations = sorted(raw_annotations, key=lambda x: int(x["frame"]))
        final_context = context if context is not None else self.default_context
        has_trajectory = self.trajectory_path is not None

        # Prepare frame image paths
        image_paths_base = []
        for ann in raw_annotations:
            frame_id = int(ann["frame"])
            img_path = self._get_image_path(frame_id)
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Frame image not found: {img_path}")
            image_paths_base.append(img_path)

        all_subtasks = []
        total_frames = len(raw_annotations)

        for start in range(0, total_frames, self.max_frames_per_call):
            end = min(start + self.max_frames_per_call, total_frames)
            batch_annotations = raw_annotations[start:end]
            batch_image_paths = image_paths_base[start:end]

            # Append global trajectory image if available
            if has_trajectory:
                batch_image_paths = batch_image_paths + [self.trajectory_path]

            prompt = self._build_prompt(batch_annotations, has_trajectory=has_trajectory, context=final_context)
            logging.info(f"Annotating batch: frames {start+1} to {end} "
                         f"({len(batch_image_paths)} images, trajectory={'YES' if has_trajectory else 'NO'})")

            response = self.qwen_client.annotate_clip(image_paths=batch_image_paths, prompt=prompt)
            parsed = safe_parse_json(response, default=[])
            print(f"Qwen response parsed: {parsed}")
            if not isinstance(parsed, list) or len(parsed) != len(batch_annotations):
                logging.error(f"Invalid response for batch. Expected {len(batch_annotations)}, got {len(parsed) if isinstance(parsed, list) else 'non-list'}")
                parsed = ["unknown"] * len(batch_annotations)

            all_subtasks.extend([str(s).strip() for s in parsed])

        # Post-process: replace idle/unknown with last valid subtask
        result_frames = []
        last_valid_subtask = "idle"
        for ann, subtask in zip(raw_annotations, all_subtasks):
            clean_sub = subtask.strip()
            if not clean_sub or clean_sub.lower() in {"idle", "unknown", "none", "no action", "n/a", "nothing"}:
                clean_sub = last_valid_subtask
            else:
                last_valid_subtask = clean_sub

            frame_entry = {
                "frame_id": int(ann["frame"]),
                "subtask": clean_sub,
                "left_arm": ann.get("left_arm", ["idle"]),
                "right_arm": ann.get("right_arm", ["idle"]),
                "left_arm_discription": ann.get("left_arm_discription", "").strip(),
                "right_arm_discription": ann.get("right_arm_discription", "").strip(),
            }
            result_frames.append(frame_entry)

        return convert_to_subtask_sequence(result_frames)