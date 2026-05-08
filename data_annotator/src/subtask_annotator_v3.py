import os
import json
import logging
import re
from typing import List, Dict, Any, Optional
from src.postprocessor import safe_parse_json, convert_to_subtask_sequence
from src.subtask_annotator_v2 import SubtaskAnnotator  

class TwoStageSubtaskAnnotator(SubtaskAnnotator):
    """
    Two-stage subtask annotator.
    Stage 1: produce a coherent high-level subtask sequence (task planning).
    Stage 2: align each frame to the best-matching subtask in the sequence and emit reasoning.
    Emits the derived fields required for HDF5 export.
    """

    def _build_prompt_stage1(self, raw_annotations: List[Dict], has_trajectory: bool, context: Optional[str] = "") -> str:
        # --- Prompt copied verbatim from the previous implementation ---
        prompt = (
            "You are an expert in robotic task analysis. "
            "Your task is to infer the HIGH-LEVEL SUBTASK (i.e., the GOAL or PURPOSE) for the video frames provided. "
            "A subtask should be more fine-grained than the overall goal, yet more abstract than basic movements. "
            "The granularity of subtasks should be consistent."
            "**All subtasks MUST strictly align with the overall task goal.\n\n"
        )

        if context and context.strip():
            prompt += f"The overall goal of the task is: {context}\n\n"
        
        total_images = len(raw_annotations) + (1 if has_trajectory else 0)
        if has_trajectory:
            prompt += f"- Image {total_images}: Global motion trajectories of both arms over the entire task\n\n"

        prompt += "Per-frame image and arm actions (low-level, DO NOT copy these!):\n"
        for i, ann in enumerate(raw_annotations):
            frame_id = ann["frame"]
            left_desc = ann.get("left_arm_description") or ann.get("left_arm_discription") or "No action."
            right_desc = ann.get("right_arm_description") or ann.get("right_arm_discription") or "No action."
            prompt += (
                f"Frame {frame_id}:\n"
                f"- Frame image: image {i+1}\n"
                f"- Left arm action: {left_desc}\n"
                f"- Right arm action: {right_desc}\n\n"
            )

        prompt += (
            "### INSTRUCTIONS:\n"
            "1. **NEVER** output low-level motions (e.g., 'reach', 'grasp', 'move', 'idle', 'adjust').\n"
            "2. Describe WHAT is being achieved: e.g., 'Pick up cup', 'Place box on shelf'.\n"
            "3. For bimanual cooperation, output ONE unified goal (e.g., 'Insert peg into hole').\n"
            "4. Use the GLOBAL TRAJECTORY IMAGE (if provided) to understand the full task flow and ensure consistency.\n"
            "5. Keep phrases imperative, active voice.\n"

            
            '### VERY IMPORTANT:\n'
            'Based on the video and the overall task goal, identify how many objects are manipulated in total, and ensure that the output subtasks collectively involve exactly those objects.\n'
            "**The task proceeds OBJECT-BY-OBJECT. Each object must undergo a complete, closed-loop sequence"
            "(especially: pickup → place/stack/hold...) before any action on a new object begins.\n"
            "Ensure subtasks for the same object are contiguous and form a logical progression.\n "


            "### OUTPUT FORMAT:\n"
            "Return ONLY a JSON list of strings. NOTHING ELSE. NO EXPLANATIONS.\n"
            "Example: [\"Pick up red cup\", \"Pour water into cup\", \"Place cup on table\"]\n"
        )
        return prompt

    def _build_prompt_stage2(
        self,
        raw_annotations: List[Dict],
        has_trajectory: bool,
        subtask_sequence: List[str],
        context: Optional[str] = ""
    ) -> str:
        # 1. Persona: switch from analyst to robot controller
        prompt = (
            "You are the autonomous onboard controller of a robot. You are currently executing a task. "
            "Your goal is to reason through your actions in the FIRST PERSON ('I', 'me', 'my').\n\n"
        )

        if context and context.strip():
            prompt += f"My Overall Final Goal: {context}\n\n"

        prompt += f"Input sequence: {len(raw_annotations)} frames of my visual sensors and internal state.\n"
        
        prompt += "\n### MY PLANNED SUBTASK SEQUENCE:\n"
        for i, sub in enumerate(subtask_sequence, 1):
            prompt += f"{i}. {sub}\n"

        prompt += "\n### MY REGISTERED LOW-LEVEL ARM ACTIONS (Ground Truth):\n"
        for ann in raw_annotations:
            fid = ann["frame"]
            left_desc = ann.get("left_arm_description") or ann.get("left_arm_discription") or "No action."
            right_desc = ann.get("right_arm_description") or ann.get("right_arm_discription") or "No action."
            prompt += f"- Frame {fid}: Left: {left_desc}; Right: {right_desc}\n"

        prompt += (
            "\n### INSTRUCTIONS:\n"
            "1. **FIRST-PERSON PERSPECTIVE**: I must describe the reasoning from my own point of view. Use 'I' instead of 'the robot'.\n"
            "2. **PRESENT MOMENT REASONING**: For each segment, explain my internal decision-making logic based on what I see and my final goal.\n"
            "   - **Visual Observation**: What do I perceive right now? (e.g., 'I see my gripper is 2cm away from the handle').\n"
            "   - **Goal-Driven Inference**: Based on my final goal, why is this specific subtask the correct next step?\n"
            "   - **Movement Logic**: What specific movements am I executing (e.g., 'I am closing my fingers') to progress toward the subtask goal?\n"
            "3. **STRICT PHYSICAL ALIGNMENT**: I cannot claim I am starting a movement-based subtask (like 'Lift') if my low-level action logs show I am still 'idle'. I must wait for the actual physical transition.\n"
            "4. **NO TEMPORAL SUMMARIES**: Do not mention frame numbers or say 'After segment X'. Focus on the *internal state* of the current segment.\n"
            "5. **BREVITY**: Keep my internal monologue under **50 words** per segment.\n"
            "6. **CONTIGUOUS SEGMENTATION**: Ensure every frame is accounted for in my execution timeline.\n"

            "\n### OUTPUT FORMAT:\n"
            "Return ONLY a JSON list of objects. NOTHING ELSE.\n"
            "Format: [{\"subtask\": \"...\", \"frame\": [start, end], \"reasoning\": \"...\"}]\n"
            "Example:\n"
            "[\n"
            "  {\n"
            "    \"subtask\": \"Pick up red cup\",\n"
            "    \"frame\": [1, 35],\n"
            "    \"reasoning\": \"I perceive the red cup is directly beneath my gripper. To achieve my goal of clearing the table, I must first secure this object. I am now lowering my arm steadily and synchronizing my fingers to grasp the handle firmly before I begin the ascent.\"\n"
            "  }\n"
            "]\n"
        )
        return prompt

    def generate_subtask_sequence(self, raw_annotations: List[Dict[str, Any]], context: Optional[str] = None) -> List[str]:
        if not raw_annotations:
            return []
        raw_annotations = sorted(raw_annotations, key=lambda x: int(x["frame"]))
        final_context = context if context is not None else self.default_context
        has_trajectory = self.trajectory_path is not None

        image_paths = [self._get_image_path(int(ann["frame"])) for ann in raw_annotations if os.path.exists(self._get_image_path(int(ann["frame"])))]
        if has_trajectory:
            image_paths.append(self.trajectory_path)

        prompt = self._build_prompt_stage1(raw_annotations, has_trajectory, final_context)
        response = self.qwen_client.annotate_clip(image_paths=image_paths, prompt=prompt)
        
        parsed = safe_parse_json(response, default=[])
        cleaned = [str(s).strip() for s in parsed if str(s).strip().lower() not in {"idle", "none", "unknown", ""}]
        logging.info(f"Stage 1: Generated subtasks: {cleaned}")
        return cleaned

    def align_subtasks_to_frames(
        self,
        raw_annotations: List[Dict[str, Any]],
        subtask_sequence: List[str],
        context: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        raw_annotations = sorted(raw_annotations, key=lambda x: int(x["frame"]))
        image_paths = [self._get_image_path(int(ann["frame"])) for ann in raw_annotations if os.path.exists(self._get_image_path(int(ann["frame"])))]
        if self.trajectory_path:
            image_paths.append(self.trajectory_path)

        prompt = self._build_prompt_stage2(raw_annotations, bool(self.trajectory_path), subtask_sequence, context)
        response = self.qwen_client.annotate_clip(image_paths=image_paths, prompt=prompt)
        
        clean_response = response.strip()
        if "```" in clean_response:
            clean_response = re.sub(r'^```[a-zA-Z]*\n|```$', '', clean_response, flags=re.MULTILINE).strip()

        segments = safe_parse_json(clean_response, default=[])
        if not isinstance(segments, list) or not segments:
            return self._generate_fallback_frames(raw_annotations, subtask_sequence, context)

        return self._expand_and_process(segments, raw_annotations, subtask_sequence)

    def _expand_and_process(self, segments: List[Dict], raw_annotations: List[Dict], subtask_sequence: List[str]):
        """Map segment-level annotations to per-frame and inject HDF5-required derived fields."""
        frame_lookup = {}
        valid_subtasks = subtask_sequence
        
        # Pre-compute indices and remaining lists
        subtask_info_map = {sub: {"idx": i, "rem": subtask_sequence[i:]} for i, sub in enumerate(subtask_sequence)}

        for seg in segments:
            # Tolerate the 'subtask' key from the prompt
            sub = seg.get("subtask") or seg.get("current_subtask") or "Unknown"
            reason = seg.get("reasoning", "")
            f_range = seg.get("frame", [])

            # Sanity-fix
            if sub not in subtask_info_map and subtask_sequence:
                for valid_sub in subtask_sequence:
                    if valid_sub.lower() in sub.lower() or sub.lower() in valid_sub.lower():
                        sub = valid_sub
                        break
            
            info = subtask_info_map.get(sub, {"idx": 0, "rem": subtask_sequence})

            if len(f_range) == 2:
                for fid in range(int(f_range[0]), int(f_range[1]) + 1):
                    frame_lookup[fid] = {
                        "val": sub,
                        "idx": info["idx"],
                        "rem": info["rem"],
                        "reason": reason
                    }

        final_data = []
        default_sub = subtask_sequence[0] if subtask_sequence else "Perform task"
        current_info = {"val": default_sub, "idx": 0, "rem": subtask_sequence, "reason": "Executing."}

        for ann in raw_annotations:
            fid = int(ann["frame"])
            if fid in frame_lookup:
                current_info = frame_lookup[fid]
            
            final_data.append({
                "frame_id": fid,
                "reasoning": current_info["reason"],
                "current_subtask": current_info["val"],
                "current_subtask_index": current_info["idx"],
                "remaining_subtasks": current_info["rem"],
                "moving_instruction": (ann.get("left_arm_description") or ann.get("left_arm_discription") or "").strip() + " " +(ann.get("right_arm_description") or ann.get("right_arm_discription") or "").strip(),
            })
        return final_data

    def annotate_subtasks(self, raw_annotations: List[Dict[str, Any]], context: Optional[str] = None) -> List[Dict[str, Any]]:
        if not raw_annotations:
            return []
        subtask_sequence = self.generate_subtask_sequence(raw_annotations, context)
        
        if not subtask_sequence:
            logging.warning("Stage 1 failed. Using fallback.")
            # fallback_res = super().annotate_subtasks(raw_annotations, context)
            # if fallback_res:
            #     sub_name = fallback_res[0].get("subtask", "Perform task")
            #     for item in fallback_res:
            #         item["current_subtask"] = sub_name
            #         item["current_subtask_index"] = 0
            #         item["remaining_subtasks"] = [sub_name]
            #         item["moving_instruction"] = sub_name
            #     fallback_res[0]["total_subtasks"] = [sub_name]
            # return fallback_res

        aligned_result = self.align_subtasks_to_frames(raw_annotations, subtask_sequence, context)
        if aligned_result:
            aligned_result[0]["total_subtasks"] = subtask_sequence
            
        return aligned_result