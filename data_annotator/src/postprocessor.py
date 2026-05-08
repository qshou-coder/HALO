# src/postprocessor.py
import json
import logging
import re
from typing import Any, Optional, Dict, List, Union
import h5py
import json
import numpy as np
from pathlib import Path

def safe_parse_json(json_string: str, default: Any = None) -> Optional[Any]:
    """
    Safely parse a JSON string, tolerating common format issues.
    
    Args:
        json_string: JSON string to parse
        default: value returned on parse failure
    
    Returns:
        Parsed Python object, or `default` on failure.
    """
    if not json_string or not isinstance(json_string, str):
        logging.warning(f"Invalid JSON input: {json_string}")
        return default
    
    # Preprocess: strip possible Markdown code-fence markers
    cleaned_string = json_string.strip()
    
    # Remove ```json ... ``` or ``` ... ``` fences
    if cleaned_string.startswith('```'):
        # Take the content after the first ``` fence
        parts = cleaned_string.split('```', 2)
        if len(parts) >= 2:
            content = parts[1].strip()
            # If the first line is 'json', skip to the next line
            if content.startswith('json'):
                content = '\n'.join(content.split('\n')[1:]).strip()
            cleaned_string = content
    
    # Remove possible leading explanatory text
    # Match content following e.g. "Here is the JSON response:"
    json_match = re.search(r'\{.*\}|\[.*\]', cleaned_string, re.DOTALL)
    if json_match:
        cleaned_string = json_match.group(0)
    
    # Clean up: drop redundant whitespace and newlines
    cleaned_string = cleaned_string.strip()
    
    # Ensure the string starts with { or [ and ends with } or ]
    if not (cleaned_string.startswith(('{', '[')) and cleaned_string.endswith(('}', ']'))):
        logging.warning(f"String doesn't look like valid JSON: {cleaned_string[:100]}...")
        return default
    
    try:
        # Try to parse as JSON
        result = json.loads(cleaned_string)
        return result
    except json.JSONDecodeError as e:
        logging.error(f"JSON decode error: {e}")
        logging.debug(f"Problematic JSON string: {cleaned_string[:200]}...")
        
        # Fall back to a more lenient parser
        return _lenient_json_parse(cleaned_string, default)
    except Exception as e:
        logging.error(f"Unexpected error during JSON parsing: {e}")
        return default


def _lenient_json_parse(json_string: str, default: Any = None) -> Optional[Any]:
    """
    Lenient JSON parser that fixes common formatting mistakes.
    """
    # Try to fix common JSON issues
    try:
        # Fix single-quoted strings
        fixed_string = json_string.replace("'", '"')
        
        # Strip trailing commas
        fixed_string = re.sub(r',(\s*[}\]])', r'\1', fixed_string)
        
        # Strip comments (simple version)
        fixed_string = re.sub(r'//.*?$', '', fixed_string, flags=re.MULTILINE)
        fixed_string = re.sub(r'/\*.*?\*/', '', fixed_string, flags=re.DOTALL)
        
        return json.loads(fixed_string.strip())
    except:
        return default


def safe_parse_annotation_response(response: str) -> Dict[str, Any]:
    """
    Helper that parses an annotation response and guarantees the expected format.
    
    Args:
        response: raw response string from the model
    
    Returns:
        Parsed dict containing left_arm_action and right_arm_action.
    """
    parsed = safe_parse_json(response)
    
    if not isinstance(parsed, dict):
        logging.warning(f"Response is not a dict, using default format: {parsed}")
        return {
            "left_arm_action": "",
            "right_arm_action": ""
        }
    
    # Ensure the returned dict has the required fields
    result = {
        "left_arm_action": parsed.get("left_arm_action", ""),
        "right_arm_action": parsed.get("right_arm_action", "")
    }
    
    # Add other optional fields
    for key, value in parsed.items():
        if key not in result:
            result[key] = value
    
    return result



def convert_to_subtask_sequence(all_frames: list) -> list:
    """
    Convert SubtaskAnnotator's all_frames output into the new format.
    - Merge consecutive identical subtasks into a logical subtask sequence.
    - Map each frame to the logical subtask segment it belongs to.
    
    Args:
        all_frames: list of dicts, each containing 'frame_id', 'subtask', 
                    'left_arm_discription', 'right_arm_discription', etc.
        
    Returns:
        list of dicts, each frame containing:
            - frame_id
            - total_subtasks: subtask list after merging consecutive duplicates (e.g. ["grasp", "place", "grasp"])
            - current_subtask: the original (un-merged) subtask at this frame
            - current_subtask_index: 0-based index of the logical segment this frame belongs to
            - remaining_subtasks: subtasks from the current logical segment to the end
            - moving_instruction: combined left+right arm description
    """
    if not all_frames:
        return []

    # Step 1: Build the logical subtask sequence (merge consecutive duplicates)
    total_subtasks = []
    frame_to_logical_index = []  # logical-segment index per frame

    prev_subtask = None
    logical_idx = -1

    for frame in all_frames:
        sub = frame["subtask"]
        if sub != prev_subtask:
            total_subtasks.append(sub)
            logical_idx += 1
            prev_subtask = sub
        frame_to_logical_index.append(logical_idx)

    # Step 2: Build output
    result = []
    for i, frame in enumerate(all_frames):
        logical_idx = frame_to_logical_index[i]
        current_sub = frame["subtask"]

        # Combine left+right arm instructions
        left_desc = frame.get("left_arm_discription", "").strip()
        right_desc = frame.get("right_arm_discription", "").strip()
        instructions = [s for s in [left_desc, right_desc] if s]
        moving_instruction = " ".join(instructions) if instructions else "No action."

        new_frame = {
            "frame_id": frame["frame_id"],
            "total_subtasks": total_subtasks,  # shared logical sequence
            "current_subtask": current_sub,
            "current_subtask_index": logical_idx,
            "remaining_subtasks": total_subtasks[logical_idx:],
            "moving_instruction": moving_instruction
        }
        result.append(new_frame)

    return result


def save_annotations_to_hdf5(enhanced_annotations, scene_description, hdf5_path):
    """
    Save enhanced frame-level annotations and a global scene description to an HDF5 file.

    Args:
        enhanced_annotations (list[dict]): List of annotation dicts with keys:
            - "frame_id" (ignored)
            - "total_subtasks"
            - "current_subtask"
            - "current_subtask_index"
            - "remaining_subtasks"
            - "moving_instruction"
        scene_description (str): Global scene-level description.
        hdf5_path (str or Path): Output HDF5 file path.
    """
    if not enhanced_annotations:
        raise ValueError("Input annotation list is empty.")

    # Ensure output directory exists
    Path(hdf5_path).parent.mkdir(parents=True, exist_ok=True)

    # Extract shared total_subtasks (assume same across all frames)
    total_subtasks = enhanced_annotations[0]["total_subtasks"]

    # Prepare per-frame arrays
    current_subtasks = []
    current_subtask_indices = []
    remaining_subtasks_str = []
    moving_instructions = []

    for ann in enhanced_annotations:
        current_subtasks.append(ann["current_subtask"])
        current_subtask_indices.append(ann["current_subtask_index"])
        remaining_subtasks_str.append(json.dumps(ann["remaining_subtasks"], ensure_ascii=False))
        moving_instructions.append(ann["moving_instruction"])

    # Convert to numpy arrays
    current_subtasks = np.array(current_subtasks, dtype=object)
    current_subtask_indices = np.array(current_subtask_indices, dtype=np.int32)
    remaining_subtasks_str = np.array(remaining_subtasks_str, dtype=object)
    moving_instructions = np.array(moving_instructions, dtype=object)

    # Write to HDF5
    # with h5py.File(hdf5_path, "w") as f:
    #     # Global scalar: scene description
    #     f.create_dataset(
    #         "scene_description",
    #         data=np.array(scene_description, dtype=object),
    #         dtype=h5py.string_dtype(encoding='utf-8')
    #     )

    #     # Global array: total_subtasks (list of strings)
    #     f.create_dataset(
    #         "total_subtasks",
    #         data=np.array(total_subtasks, dtype=object),
    #         dtype=h5py.string_dtype(encoding='utf-8')
    #     )

    #     # Per-frame fields
    #     f.create_dataset(
    #         "current_subtask",
    #         data=current_subtasks,
    #         dtype=h5py.string_dtype(encoding='utf-8')
    #     )
    #     f.create_dataset(
    #         "current_subtask_index",
    #         data=current_subtask_indices,
    #         dtype=np.int32
    #     )
    #     f.create_dataset(
    #         "remaining_subtasks",
    #         data=remaining_subtasks_str,
    #         dtype=h5py.string_dtype(encoding='utf-8')
    #     )
    #     f.create_dataset(
    #         "moving_instruction",
    #         data=moving_instructions,
    #         dtype=h5py.string_dtype(encoding='utf-8')
    #     )

    # print(f"Saved HDF5 to: {hdf5_path}")