# label.py
import json
import numpy as np
import argparse
import yaml
import os


def get_direction_from_delta(dx, dy, dz, eps=0.005, dir_th=0.1):
    """
    Decide motion direction from a displacement vector. An axis is only labelled if its share
    of the total displacement is >= dir_th.
    
    Args:
        dx, dy, dz: displacement components
        eps: total displacement below this is considered idle
        dir_th: per-axis share threshold over total displacement (0~1)
    
    Returns:
        str: e.g. "right_up", "left_forward", or "" for idle
    """
    total = np.sqrt(dx**2 + dy**2 + dz**2)
    if total < eps:
        return ""

    dirs = []
    # X axis
    if abs(dx) / total >= dir_th:
        if dx > 0:
            dirs.append("right")
        else:
            dirs.append("left")
    
    # Y axis
    if abs(dy) / total >= dir_th:
        if dy > 0:
            dirs.append("forward")
        else:
            dirs.append("backward")
    
    # Z axis
    if abs(dz) / total >= dir_th:
        if dz > 0:
            dirs.append("up")
        else:
            dirs.append("down")
    
    return "_".join(dirs)


def classify_move_action(gripper_val, gripper_closed_th=0.01, gripper_open_th=0.99):
    """Decide the motion action type from the gripper state."""
    if gripper_val < gripper_closed_th:
        return "carrying object"
    elif gripper_val > gripper_open_th:
        return "approaching/leaving object"
    else:
        return "other"


def find_action_segments_from_idle(idle_flags, min_idle_frames=3):
    """
    Given per-frame idle flags, find all idle segments of length >= min_idle_frames
    and return the action (non-idle) segments separated by them.
    """
    n = len(idle_flags)
    if n == 0:
        return []

    # Step 1: find all contiguous idle runs
    idle_segments = []
    i = 0
    while i < n:
        if idle_flags[i]:
            start = i
            while i < n and idle_flags[i]:
                i += 1
            end = i - 1
            idle_segments.append((start, end))
        else:
            i += 1

    # Step 2: keep only idle runs with length >= min_idle_frames
    long_idle_segments = [
        (s, e) for s, e in idle_segments if (e - s + 1) >= min_idle_frames
    ]

    # If no sufficiently long idle run exists
    if not long_idle_segments:
        if all(idle_flags):
            return []  # entirely idle
        else:
            return [(0, n - 1)]  # the whole sequence is one action segment

    # Step 3: use the long idle runs as separators and extract action segments
    action_segments = []

    # Before the first long idle run
    first_idle_start = long_idle_segments[0][0]
    if first_idle_start > 0:
        action_segments.append((0, first_idle_start - 1))

    # Between consecutive long idle runs
    for i in range(len(long_idle_segments) - 1):
        end_prev = long_idle_segments[i][1]
        start_next = long_idle_segments[i + 1][0]
        if start_next > end_prev + 1:
            action_segments.append((end_prev + 1, start_next - 1))

    # After the last long idle run
    last_idle_end = long_idle_segments[-1][1]
    if last_idle_end < n - 1:
        action_segments.append((last_idle_end + 1, n - 1))

    # Drop invalid (empty / inverted) segments
    action_segments = [(s, e) for s, e in action_segments if s <= e]
    return action_segments


def is_frame_idle(pos_curr, pos_prev, g_curr, g_prev, v_th=0.01, dg_th=0.01):
    """Decide whether the current frame is instantaneously idle relative to the previous one."""
    disp = np.array(pos_curr) - np.array(pos_prev)
    speed = np.linalg.norm(disp)
    dg = g_curr - g_prev
    return speed < v_th and abs(dg) < dg_th


def label_move_subsegments(arm_results, arm_gripper_data, start, end, frame_deltas,
                          cls_th, ops_th, eps, dir_th):
    """
    Within [start, end], label any unlabelled contiguous sub-range as a motion action.
    arm_results: mutable list where each entry is [action, direction]
    """
    t = start
    while t <= end:
        if arm_results[t][0] == "idle": 
            sub_start = t
            while t <= end and arm_results[t][0] == "idle":
                t += 1
            sub_end = t - 1

            # Choose the action type from the gripper state at the sub-range start
            g_start = arm_gripper_data[sub_start]
            # action = classify_move_action(g_start, cls_th, ops_th)
            action = "move"
            # Compute total displacement (from sub_start+1 to sub_end)
            total_disp = np.array([0.0, 0.0, 0.0])
            for i in range(sub_start + 1, sub_end + 1):
                total_disp += frame_deltas[i]["delta_pos"]

            # Direction is only computed for motion-type actions
            # if action in ["carrying object", "approaching/leaving object", "other"]:
            if action == "move":
                direction = get_direction_from_delta(
                    total_disp[0], total_disp[1], total_disp[2], eps, dir_th
                )
            else:
                direction = ""

            # Assign
            for i in range(sub_start, sub_end + 1):
                arm_results[i] = [action, direction]
        else:
            t += 1
get_sentence_dict_left = {
    ("idle", ""): "Keep the left arm idle.",
    ("grasp", ""): "Close the left gripper.",
    ("release", ""): "Open the left gripper.",
    ("move", "up"): "Move the left gripper to the upper.",
    ("move", "down"): "Move the left gripper to the lower.",
    ("move", "left"): "Move the left gripper to the left.",
    ("move", "right"): "Move the left gripper to the right.",
    ("move", "forward"): "Move the left gripper to the front.",
    ("move", "backward"): "Move the left gripper to the back.",
    ("move", "left_up"): "Move the left gripper to the upper-left.",
    ("move", "right_up"): "Move the left gripper to the upper-right.",
    ("move", "left_down"): "Move the left gripper to the lower-left.",
    ("move", "right_down"): "Move the left gripper to the lower-right.",
    ("move", "left_forward"): "Move the left gripper to the front-left.",
    ("move", "right_forward"): "Move the left gripper to the front-right.",
    ("move", "left_backward"): "Move the left gripper to the back-left.",
    ("move", "right_backward"): "Move the left gripper to the back-right.",
    ("move", "backward_up"): "Move the left gripper to the upper-back.",
    ("move", "backward_down"): "Move the left gripper to the lower-back.",
    ("move", "forward_up"): "Move the left gripper to the upper-front.",
    ("move", "forward_down"): "Move the left gripper to the lower-front.",
    ("move", "left_forward_up"): "Move the left gripper to the upper-left-front.",
    ("move", "right_forward_up"): "Move the left gripper to the upper-right-front.",
    ("move", "left_backward_up"): "Move the left gripper to the upper-left-back.",
    ("move", "right_backward_up"): "Move the left gripper to the upper-right-back.",
    ("move", "left_forward_down"): "Move the left gripper to the lower-left-front.",
    ("move", "right_forward_down"): "Move the left gripper to the lower-right-front.",
    ("move", "left_backward_down"): "Move the left gripper to the lower-left-back.",
    ("move", "right_backward_down"): "Move the left gripper to the lower-right-back."
}
get_sentence_dict_right = {
    ("idle", ""): "Keep the right arm idle.",
    ("grasp", ""): "Close the right gripper.",
    ("release", ""): "Open the right gripper.",
    ("move", "up"): "Move the right gripper to the upper.",
    ("move", "down"): "Move the right gripper to the lower.",
    ("move", "left"): "Move the right gripper to the left.",
    ("move", "right"): "Move the right gripper to the right.",
    ("move", "forward"): "Move the right gripper to the front.",
    ("move", "backward"): "Move the right gripper to the back.",
    ("move", "left_up"): "Move the right gripper to the upper-left.",
    ("move", "right_up"): "Move the right gripper to the upper-right.",
    ("move", "left_down"): "Move the right gripper to the lower-left.",
    ("move", "right_down"): "Move the right gripper to the lower-right.",
    ("move", "left_forward"): "Move the right gripper to the front-left.",
    ("move", "right_forward"): "Move the right gripper to the front-right.",
    ("move", "left_backward"): "Move the right gripper to the back-left.",
    ("move", "right_backward"): "Move the right gripper to the back-right.",
    ("move", "backward_up"): "Move the right gripper to the upper-back.",
    ("move", "backward_down"): "Move the right gripper to the lower-back.",
    ("move", "forward_up"): "Move the right gripper to the upper-front.",
    ("move", "forward_down"): "Move the right gripper to the lower-front.",
    ("move", "left_forward_up"): "Move the right gripper to the upper-left-front.",
    ("move", "right_forward_up"): "Move the right gripper to the upper-right-front.",
    ("move", "left_backward_up"): "Move the right gripper to the upper-left-back.",
    ("move", "right_backward_up"): "Move the right gripper to the upper-right-back.",
    ("move", "left_forward_down"): "Move the right gripper to the lower-left-front.",
    ("move", "right_forward_down"): "Move the right gripper to the lower-right-front.",
    ("move", "left_backward_down"): "Move the right gripper to the lower-left-back.",
    ("move", "right_backward_down"): "Move the right gripper to the lower-right-back."
}

def main():
    parser = argparse.ArgumentParser(description="Label arm actions by segmenting using long idle periods.")
    parser.add_argument("--config", required=True, help="Path to the YAML config file")
    parser.add_argument("--input_file", required=True, help="Path to the input file")
    parser.add_argument("--output_file", required=True, help="Path to the output file")
    args = parser.parse_args()

    # --- Load config ---
    try:
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Config file not found: {args.config}")
        return
    except yaml.YAMLError as e:
        print(f"Error parsing YAML config: {e}")
        return

    v_th = config.get('velocity_threshold', 0.01)
    dg_th = config.get('delta_gripper_threshold', 0.01)
    eps = config.get('direction_eps', 0.005)
    cls_th = config.get('closed_gripper_threshold', 0.01)
    ops_th = config.get('open_gripper_threshold', 0.99)
    min_idle_frames = config.get('min_idle_frames', 3)
    dir_th = config.get('direction_ratio_threshold', 0.1)  # added parameter

    # --- Load input data ---
    input_file = args.input_file
    output_file = args.output_file

    if not os.path.isfile(input_file):
        print(f"Input file does not exist: {input_file}")
        return

    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading input: {e}")
        return

    required_keys = ["left_gripper", "right_gripper", "left_endpose", "right_endpose"]
    if not all(k in data for k in required_keys):
        print("Missing required keys in input JSON.")
        return

    n_frames = len(data["left_gripper"])
    print(f"Total frames: {n_frames}")

    # --- Step 1: Compute per-frame instantaneous idle flags (for t >= 1) ---
    left_is_idle = [True] * n_frames   # frame 0 defaults to idle
    right_is_idle = [True] * n_frames

    for t in range(1, n_frames):
        # Left arm
        pos_curr_l = data["left_endpose"][t][:3]
        pos_prev_l = data["left_endpose"][t-1][:3]
        g_curr_l = data["left_gripper"][t]
        g_prev_l = data["left_gripper"][t-1]
        left_is_idle[t] = is_frame_idle(pos_curr_l, pos_prev_l, g_curr_l, g_prev_l, v_th, dg_th)

        # Right arm
        pos_curr_r = data["right_endpose"][t][:3]
        pos_prev_r = data["right_endpose"][t-1][:3]
        g_curr_r = data["right_gripper"][t]
        g_prev_r = data["right_gripper"][t-1]
        right_is_idle[t] = is_frame_idle(pos_curr_r, pos_prev_r, g_curr_r, g_prev_r, v_th, dg_th)

    # --- Step 2: Extract action segments using long idle periods ---
    left_action_segs = find_action_segments_from_idle(left_is_idle, min_idle_frames)
    right_action_segs = find_action_segments_from_idle(right_is_idle, min_idle_frames)

    # --- Step 3: Precompute per-frame deltas (for t >= 1) ---
    left_frame_deltas = []
    right_frame_deltas = []

    for t in range(n_frames):
        if t == 0:
            left_frame_deltas.append({"delta_pos": np.array([0.0, 0.0, 0.0]), "speed": 0.0, "dg": 0.0})
            right_frame_deltas.append({"delta_pos": np.array([0.0, 0.0, 0.0]), "speed": 0.0, "dg": 0.0})
        else:
            # Left
            pos_curr_l = np.array(data["left_endpose"][t][:3])
            pos_prev_l = np.array(data["left_endpose"][t-1][:3])
            delta_pos_l = pos_curr_l - pos_prev_l
            speed_l = np.linalg.norm(delta_pos_l)
            dg_l = data["left_gripper"][t] - data["left_gripper"][t-1]
            left_frame_deltas.append({"delta_pos": delta_pos_l, "speed": speed_l, "dg": dg_l})

            # Right
            pos_curr_r = np.array(data["right_endpose"][t][:3])
            pos_prev_r = np.array(data["right_endpose"][t-1][:3])
            delta_pos_r = pos_curr_r - pos_prev_r
            speed_r = np.linalg.norm(delta_pos_r)
            dg_r = data["right_gripper"][t] - data["right_gripper"][t-1]
            right_frame_deltas.append({"delta_pos": delta_pos_r, "speed": speed_r, "dg": dg_r})

    # --- Step 4: Initialize all frames as idle ---
    results = []
    for t in range(n_frames):
        results.append({
            "frame": str(t),
            "left_arm": ["idle", ""],
            "right_arm": ["idle", ""]
        })

    # --- Step 5: Label left arm action segments with fine-grained logic ---
    for start, end in left_action_segs:
        # Pass 1: mark grasp/release per frame (skip t=0)
        for t in range(max(1, start), end + 1):
            dg = left_frame_deltas[t]["dg"]
            if dg <= -dg_th:
                results[t]["left_arm"] = ["grasp", ""]
            elif dg >= dg_th:
                results[t]["left_arm"] = ["release", ""]

        # Pass 2: label remaining frames in [start, end] as move actions
        left_arm_results = [r["left_arm"] for r in results]  # this is a list of mutable lists
        label_move_subsegments(
            left_arm_results,
            data["left_gripper"],
            start, end,
            left_frame_deltas,
            cls_th, ops_th, eps, dir_th  
        )
        # Update results in-place (since lists are mutable)
        for i, (action, direction) in enumerate(left_arm_results):
            results[i]["left_arm"] = [action, direction]

    # --- Step 6: Label right arm action segments ---
    for start, end in right_action_segs:
        # Pass 1: mark grasp/release
        for t in range(max(1, start), end + 1):
            dg = right_frame_deltas[t]["dg"]
            if dg <= -dg_th:
                results[t]["right_arm"] = ["grasp", ""]
            elif dg >= dg_th:
                results[t]["right_arm"] = ["release", ""]

        # Pass 2: label move subsegments
        right_arm_results = [r["right_arm"] for r in results]
        label_move_subsegments(
            right_arm_results,
            data["right_gripper"],
            start, end,
            right_frame_deltas,
            cls_th, ops_th, eps, dir_th  
        )
        for i, (action, direction) in enumerate(right_arm_results):
            results[i]["right_arm"] = [action, direction]

    # --- Step 7: Convert to sentence descriptions ---
    for i in range(n_frames):
        left_key = tuple(results[i]["left_arm"])
        right_key = tuple(results[i]["right_arm"])
        results[i]["left_arm_discription"] = get_sentence_dict_left.get(left_key, "Unknown action.")
        results[i]["right_arm_discription"] = get_sentence_dict_right.get(right_key, "Unknown action.")
    # --- Step 8: Save output ---
    with open(output_file, 'w', encoding='utf-8') as f_out:
        json.dump(results, f_out, ensure_ascii=False, indent=2)

    print(f"Segment-based labels saved to: {output_file}")
    print(f"Left action segments: {len(left_action_segs)}")
    print(f"Right action segments: {len(right_action_segs)}")


if __name__ == "__main__":
    main()