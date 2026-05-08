# visualize_labels.py
import json
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

# Color palette for action categories (override as needed)
ACTION_COLORS = {
    "idle": "#f0f0f0",
    "grasp": "#ff6b6b",
    "release": "#4ecdc4",
    "move": "#45b7d1",
    # "approaching/leaving object": "#f9ca24",
    "other": "#a55eea",
}

# Canonical action order used in the legend (keep stable)
ACTION_ORDER = [
    "grasp",
    "release",
    # "carrying object",
    # "approaching/leaving object",
    "move",
    "other",
    "idle"
]

def load_label_data(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def extract_segments(data, arm_key):
    """Extract contiguous segments of identical (label, direction)."""
    if not data:
        return []
    segments = []
    current_label, current_dir = data[0][arm_key]
    start = 0
    for i in range(1, len(data)):
        label, direction = data[i][arm_key]
        if label != current_label or direction != current_dir:
            segments.append((start, i - 1, current_label, current_dir))
            current_label, current_dir = label, direction
            start = i
    segments.append((start, len(data) - 1, current_label, current_dir))
    return segments

def plot_arm(ax, segments, total_frames, arm_name, action_colors):
    ax.set_ylim(0, 1)
    ax.set_xlim(0, total_frames - 1)
    ax.set_ylabel(arm_name, fontsize=12, fontweight='bold')
    ax.set_yticks([])
    ax.tick_params(axis='x', labelsize=9)
    ax.grid(True, axis='x', linestyle='--', alpha=0.3)

    # Draw each segment
    for start, end, label, direction in segments:
        width = end - start + 1
        color = action_colors.get(label, "#cccccc")
        rect = patches.Rectangle(
            (start, 0), width, 1,
            linewidth=0.5,
            facecolor=color,
            edgecolor='black',
            alpha=0.85
        )
        ax.add_patch(rect)

        # Only render text when a direction is available
        if direction and direction.strip():
            mid = (start + end) / 2
            ax.text(
                mid, 0.5, direction,
                ha='center', va='center',
                fontsize=8.5,
                fontweight='bold',
                color='black',
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7)
            )

def create_legend(ax, action_colors, action_order):
    """Create a unified legend on the given axes."""
    legend_elements = []
    for action in action_order:
        if action in action_colors:
            label_display = action
            # Optional: prettify display name
            display_map = {
                "carrying object": "Carrying",
                "approaching/leaving object": "Approaching/Leaving",
                "idle": "Idle",
                "grasp": "Grasp",
                "release": "Release",
                "other": "Other"
            }
            legend_elements.append(
                patches.Patch(facecolor=action_colors[action], edgecolor='black', label=display_map.get(action, action))
            )
    ax.legend(handles=legend_elements, loc='center', ncol=len(legend_elements), fontsize=10)
    ax.axis('off')  # hide axes

def main():
    parser = argparse.ArgumentParser(description="Visualize arm action labels with legend and clean direction labels.")
    parser.add_argument("--input", required=True, help="Path to the labeled JSON file")
    parser.add_argument("--output", help="Path to save the visualization image (e.g., labels_viz.png)")
    parser.add_argument("--show", action="store_true", help="Show plot in window")
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"Input file not found: {args.input}")
        return

    data = load_label_data(args.input)
    total_frames = len(data)

    left_segments = extract_segments(data, "left_arm")
    right_segments = extract_segments(data, "right_arm")

    # Build figure: 3 rows -- two arm rows + one legend row
    fig = plt.figure(figsize=(18, 7))
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 0.2], hspace=0.3)
    
    ax_left = fig.add_subplot(gs[0])
    ax_right = fig.add_subplot(gs[1], sharex=ax_left)
    ax_legend = fig.add_subplot(gs[2])

    fig.suptitle("Arm Action Label Visualization", fontsize=16, fontweight='bold')

    plot_arm(ax_left, left_segments, total_frames, "Left Arm", ACTION_COLORS)
    plot_arm(ax_right, right_segments, total_frames, "Right Arm", ACTION_COLORS)
    create_legend(ax_legend, ACTION_COLORS, ACTION_ORDER)

    ax_right.set_xlabel("Frame Index", fontsize=12)

    # NOTE: avoid plt.tight_layout(); use figure-level adjustments instead
    # or rely entirely on gridspec + bbox_inches='tight' on save
    # Here we only reserve top space for the suptitle
    fig.subplots_adjust(top=0.92)  # leave room for suptitle

    # plt.tight_layout() intentionally not called

    # Save or show
    if args.output:
        plt.savefig(args.output, dpi=200, bbox_inches='tight')
        print(f"Visualization saved to: {args.output}")
    elif not args.show:
        default_out = os.path.splitext(args.input)[0] + "_viz.png"
        plt.savefig(default_out, dpi=200, bbox_inches='tight')
        print(f"Visualization saved to: {default_out}")

    if args.show:
        plt.show()

if __name__ == "__main__":
    main()