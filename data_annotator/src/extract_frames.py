import cv2
import os
import argparse
from concurrent.futures import ThreadPoolExecutor

def save_frame(img_path, frame):
    """Worker function used by the thread pool to save a single frame."""
    try:
        cv2.imwrite(img_path, frame)
    except Exception as e:
        print(f"Error saving {img_path}: {e}")

def extract_frames(video_path: str, output_dir: str, start_frame: int = 0, max_frames: int = -1):
    """
    Extract frames from a video using a thread pool.
    
    Args:
        video_path: input video path
        output_dir: output image directory
        start_frame: starting frame index
        max_frames: maximum number of frames to extract
        max_workers: thread-pool size (1-2x CPU cores recommended)
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video FPS: {fps:.2f}, Total frames: {total_frames}")

    # --- Optimization 1: fast frame skipping ---
    if start_frame > 0:
        print(f"Skipping to frame {start_frame}...")
        for _ in range(start_frame):
            cap.grab()  # grab() reads headers only and is much faster than read()
    
    frame_id = start_frame
    saved_count = 0
    
    # --- Optimization 2: thread pool ---
    # 'with' ensures all worker threads finish before the pool is closed
    with ThreadPoolExecutor(max_workers=8) as executor:
        print(f"Starting extraction with 8 threads...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if max_frames > 0 and saved_count >= max_frames:
                break
            
            img_name = f"frame_{frame_id:05d}.jpg"
            img_path = os.path.join(output_dir, img_name)
            
            # Submit disk-write task to the thread pool
            executor.submit(save_frame, img_path, frame)
            
            saved_count += 1
            frame_id += 1
            
            if saved_count % 100 == 0:
                print(f"Queued {saved_count} frames...")
    
    cap.release()
    print(f"Done. Successfully queued {saved_count} frames to {output_dir}")
    print("Wait for background threads to finish writing to disk...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract video frames as images (Multi-threaded)")
    parser.add_argument("video", help="Path to input video file")
    parser.add_argument("output_dir", help="Directory to save frames")
    parser.add_argument("--start", type=int, default=0, help="Start frame index (default: 0)")
    parser.add_argument("--max", type=int, default=-1, help="Max number of frames to extract (default: all)")
    
    args = parser.parse_args()
    
    extract_frames(
        video_path=args.video,
        output_dir=args.output_dir,
        start_frame=args.start,
        max_frames=args.max,
    )