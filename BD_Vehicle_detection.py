#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(
        description="Vehicle detection on a video using Ultralytics YOLO."
    )
    parser.add_argument(
        "source",
        type=str,
        help="Path to input video (e.g., test.mp4). For webcam use '0'."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="BD_Vehicle_YOLO11.pt",
        help="Path to YOLO model weights (.pt). Default: BD_Vehicle_YOLO11.pt"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold (0-1). Default: 0.25"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device, e.g., 'cpu', 'cuda', 'cuda:0'. Default: auto"
    )
    parser.add_argument(
        "--project",
        type=str,
        default="runs/detect",
        help="Project directory to save results. Default: runs/detect"
    )
    parser.add_argument(
        "--name",
        type=str,
        default="predict",
        help="Experiment name (subfolder in project). Default: predict"
    )
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="Allow existing project/name without incrementing."
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show annotated frames in a window during processing."
    )
    parser.add_argument(
        "--save-txt",
        action="store_true",
        help="Save detections as YOLO label text files."
    )
    parser.add_argument(
        "--save-crop",
        action="store_true",
        help="Save cropped detections as images."
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size. Default: 640"
    )
    return parser.parse_args()

def main():
    args = parse_args()

    # Defer imports so --help is fast and clean
    try:
        from ultralytics import YOLO
    except Exception as e:
        print("Failed to import ultralytics. Please install with: pip install ultralytics", file=sys.stderr)
        raise

    # Load model
    model = YOLO(args.model)

    # Prepare source (int for webcam, string path for file)
    source = args.source
    if source.isdigit():
        source = int(source)

    # Ensure project directory exists
    Path(args.project).mkdir(parents=True, exist_ok=True)

    # Run prediction on video with stream=True for memory efficiency
    # save=True writes an annotated video to project/name
    results = model.predict(
        source=source,
        conf=args.conf,
        device=args.device,
        save=True,
        show=args.show,
        save_txt=args.save_txt,
        save_crop=args.save_crop,
        imgsz=args.imgsz,
        project=args.project,
        name=args.name,
        exist_ok=args.exist_ok,
        stream=True,  # generator over frames
    )

    # Consume the generator to ensure processing completes when stream=True
    # This also gives access to per-frame info if needed.
    last_save_dir = None
    for r in results:
        # r.save_dir points to the output directory for this run
        last_save_dir = r.save_dir
        # If showing, frames are displayed by Ultralytics when show=True.
        # Annotated frames are written to the output video automatically when save=True.

        # Example: print basic speed stats per frame (optional)
        if hasattr(r, "speed"):
            # speed dict has 'preprocess', 'inference', 'postprocess' in ms
            spd = r.speed
            print(f"Speed ms/frame - pre: {spd.get('preprocess', 0):.1f}, "
                  f"inf: {spd.get('inference', 0):.1f}, "
                  f"post: {spd.get('postprocess', 0):.1f}")

    if last_save_dir:
        print(f"Results saved to: {last_save_dir}")
    else:
        print("Processing finished, but no save directory was reported. Check your inputs and arguments.")

if __name__ == "__main__":
    main()
