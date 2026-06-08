import argparse
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser(description="Export YOLO pose model to CoreML")
    ap.add_argument("--model", default="checkpoints/yolo11m-pose.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out", default="yolo11m-pose.mlpackage")
    args = ap.parse_args()

    # Create parent directory for the model if it doesn't exist
    model_dir = os.path.dirname(args.model)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)

    # Automatically downloads to args.model if not present
    model = YOLO(args.model)

    # Export directly to the path specified by --out
    model.export(format="coreml", imgsz=args.imgsz, half=True, name=args.out)


if __name__ == "__main__":
    main()
