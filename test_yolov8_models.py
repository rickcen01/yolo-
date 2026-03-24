import argparse
import os
import sys
from pathlib import Path

def _require_pkg(name: str) -> None:
    try:
        __import__(name)
    except ImportError as e:
        raise SystemExit(
            f"Missing dependency: {name}.\n"
            f"Install with: python -m pip install {name}\n"
        ) from e


def main() -> int:
    parser = argparse.ArgumentParser(description="Quick YOLOv8 weights test using Ultralytics")
    parser.add_argument(
        "--weights-dir",
        default=r"C:\\Users\\rick\\.cache\\yolov8",
        help="Folder containing yolov8m.pt/yolov8l.pt/yolov8x.pt",
    )
    parser.add_argument(
        "--image",
        default=None,
        help="Path to an image to run inference on (recommended).",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="cpu, 0, 0,1 ... (Ultralytics device string)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="IoU threshold",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size",
    )
    args = parser.parse_args()

    _require_pkg("ultralytics")

    from ultralytics import YOLO  # noqa: WPS433

    weights_dir = Path(args.weights_dir)
    if not weights_dir.exists():
        print(f"weights dir not found: {weights_dir}", file=sys.stderr)
        return 2

    weight_files = [
        weights_dir / "yolov8m.pt",
        weights_dir / "yolov8l.pt",
        weights_dir / "yolov8x.pt",
    ]
    missing = [str(p) for p in weight_files if not p.exists()]
    if missing:
        print("Missing weights:")
        for p in missing:
            print(f"- {p}")
        return 2

    if args.image is None:
        print(
            "No --image provided. Please pass an image path, e.g.\n"
            "  python scripts\\test_yolov8_models.py --image path\\to\\test.jpg\n"
        )
        return 2

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"image not found: {image_path}", file=sys.stderr)
        return 2

    out_dir = Path("yolov8_test_output")
    out_dir.mkdir(parents=True, exist_ok=True)

    for w in weight_files:
        print("=" * 80)
        print(f"Model: {w.name}")
        print(f"Loading weights from: {w}")

        model = YOLO(str(w))
        results = model.predict(
            source=str(image_path),
            device=args.device,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            save=True,
            project=str(out_dir),
            name=w.stem,
            exist_ok=True,
            verbose=False,
        )

        r0 = results[0]
        names = r0.names
        boxes = r0.boxes

        n = 0 if boxes is None else len(boxes)
        print(f"Detections: {n}")
        if boxes is not None and n > 0:
            # Print top 10 boxes
            for i in range(min(n, 10)):
                cls_id = int(boxes.cls[i].item())
                conf = float(boxes.conf[i].item())
                xyxy = [float(x) for x in boxes.xyxy[i].tolist()]
                print(f"#{i+1}: {names.get(cls_id, str(cls_id))} conf={conf:.3f} xyxy={xyxy}")

        print(f"Saved annotated image(s) under: {out_dir / w.stem}")

    print("=" * 80)
    print("Done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
