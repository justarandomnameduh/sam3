#!/usr/bin/env python3
"""Run SAM3 text-prompt video segmentation on a folder of frames.

This script is intended for an already allocated Bunya VSCode interactive GPU
session. It does not submit Slurm jobs or require a display.
"""

import argparse
import getpass
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


SAM3_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(SAM3_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(SAM3_REPO_ROOT))

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
PALETTE = (
    (230, 25, 75),
    (60, 180, 75),
    (0, 130, 200),
    (245, 130, 48),
    (145, 30, 180),
    (70, 240, 240),
    (240, 50, 230),
    (210, 245, 60),
    (250, 190, 190),
    (0, 128, 128),
    (230, 190, 255),
    (170, 110, 40),
    (255, 250, 200),
    (128, 0, 0),
    (170, 255, 195),
    (128, 128, 0),
    (255, 215, 180),
    (0, 0, 128),
)


def normalize_path(path_str):
    """Normalize local paths and common Windows WSL UNC paths."""
    raw = str(path_str).strip().strip("\"'")
    normalized = raw.replace("\\", "/")
    lowered = normalized.lower()
    for prefix in ("//wsl.localhost/", "//wsl$/"):
        if lowered.startswith(prefix):
            parts = normalized[len(prefix) :].split("/", 1)
            if len(parts) == 2:
                normalized = "/" + parts[1].lstrip("/")
            break
    return Path(normalized).expanduser()


def discover_frames(folder):
    folder = Path(folder)
    paths = [
        path for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    ]
    if not paths:
        raise FileNotFoundError(f"No image frames found in {folder}")
    try:
        paths.sort(key=lambda path: int(path.stem))
    except ValueError:
        paths.sort(key=lambda path: path.name)
    return paths


def _to_numpy(value):
    if value is None:
        return None
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _normalize_mask_array(mask):
    mask = np.asarray(mask)
    while mask.ndim > 2:
        mask = mask[0]
    return mask.astype(bool)


def _resize_mask(mask, frame_shape):
    height, width = frame_shape
    if mask.shape == (height, width):
        return mask
    image = Image.fromarray(mask.astype(np.uint8) * 255)
    nearest = getattr(getattr(Image, "Resampling", Image), "NEAREST")
    resized = image.resize((width, height), resample=nearest)
    return np.asarray(resized) > 0


def mask_id_from_sam_obj_id(obj_id):
    """Map a SAM object ID to a PNG-safe mask ID.

    PNG value 0 is reserved for background, while SAM3 may emit object ID 0.
    """
    mask_id = int(obj_id) + 1
    if mask_id <= 0 or mask_id > np.iinfo(np.uint16).max:
        raise ValueError("Object IDs must fit in uint16 PNG masks")
    return mask_id


def sam_obj_id_from_mask_id(mask_id):
    return int(mask_id) - 1


def compose_instance_mask(outputs, frame_shape):
    """Compose SAM3 per-object masks into one uint16 instance-ID mask.

    Mask pixel value 0 is background. Nonzero pixels store ``sam_obj_id + 1``.
    """
    height, width = frame_shape
    instance_mask = np.zeros((height, width), dtype=np.uint16)
    if not outputs:
        return instance_mask

    obj_ids = _to_numpy(outputs.get("out_obj_ids"))
    binary_masks = _to_numpy(outputs.get("out_binary_masks"))
    if obj_ids is None or binary_masks is None or len(obj_ids) == 0:
        return instance_mask

    obj_ids = [int(obj_id) for obj_id in np.asarray(obj_ids).reshape(-1)]
    if any(obj_id < 0 for obj_id in obj_ids):
        raise ValueError("Object IDs must be non-negative")

    binary_masks = np.asarray(binary_masks)
    order = sorted(range(len(obj_ids)), key=lambda idx: obj_ids[idx])
    for idx in order:
        obj_id = obj_ids[idx]
        mask_id = mask_id_from_sam_obj_id(obj_id)
        mask = _normalize_mask_array(binary_masks[idx])
        mask = _resize_mask(mask, (height, width))
        write_pixels = mask & (instance_mask == 0)
        instance_mask[write_pixels] = mask_id
    return instance_mask


def save_instance_mask(mask, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask).save(output_path)


def color_for_obj_id(obj_id):
    if obj_id < len(PALETTE):
        return PALETTE[obj_id]
    rng = np.random.default_rng(obj_id)
    return tuple(int(value) for value in rng.integers(48, 240, size=3))


def _load_font(size=18):
    for font_path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(font_path, size)
        except OSError:
            pass
    return ImageFont.load_default()


def render_overlay_frame(frame_rgb, instance_mask, query=None, alpha=0.45):
    overlay = frame_rgb.astype(np.float32).copy()
    mask_ids = [int(mask_id) for mask_id in np.unique(instance_mask) if mask_id != 0]
    for mask_id in mask_ids:
        sam_obj_id = sam_obj_id_from_mask_id(mask_id)
        color = np.asarray(color_for_obj_id(sam_obj_id), dtype=np.float32)
        mask = instance_mask == mask_id
        overlay[mask] = overlay[mask] * (1.0 - alpha) + color * alpha

    image = Image.fromarray(overlay.astype(np.uint8))
    draw = ImageDraw.Draw(image)
    font = _load_font()

    if query:
        text = str(query)
        padding_x = 8
        padding_y = 6
        try:
            bbox = draw.textbbox((0, 0), text, font=font)
            text_h = bbox[3] - bbox[1]
        except AttributeError:
            _, text_h = draw.textsize(text, font=font)
        banner_h = text_h + padding_y * 2
        draw.rectangle((0, 0, image.width, banner_h), fill=(0, 0, 0))
        draw.text((padding_x, padding_y), text, fill=(255, 255, 255), font=font)

    for mask_id in mask_ids:
        sam_obj_id = sam_obj_id_from_mask_id(mask_id)
        ys, xs = np.where(instance_mask == mask_id)
        if len(xs) == 0:
            continue
        cx = int(xs.mean())
        cy = int(ys.mean())
        color = color_for_obj_id(sam_obj_id)
        label = str(sam_obj_id)
        try:
            bbox = draw.textbbox((0, 0), label, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
        except AttributeError:
            text_w, text_h = draw.textsize(label, font=font)
        pad = 3
        draw.rectangle(
            (cx - text_w // 2 - pad, cy - text_h // 2 - pad,
             cx + text_w // 2 + pad, cy + text_h // 2 + pad),
            fill=color,
        )
        draw.text(
            (cx - text_w // 2, cy - text_h // 2),
            label,
            fill=(255, 255, 255),
            font=font,
        )
    return np.asarray(image)


def generate_overlay_video(frame_paths, mask_dir, output_path, query, fps=10, alpha=0.45):
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python is required to write overlay videos") from exc

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    try:
        for frame_path in frame_paths:
            frame_rgb = np.asarray(Image.open(frame_path).convert("RGB"))
            mask_path = Path(mask_dir) / f"{Path(frame_path).stem}.png"
            if not mask_path.exists():
                raise FileNotFoundError(f"Missing mask frame: {mask_path}")
            instance_mask = np.asarray(Image.open(mask_path))
            frame_rgb = render_overlay_frame(
                frame_rgb,
                instance_mask,
                query=query,
                alpha=alpha,
            )
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            if writer is None:
                height, width = frame_bgr.shape[:2]
                writer = cv2.VideoWriter(
                    str(output_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (width, height),
                )
                if not writer.isOpened():
                    raise RuntimeError(f"Could not open video writer for {output_path}")
            writer.write(frame_bgr)
    finally:
        if writer is not None:
            writer.release()
    return output_path


def validate_output_path(out_path, create=False):
    out_path = Path(out_path)
    if create:
        out_path.mkdir(parents=True, exist_ok=True)
        return

    parent = out_path
    while not parent.exists() and parent != parent.parent:
        parent = parent.parent
    if not parent.exists():
        raise FileNotFoundError(f"No existing parent directory for {out_path}")
    if not os.access(parent, os.W_OK):
        raise PermissionError(f"Output parent is not writable: {parent}")


def validate_cv2_available():
    try:
        import cv2  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "opencv-python is required for overlay video output; use --skip-overlay "
            "to write masks only"
        ) from exc


def collect_cuda_info(require_cuda=True):
    try:
        import torch
    except Exception as exc:
        raise RuntimeError(f"PyTorch import failed: {exc}") from exc

    cuda_available = torch.cuda.is_available()
    info = {
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": bool(cuda_available),
        "cuda_device_count": int(torch.cuda.device_count()) if cuda_available else 0,
        "gpu_name": None,
    }
    if cuda_available:
        info["gpu_name"] = torch.cuda.get_device_name(torch.cuda.current_device())
    elif require_cuda:
        raise RuntimeError(
            "CUDA is not active in this shell. In a Bunya VSCode interactive "
            "session, confirm the GPU allocation is active and CUDA/PyTorch are loaded."
        )
    return info


def print_preflight(args, frame_paths, cuda_info):
    print("[sam3-cli] Preflight")
    print(f"[sam3-cli] cwd: {Path.cwd()}")
    print(f"[sam3-cli] python: {sys.executable}")
    print(f"[sam3-cli] torch: {cuda_info.get('torch')}")
    print(f"[sam3-cli] torch_cuda: {cuda_info.get('torch_cuda')}")
    print(f"[sam3-cli] cuda_available: {cuda_info.get('cuda_available')}")
    print(f"[sam3-cli] gpu: {cuda_info.get('gpu_name')}")
    print(f"[sam3-cli] in_path: {args.in_path}")
    print(f"[sam3-cli] out_path: {args.out_path}")
    print(f"[sam3-cli] frames: {len(frame_paths)}")
    print(f"[sam3-cli] query: {args.query}")


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Run SAM3 text-prompt segmentation on a folder of video frames."
    )
    parser.add_argument("--in-path", required=True, help="Folder containing image frames")
    parser.add_argument("--out-path", required=True, help="Output folder")
    parser.add_argument("--query", required=True, help="Text prompt, e.g. 'gold fish'")
    parser.add_argument(
        "--version", default="sam3.1", choices=("sam3", "sam3.1"), help="SAM3 version"
    )
    parser.add_argument("--checkpoint", default=None, help="Optional local checkpoint")
    parser.add_argument("--prompt-frame", type=int, default=0)
    parser.add_argument(
        "--propagation-direction",
        default="forward",
        choices=("forward", "backward", "both"),
    )
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument("--output-prob-thresh", type=float, default=0.5)
    parser.add_argument("--max-num-objects", type=int, default=16)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--warm-up", action="store_true")
    parser.add_argument("--async-loading-frames", action="store_true")
    parser.add_argument("--skip-overlay", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--allow-no-cuda",
        action="store_true",
        help="Allow dry-run preflight on a CPU-only shell.",
    )
    return parser


def _prepare_args(args):
    args.in_path = normalize_path(args.in_path)
    args.out_path = normalize_path(args.out_path)
    if args.prompt_frame < 0:
        raise ValueError("--prompt-frame must be non-negative")
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("--alpha must be between 0 and 1")
    if not 0.0 <= args.output_prob_thresh <= 1.0:
        raise ValueError("--output-prob-thresh must be between 0 and 1")
    if args.max_num_objects <= 0:
        raise ValueError("--max-num-objects must be positive")
    if args.allow_no_cuda and not args.dry_run:
        raise ValueError("--allow-no-cuda is only valid with --dry-run")
    return args


def _configure_runtime_cache():
    username = getpass.getuser()
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", f"/tmp/torchinductor_cache_{username}")
    os.environ.setdefault("USE_PERFLIB", "1")


def _collect_outputs(model, session_id, args):
    frame_outputs = {}
    prompt_response = model.handle_request(
        {
            "type": "add_prompt",
            "session_id": session_id,
            "frame_index": args.prompt_frame,
            "text": args.query,
            "output_prob_thresh": args.output_prob_thresh,
        }
    )
    frame_outputs[int(prompt_response["frame_index"])] = prompt_response.get(
        "outputs", {}
    )

    request = {
        "type": "propagate_in_video",
        "session_id": session_id,
        "propagation_direction": args.propagation_direction,
        "start_frame_index": args.prompt_frame,
        "output_prob_thresh": args.output_prob_thresh,
    }
    for response in model.handle_stream_request(request):
        frame_idx = response.get("frame_index")
        if frame_idx is None:
            continue
        frame_outputs[int(frame_idx)] = response.get("outputs", {})
    return frame_outputs


def run(args):
    args = _prepare_args(args)
    if not args.in_path.is_dir():
        raise FileNotFoundError(f"Input frame folder does not exist: {args.in_path}")

    frame_paths = discover_frames(args.in_path)
    if args.prompt_frame >= len(frame_paths):
        raise ValueError(
            f"--prompt-frame {args.prompt_frame} is outside {len(frame_paths)} frames"
        )
    validate_output_path(args.out_path, create=not args.dry_run)
    if not args.skip_overlay:
        validate_cv2_available()
    cuda_info = collect_cuda_info(require_cuda=not args.allow_no_cuda)
    print_preflight(args, frame_paths, cuda_info)
    if args.dry_run:
        print("[sam3-cli] Dry run passed; model was not initialized.")
        return 0

    _configure_runtime_cache()
    masks_dir = args.out_path / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    for old_mask_path in masks_dir.glob("*.png"):
        old_mask_path.unlink()
    overlay_path = args.out_path / "overlay.mp4"
    metadata_path = args.out_path / "metadata.json"

    from sam3 import build_sam3_predictor

    build_kwargs = {
        "version": args.version,
        "compile": args.compile,
        "warm_up": args.warm_up,
        "async_loading_frames": args.async_loading_frames,
    }
    if args.checkpoint:
        build_kwargs["checkpoint_path"] = str(normalize_path(args.checkpoint))
    if args.version == "sam3.1":
        build_kwargs["max_num_objects"] = args.max_num_objects

    print(f"[sam3-cli] Building {args.version} predictor")
    try:
        model = build_sam3_predictor(**build_kwargs)
    except Exception as exc:
        if args.checkpoint:
            raise
        raise RuntimeError(
            "SAM3 model initialization failed. If this was a Hugging Face auth or "
            "network problem on Bunya, download the checkpoint manually and rerun "
            "with --checkpoint /path/to/sam3.1_multiplex.pt."
        ) from exc

    response = model.handle_request(
        {
            "type": "start_session",
            "resource_path": str(args.in_path),
            "offload_video_to_cpu": False,
            "offload_state_to_cpu": False,
        }
    )
    session_id = response["session_id"]
    started_at = time.time()
    try:
        frame_outputs = _collect_outputs(model, session_id, args)

        detected_sam_object_ids = set()
        mask_id_to_sam_obj_id = {}
        frames_with_detections = 0
        print(f"[sam3-cli] Writing masks to {masks_dir}")
        for frame_idx, frame_path in enumerate(frame_paths):
            frame_shape = np.asarray(Image.open(frame_path).convert("RGB")).shape[:2]
            mask = compose_instance_mask(frame_outputs.get(frame_idx, {}), frame_shape)
            nonzero_mask_ids = [
                int(mask_id) for mask_id in np.unique(mask) if mask_id != 0
            ]
            if nonzero_mask_ids:
                frames_with_detections += 1
                for mask_id in nonzero_mask_ids:
                    sam_obj_id = sam_obj_id_from_mask_id(mask_id)
                    detected_sam_object_ids.add(sam_obj_id)
                    mask_id_to_sam_obj_id[str(mask_id)] = sam_obj_id
            save_instance_mask(mask, masks_dir / f"{frame_path.stem}.png")

        if args.skip_overlay:
            overlay_output = None
        else:
            print(f"[sam3-cli] Writing overlay video to {overlay_path}")
            overlay_output = str(
                generate_overlay_video(
                    frame_paths,
                    masks_dir,
                    overlay_path,
                    query=args.query,
                    fps=args.fps,
                    alpha=args.alpha,
                )
            )

        metadata = {
            "input_path": str(args.in_path),
            "output_path": str(args.out_path),
            "query": args.query,
            "version": args.version,
            "checkpoint": str(normalize_path(args.checkpoint)) if args.checkpoint else None,
            "frame_count": len(frame_paths),
            "prompt_frame": args.prompt_frame,
            "propagation_direction": args.propagation_direction,
            "fps": args.fps,
            "alpha": args.alpha,
            "output_prob_thresh": args.output_prob_thresh,
            "mask_dir": str(masks_dir),
            "overlay_path": overlay_output,
            "detected_sam_object_ids": sorted(detected_sam_object_ids),
            "mask_id_to_sam_obj_id": mask_id_to_sam_obj_id,
            "frames_with_detections": frames_with_detections,
            "elapsed_sec": round(time.time() - started_at, 3),
            "cuda": cuda_info,
        }
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
            f.write("\n")

        print(f"[sam3-cli] Metadata: {metadata_path}")
        print(
            "[sam3-cli] Done: "
            f"{len(detected_sam_object_ids)} object ids across "
            f"{frames_with_detections}/{len(frame_paths)} frames"
        )
    finally:
        model.handle_request(
            {
                "type": "close_session",
                "session_id": session_id,
                "run_gc_collect": True,
            }
        )
    return 0


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        return run(args)
    except Exception as exc:
        print(f"[sam3-cli] ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
