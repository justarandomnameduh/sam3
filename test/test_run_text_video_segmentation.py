import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_text_video_segmentation.py"
SPEC = importlib.util.spec_from_file_location("run_text_video_segmentation", SCRIPT_PATH)
run_text_video_segmentation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(run_text_video_segmentation)


class RunTextVideoSegmentationTest(unittest.TestCase):
    def test_normalizes_wsl_unc_path(self):
        path = run_text_video_segmentation.normalize_path(
            r"\\wsl.localhost\Ubuntu\home\nqmtien\phd\experiment\samples"
        )
        self.assertEqual(
            path,
            Path("/home/nqmtien/phd/experiment/samples"),
        )

    def test_discovers_frames_with_numeric_sort(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            frame_dir = Path(tmp_dir)
            for name in ("10.jpg", "2.jpg", "1.png"):
                Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(
                    frame_dir / name
                )

            paths = run_text_video_segmentation.discover_frames(frame_dir)

            self.assertEqual([path.name for path in paths], ["1.png", "2.jpg", "10.jpg"])

    def test_composes_instance_mask_with_stable_id_priority(self):
        outputs = {
            "out_obj_ids": np.array([1, 0], dtype=np.int64),
            "out_binary_masks": np.array(
                [
                    [
                        [1, 1, 0, 0],
                        [1, 1, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                    ],
                    [
                        [0, 0, 0, 0],
                        [0, 1, 1, 0],
                        [0, 1, 1, 0],
                        [0, 0, 0, 0],
                    ],
                ],
                dtype=bool,
            ),
        }

        mask = run_text_video_segmentation.compose_instance_mask(outputs, (4, 4))

        self.assertEqual(mask.dtype, np.uint16)
        self.assertEqual(mask[1, 1], 1)
        self.assertEqual(mask[0, 0], 2)
        self.assertEqual(mask[2, 2], 1)
        self.assertEqual(
            run_text_video_segmentation.sam_obj_id_from_mask_id(mask[1, 1]),
            0,
        )

    def test_maps_sam_object_ids_to_png_safe_mask_ids(self):
        self.assertEqual(run_text_video_segmentation.mask_id_from_sam_obj_id(0), 1)
        self.assertEqual(run_text_video_segmentation.mask_id_from_sam_obj_id(7), 8)

    def test_generates_overlay_video(self):
        try:
            import cv2  # noqa: F401
        except ImportError:
            self.skipTest("opencv-python is not installed")

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            frame_dir = tmp_path / "frames"
            mask_dir = tmp_path / "masks"
            frame_dir.mkdir()
            mask_dir.mkdir()
            for frame_idx in range(2):
                frame = np.full((16, 16, 3), 80, dtype=np.uint8)
                mask = np.zeros((16, 16), dtype=np.uint16)
                mask[4:12, 4:12] = frame_idx + 1
                Image.fromarray(frame).save(frame_dir / f"{frame_idx:05d}.jpg")
                Image.fromarray(mask).save(mask_dir / f"{frame_idx:05d}.png")

            output_path = run_text_video_segmentation.generate_overlay_video(
                run_text_video_segmentation.discover_frames(frame_dir),
                mask_dir,
                tmp_path / "overlay.mp4",
                query="gold fish",
                fps=5,
            )

            self.assertTrue(output_path.is_file())
            self.assertGreater(output_path.stat().st_size, 0)

    def test_dry_run_does_not_initialize_model(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            frame_dir = tmp_path / "frames"
            out_dir = tmp_path / "out"
            frame_dir.mkdir()
            Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(
                frame_dir / "00000.jpg"
            )
            cuda_info = {
                "torch": "test",
                "torch_cuda": "test",
                "cuda_available": False,
                "cuda_device_count": 0,
                "gpu_name": None,
            }

            with patch.object(
                run_text_video_segmentation,
                "collect_cuda_info",
                return_value=cuda_info,
            ):
                exit_code = run_text_video_segmentation.main(
                    [
                        "--in-path",
                        str(frame_dir),
                        "--out-path",
                        str(out_dir),
                        "--query",
                        "gold fish",
                        "--dry-run",
                        "--skip-overlay",
                        "--allow-no-cuda",
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertFalse(out_dir.exists())


if __name__ == "__main__":
    unittest.main()
