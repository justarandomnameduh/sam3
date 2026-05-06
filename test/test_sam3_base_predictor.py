import importlib.util
import sys
import types
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_PREDICTOR_PATH = REPO_ROOT / "sam3" / "model" / "sam3_base_predictor.py"

logger_module = types.ModuleType("sam3.logger")
logger_module.get_logger = lambda name: types.SimpleNamespace(info=lambda *args, **kwargs: None)
sys.modules.setdefault("sam3.logger", logger_module)

SPEC = importlib.util.spec_from_file_location("sam3_base_predictor", BASE_PREDICTOR_PATH)
sam3_base_predictor = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sam3_base_predictor)
Sam3BasePredictor = sam3_base_predictor.Sam3BasePredictor


class DummyModel:
    def __init__(self):
        self.last_kwargs = None

    def init_state(self, resource_path, offload_video_to_cpu=False):
        self.last_kwargs = {
            "resource_path": resource_path,
            "offload_video_to_cpu": offload_video_to_cpu,
        }
        return {"num_frames": 1}


class Sam3BasePredictorTest(unittest.TestCase):
    def test_start_session_filters_unsupported_init_state_kwargs(self):
        predictor = Sam3BasePredictor()
        predictor.model = DummyModel()

        response = predictor.start_session(
            "/tmp/frames",
            offload_video_to_cpu=True,
            offload_state_to_cpu=False,
        )

        self.assertIn("session_id", response)
        self.assertEqual(
            predictor.model.last_kwargs,
            {
                "resource_path": "/tmp/frames",
                "offload_video_to_cpu": True,
            },
        )


if __name__ == "__main__":
    unittest.main()
