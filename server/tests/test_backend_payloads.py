import base64
import tempfile
import unittest
import sys
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from image_to_3d_service import build_modular_fallback_obj
from main import decode_base64_payload, infer_voice_intent_fallback, validate_obj_modularity


class BackendPayloadHelpersTest(unittest.TestCase):
    def test_decode_base64_payload_accepts_data_uri(self) -> None:
        payload = "data:image/png;base64," + base64.b64encode(b"hello").decode("ascii")
        decoded = decode_base64_payload(payload)
        self.assertEqual(decoded, b"hello")

    def test_validate_obj_modularity_requires_multiple_tags(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            obj_path = Path(temp_dir) / "single.obj"
            obj_path.write_text("o part\nv 0 0 0\n", encoding="utf-8")
            ok, message = validate_obj_modularity(obj_path)
            self.assertFalse(ok)
            self.assertIn("separate child objects", message)

    def test_validate_obj_modularity_accepts_multiple_objects(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            obj_path = Path(temp_dir) / "multi.obj"
            obj_path.write_text("o body\nv 0 0 0\no wheel\nv 1 0 0\n", encoding="utf-8")
            ok, message = validate_obj_modularity(obj_path)
            self.assertTrue(ok)
            self.assertEqual(message, "ok")

    def test_build_modular_fallback_obj_creates_separate_parts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            obj_path = Path(temp_dir) / "fallback.obj"
            result = build_modular_fallback_obj("a keychain with a tag", "demo", obj_path)
            content = result.read_text(encoding="utf-8")
            self.assertIn("o ring", content)
            self.assertIn("o tag", content)
            self.assertIn("g ring", content)
            self.assertIn("g tag", content)

    def test_infer_voice_intent_fallback_turns_make_prompt_into_generation(self) -> None:
        intent = infer_voice_intent_fallback("Make a high polygon 3D model of an apple.")
        self.assertEqual(intent.get("intent"), "generate_blender")
        self.assertEqual(intent.get("prompt"), "high polygon 3d model of an apple")


if __name__ == "__main__":
    unittest.main()