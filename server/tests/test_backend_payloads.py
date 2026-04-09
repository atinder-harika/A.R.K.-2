import base64
import asyncio
import tempfile
import unittest
import sys
import json
from unittest.mock import patch
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from main import (
    decode_base64_payload,
    describe_language_model_runtime,
    infer_voice_intent_fallback,
    is_low_detail_blender_script,
    normalize_voice_edit_target,
    query_voice_intent,
    validate_obj_modularity,
)


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

    def test_describe_language_model_runtime_reports_gemini_only(self) -> None:
        runtime = describe_language_model_runtime()
        self.assertEqual(runtime.get("active_provider"), "gemini")
        self.assertIn("gemini_model", runtime)
        self.assertIn("gemini_configured", runtime)

    def test_infer_voice_intent_fallback_turns_make_prompt_into_generation(self) -> None:
        intent = infer_voice_intent_fallback("Make a high polygon 3D model of an apple.")
        self.assertEqual(intent.get("intent"), "generate_blender")
        self.assertEqual(intent.get("prompt"), "high polygon 3d model of an apple")

    def test_infer_voice_intent_fallback_turns_make_it_metallic_into_edit(self) -> None:
        intent = infer_voice_intent_fallback("Make it metallic.")
        self.assertEqual(intent.get("intent"), "edit_unity")
        self.assertEqual(intent.get("mode"), "material")
        self.assertEqual(intent.get("target"), "body")
        self.assertEqual(intent.get("color"), "#8a8a8a")

    def test_infer_voice_intent_fallback_keeps_metallic_generation_phrase_as_generation(self) -> None:
        intent = infer_voice_intent_fallback("Create a metallic robot model with glowing eyes.")
        self.assertEqual(intent.get("intent"), "generate_blender")
        self.assertEqual(intent.get("prompt"), "metallic robot model with glowing eyes")

    def test_infer_voice_intent_fallback_turns_asset_replace_into_edit(self) -> None:
        intent = infer_voice_intent_fallback("Replace the apple logo with a dragon sticker.")
        self.assertEqual(intent.get("intent"), "edit_unity")
        self.assertEqual(intent.get("mode"), "asset_swap")
        self.assertEqual(intent.get("target"), "logo")
        self.assertEqual(intent.get("asset_name"), "dragon")

    def test_normalize_voice_edit_target_reduces_sentence_to_core_noun(self) -> None:
        target = normalize_voice_edit_target("texture of the stem to a metallic color")
        self.assertEqual(target, "stem")

    def test_normalize_voice_edit_target_handles_multiword_mesh_name(self) -> None:
        target = normalize_voice_edit_target("the left earbud")
        self.assertEqual(target, "earbud")

    def test_query_voice_intent_sanitizes_llm_edit_target(self) -> None:
        async def run_test() -> None:
            bad_payload = json.dumps({
                "intent": "edit_unity",
                "target": "texture of the stem to a",
                "color": "metallic",
            })
            with patch("main.query_gemini", return_value=bad_payload):
                result = await query_voice_intent("Change the texture of the stem to a metallic color.")
            self.assertEqual(result.get("intent"), "edit_unity")
            self.assertEqual(result.get("target"), "stem")

        asyncio.run(run_test())

    def test_is_low_detail_blender_script_flags_placeholder_geometry(self) -> None:
        script = """import bpy
bpy.ops.mesh.primitive_cube_add(location=(0, 0, 0))
bpy.ops.mesh.primitive_cube_add(location=(1, 0, 0))
"""
        low_detail, reason = is_low_detail_blender_script(script)
        self.assertTrue(low_detail)
        self.assertIn("too simple", reason)


if __name__ == "__main__":
    unittest.main()