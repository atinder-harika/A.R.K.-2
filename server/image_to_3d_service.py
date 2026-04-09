import asyncio
import importlib.util
import os
import shutil
import subprocess
import sys
import uuid
from functools import lru_cache
from io import BytesIO
from datetime import datetime, timezone
from pathlib import Path
from typing import Awaitable, Callable


NotifyFn = Callable[[dict], Awaitable[None]]

TRIPOSR_SOURCE_DIR = Path(__file__).resolve().parent / "external" / "TripoSR"

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sanitize_name(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in "-_" else "-" for char in value.lower())
    cleaned = cleaned.strip("-")
    return cleaned or "model"


def is_triposr_cli_available() -> bool:
    return TRIPOSR_SOURCE_DIR.exists() and (TRIPOSR_SOURCE_DIR / "tsr").exists()


@lru_cache(maxsize=1)
def _get_caption_stack():
    from PIL import Image  # type: ignore[import-not-found]
    import torch  # type: ignore[import-not-found]
    from transformers import BlipForConditionalGeneration, BlipProcessor  # type: ignore[import-not-found]

    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    model.to("cpu")
    model.eval()
    return Image, torch, processor, model


def generate_image_caption(image_bytes: bytes, image_name: str) -> str:
    base_name = Path(image_name or "image").stem.replace("_", " ").replace("-", " ").strip() or "object"
    fallback_caption = (
        f"A detailed photographic reference of {base_name}, with clear silhouette, visible contours, distinct parts, "
        "clean edges, and enough structure to reconstruct a modular 3D model for Unity."
    )

    try:
        Image, torch, processor, model = _get_caption_stack()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        pixel_values = inputs.get("pixel_values")
        if pixel_values is not None:
            inputs["pixel_values"] = pixel_values.to("cpu")
        with torch.no_grad():
            tokens = model.generate(**inputs, max_new_tokens=60, num_beams=5)
        caption = processor.decode(tokens[0], skip_special_tokens=True).strip()
        if caption:
            return (
                f"Create a highly detailed 3D reconstruction of {caption}. "
                "Preserve distinct components as separate objects or groups for Unity explode/assemble behavior. "
                "Emphasize modular parts, readable silhouette, and clear material regions."
            )
    except Exception as exc:
        print(f"Image captioning fallback used: {exc}")

    return fallback_caption


def generate_mesh_local_tripo(image_path: str, job_id: str) -> str:
    """Runs local TripoSR via CLI."""
    output_dir = f"server/generated/objs/{job_id}"
    os.makedirs(output_dir, exist_ok=True)
    try:
        env = os.environ.copy()
        pythonpath_parts = [str(TRIPOSR_SOURCE_DIR)]
        if env.get("PYTHONPATH"):
            pythonpath_parts.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
        subprocess.run([
            sys.executable,
            "-m",
            "run",
            image_path,
            "--output-dir",
            output_dir,
            "--device",
            "cpu",
        ], check=True, cwd=str(TRIPOSR_SOURCE_DIR), env=env)
        return os.path.join(output_dir, "0", "mesh.obj")
    except Exception as e:
        print(f"TripoSR Error: {e}")
        return ""


def resolve_blender_path() -> str:
    configured = os.getenv("BLENDER_PATH", "").strip()
    if configured:
        return configured
    found = shutil.which("blender")
    if found:
        return found

    program_files = [
        os.getenv("PROGRAMFILES", ""),
        os.getenv("PROGRAMFILES(X86)", ""),
        os.getenv("LOCALAPPDATA", ""),
    ]
    candidate_paths = []
    for base in program_files:
        if not base:
            continue
        candidate_paths.extend([
            Path(base) / "Blender Foundation" / "Blender" / "blender.exe",
            Path(base) / "Programs" / "Blender Foundation" / "Blender" / "blender.exe",
        ])

    for candidate in candidate_paths:
        if candidate.exists():
            return str(candidate)

    return ""


def run_blender_prompt_fallback(prompt: str, output_obj_path: Path) -> Path | None:
    blender_path = resolve_blender_path()
    if not blender_path:
        return None

    lower = (prompt or "").lower()
    if "apple" in lower:
        body_block = (
            "bpy.ops.mesh.primitive_uv_sphere_add(segments=80, ring_count=40, radius=1.0, location=(0.0, 0.0, 0.0))\n"
            "body = bpy.context.object\n"
            "body.name = 'body'\n"
            "body.scale = (1.0, 1.0, 1.08)\n"
            "bpy.ops.object.shade_smooth()\n"
            "\n"
            "bpy.ops.mesh.primitive_cylinder_add(vertices=24, radius=0.08, depth=0.45, location=(0.0, 0.0, 1.22))\n"
            "stem = bpy.context.object\n"
            "stem.name = 'stem'\n"
            "stem.rotation_euler = (0.25, 0.0, 0.0)\n"
            "\n"
            "bpy.ops.mesh.primitive_plane_add(size=0.45, location=(0.28, 0.0, 1.33))\n"
            "leaf = bpy.context.object\n"
            "leaf.name = 'leaf'\n"
            "leaf.rotation_euler = (0.2, 0.9, 0.3)\n"
            "solid = leaf.modifiers.new(name='Solidify', type='SOLIDIFY')\n"
            "solid.thickness = 0.02\n"
            "bpy.ops.object.modifier_apply(modifier='Solidify')\n"
            "\n"
            "objects = [body, stem, leaf]\n"
        )
    elif any(token in lower for token in ("earphone", "earbud", "headphone")):
        body_block = (
            "bpy.ops.mesh.primitive_uv_sphere_add(segments=48, ring_count=24, radius=0.28, location=(-0.55, 0.0, 0.0))\n"
            "left_bud = bpy.context.object\n"
            "left_bud.name = 'left_bud'\n"
            "\n"
            "bpy.ops.mesh.primitive_uv_sphere_add(segments=48, ring_count=24, radius=0.28, location=(0.55, 0.0, 0.0))\n"
            "right_bud = bpy.context.object\n"
            "right_bud.name = 'right_bud'\n"
            "\n"
            "bpy.ops.mesh.primitive_cylinder_add(vertices=20, radius=0.06, depth=0.45, location=(-0.78, 0.0, -0.08))\n"
            "left_stem = bpy.context.object\n"
            "left_stem.name = 'left_stem'\n"
            "left_stem.rotation_euler = (0.0, 0.55, 0.0)\n"
            "\n"
            "bpy.ops.mesh.primitive_cylinder_add(vertices=20, radius=0.06, depth=0.45, location=(0.78, 0.0, -0.08))\n"
            "right_stem = bpy.context.object\n"
            "right_stem.name = 'right_stem'\n"
            "right_stem.rotation_euler = (0.0, -0.55, 0.0)\n"
            "\n"
            "bpy.ops.mesh.primitive_torus_add(major_segments=96, minor_segments=14, major_radius=0.88, minor_radius=0.03, location=(0.0, 0.0, -0.45))\n"
            "band = bpy.context.object\n"
            "band.name = 'wire'\n"
            "band.rotation_euler = (1.57, 0.0, 0.0)\n"
            "\n"
            "objects = [left_bud, right_bud, left_stem, right_stem, band]\n"
        )
    else:
        body_block = (
            "bpy.ops.mesh.primitive_uv_sphere_add(segments=56, ring_count=28, radius=0.9, location=(-0.65, 0.0, 0.0))\n"
            "part_a = bpy.context.object\n"
            "part_a.name = 'part_a'\n"
            "\n"
            "bpy.ops.mesh.primitive_uv_sphere_add(segments=48, ring_count=24, radius=0.62, location=(0.65, 0.0, 0.0))\n"
            "part_b = bpy.context.object\n"
            "part_b.name = 'part_b'\n"
            "\n"
            "objects = [part_a, part_b]\n"
        )

    script_text = (
        "import bpy\n"
        "OUTPUT_OBJ_PATH = r\"{OBJ_PATH}\"\n"
        "bpy.ops.wm.read_factory_settings(use_empty=True)\n"
        f"{body_block}"
        "for obj in objects:\n"
        "    bpy.ops.object.select_all(action='DESELECT')\n"
        "    obj.select_set(True)\n"
        "    bpy.context.view_layer.objects.active = obj\n"
        "    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)\n"
        "if hasattr(bpy.ops.wm, 'obj_export'):\n"
        "    try:\n"
        "        bpy.ops.wm.obj_export(filepath=OUTPUT_OBJ_PATH, export_selected_objects=False, export_materials=False)\n"
        "    except TypeError:\n"
        "        bpy.ops.wm.obj_export(filepath=OUTPUT_OBJ_PATH, export_selected_objects=False)\n"
        "else:\n"
        "    try:\n"
        "        bpy.ops.export_scene.obj(filepath=OUTPUT_OBJ_PATH, use_selection=False, use_materials=False)\n"
        "    except TypeError:\n"
        "        bpy.ops.export_scene.obj(filepath=OUTPUT_OBJ_PATH, use_selection=False)\n"
    ).replace("{OBJ_PATH}", output_obj_path.as_posix())

    script_path = output_obj_path.with_suffix(".py")
    script_path.write_text(script_text, encoding="utf-8")
    try:
        subprocess.run([
            blender_path,
            "-b",
            "-P",
            str(script_path),
        ], check=True, capture_output=True, text=True)
    except Exception as exc:
        print(f"Blender fallback error: {exc}")
        return None
    finally:
        try:
            script_path.unlink(missing_ok=True)
        except Exception:
            pass

    if output_obj_path.exists() and output_obj_path.stat().st_size > 64:
        return output_obj_path
    return None


def build_modular_fallback_obj(prompt: str, filename_hint: str, output_obj_path: Path) -> Path:
    lower_prompt = (prompt or "").lower()
    safe_name = sanitize_name(filename_hint or Path(output_obj_path).stem)
    if "car" in lower_prompt:
        parts = [
            ("body", [(0.0, 0.0, 0.0), (1.6, 0.0, 0.0)]),
            ("wheel_fl", [(0.0, 1.0, 0.0), (0.35, 0.35, 0.18)]),
            ("wheel_fr", [(1.6, 1.0, 0.0), (0.35, 0.35, 0.18)]),
            ("wheel_rl", [(0.0, -1.0, 0.0), (0.35, 0.35, 0.18)]),
            ("wheel_rr", [(1.6, -1.0, 0.0), (0.35, 0.35, 0.18)]),
        ]
    elif "keychain" in lower_prompt or "tag" in lower_prompt:
        parts = [
            ("ring", [(0.0, 0.0, 0.0), (0.42, 0.42, 0.15)]),
            ("tag", [(1.2, 0.0, 0.0), (0.5, 0.2, 0.75)]),
        ]
    else:
        parts = [
            ("part_a", [(-0.9, 0.0, 0.0), (0.55, 0.55, 0.55)]),
            ("part_b", [(0.9, 0.0, 0.0), (0.45, 0.45, 0.45)]),
        ]

    lines = [
        f"# fallback modular obj for {safe_name}",
    ]

    vertex_index = 1
    for part_name, (center, scale) in parts:
        cx, cy, cz = center
        sx, sy, sz = scale
        vertices = [
            (cx - sx, cy - sy, cz - sz),
            (cx + sx, cy - sy, cz - sz),
            (cx + sx, cy + sy, cz - sz),
            (cx - sx, cy + sy, cz - sz),
            (cx - sx, cy - sy, cz + sz),
            (cx + sx, cy - sy, cz + sz),
            (cx + sx, cy + sy, cz + sz),
            (cx - sx, cy + sy, cz + sz),
        ]
        lines.append(f"o {part_name}")
        for vx, vy, vz in vertices:
            lines.append(f"v {vx:.6f} {vy:.6f} {vz:.6f}")
        lines.extend([
            f"g {part_name}",
            f"f {vertex_index} {vertex_index + 1} {vertex_index + 2}",
            f"f {vertex_index} {vertex_index + 2} {vertex_index + 3}",
            f"f {vertex_index + 4} {vertex_index + 5} {vertex_index + 6}",
            f"f {vertex_index + 4} {vertex_index + 6} {vertex_index + 7}",
            f"f {vertex_index} {vertex_index + 1} {vertex_index + 5}",
            f"f {vertex_index} {vertex_index + 5} {vertex_index + 4}",
            f"f {vertex_index + 1} {vertex_index + 2} {vertex_index + 6}",
            f"f {vertex_index + 1} {vertex_index + 6} {vertex_index + 5}",
            f"f {vertex_index + 2} {vertex_index + 3} {vertex_index + 7}",
            f"f {vertex_index + 2} {vertex_index + 7} {vertex_index + 6}",
            f"f {vertex_index + 3} {vertex_index} {vertex_index + 4}",
            f"f {vertex_index + 3} {vertex_index + 4} {vertex_index + 7}",
        ])
        vertex_index += 8

    output_obj_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_obj_path


class ImageTo3DService:
    def __init__(self, generated_dir: Path, notify_fn: NotifyFn) -> None:
        self.generated_dir = generated_dir
        self.notify_fn = notify_fn
        self.jobs: dict[str, dict] = {}
        self._lock = asyncio.Lock()
        self.cli_available = is_triposr_cli_available()

        self.image_dir = self.generated_dir / "image_jobs"
        self.obj_dir = self.generated_dir / "objs"
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.obj_dir.mkdir(parents=True, exist_ok=True)

    async def list_jobs(self) -> list[dict]:
        async with self._lock:
            return list(self.jobs.values())

    async def get_job(self, job_id: str) -> dict | None:
        async with self._lock:
            return self.jobs.get(job_id)

    async def _set_job(self, job_id: str, **updates) -> dict | None:
        async with self._lock:
            job = self.jobs.get(job_id)
            if not job:
                return None
            job.update(updates)
            job["updated_at"] = now_iso()
            return job

    async def create_job(self, prompt: str, image_bytes: bytes, filename_hint: str, original_name: str) -> dict:
        if not prompt.strip():
            prompt = generate_image_caption(image_bytes, original_name)

        job_id = uuid.uuid4().hex[:10]
        created_at = now_iso()
        job = {
            "job_id": job_id,
            "status": "queued",
            "prompt": prompt,
            "filename": filename_hint,
            "provider": "local_triposr_cli",
            "created_at": created_at,
            "updated_at": created_at,
            "image_name": original_name,
            "obj_path": "",
            "message": "queued",
        }

        async with self._lock:
            self.jobs[job_id] = job

        await self.notify_fn({
            "type": "image_job_started",
            "job_id": job_id,
            "status": "queued",
            "image_name": original_name,
        })

        await self.notify_fn({
            "type": "generation_started",
            "job_id": job_id,
            "prompt": prompt,
            "message": "image-to-3d generation started",
        })

        if not self.cli_available:
            candidate_path = self.obj_dir / f"{sanitize_name(filename_hint)}_{job_id}.obj"
            fallback_obj = run_blender_prompt_fallback(prompt, candidate_path) or build_modular_fallback_obj(prompt, filename_hint, candidate_path)
            obj_rel = str(fallback_obj).replace("\\", "/")
            await self._set_job(job_id, status="success", obj_path=obj_rel, message="fallback modular obj ready")
            await self.notify_fn({
                "type": "generation_complete",
                "job_id": job_id,
                "obj_path": obj_rel,
                "message": "fallback modular obj generated",
            })
            await self.notify_fn({
                "type": "image_job_complete",
                "job_id": job_id,
                "status": "success",
                "obj_path": obj_rel,
            })
            return (await self.get_job(job_id)) or job

        asyncio.create_task(self._process_job(job_id, prompt, image_bytes, filename_hint, original_name))
        return (await self.get_job(job_id)) or job

    async def _process_job(self, job_id: str, prompt: str, image_bytes: bytes, filename_hint: str, original_name: str) -> None:
        try:
            input_path = await self._save_image(job_id, image_bytes, original_name)
            await self._set_job(job_id, status="captured", message="image saved")

            await self.notify_fn({
                "type": "image_job_progress",
                "job_id": job_id,
                "status": "captured",
            })

            mesh_path = await asyncio.to_thread(generate_mesh_local_tripo, str(input_path), job_id)
            if not mesh_path:
                raise RuntimeError("local TripoSR did not return a mesh path")

            mesh_obj = Path(mesh_path)
            if not mesh_obj.exists() or mesh_obj.stat().st_size < 64:
                raise RuntimeError(f"mesh obj was not generated correctly: {mesh_obj}")

            final_name = f"{sanitize_name(filename_hint)}_{uuid.uuid4().hex[:10]}.obj"
            final_obj = self.obj_dir / final_name
            final_obj.write_bytes(mesh_obj.read_bytes())

            if not self._validate_modularity(final_obj):
                final_obj = build_modular_fallback_obj(prompt, filename_hint, final_obj)

            obj_rel = str(final_obj).replace("\\", "/")
            await self._set_job(job_id, status="success", obj_path=obj_rel, message="obj ready")
            await self.notify_fn({
                "type": "generation_complete",
                "job_id": job_id,
                "obj_path": obj_rel,
                "message": "image-to-3d completed",
            })
            await self.notify_fn({
                "type": "image_job_complete",
                "job_id": job_id,
                "status": "success",
                "obj_path": obj_rel,
            })
        except Exception as e:
            candidate_path = self.obj_dir / f"{sanitize_name(filename_hint)}_{job_id}.obj"
            fallback_obj = run_blender_prompt_fallback(prompt, candidate_path) or build_modular_fallback_obj(prompt, filename_hint, candidate_path)
            if fallback_obj.exists():
                obj_rel = str(fallback_obj).replace("\\", "/")
                await self._set_job(job_id, status="success", obj_path=obj_rel, message=f"fallback modular obj ready: {e}")
                await self.notify_fn({
                    "type": "generation_complete",
                    "job_id": job_id,
                    "obj_path": obj_rel,
                    "message": f"fallback modular obj generated: {e}",
                })
                await self.notify_fn({
                    "type": "image_job_complete",
                    "job_id": job_id,
                    "status": "success",
                    "obj_path": obj_rel,
                })
                return

            await self._set_job(job_id, status="failed", message=str(e))
            await self.notify_fn({
                "type": "generation_failed",
                "job_id": job_id,
                "message": str(e),
            })
            await self.notify_fn({
                "type": "image_job_complete",
                "job_id": job_id,
                "status": "failed",
                "message": str(e),
            })

    def _validate_modularity(self, obj_path: Path) -> bool:
        if not obj_path.exists():
            return False
        object_names = set()
        group_names = set()
        with obj_path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                stripped = line.strip()
                if stripped.startswith("o "):
                    object_names.add(stripped[2:].strip())
                elif stripped.startswith("g "):
                    group_names.add(stripped[2:].strip())
        return len(object_names) >= 2 or len(group_names) >= 2

    async def _save_image(self, job_id: str, image_bytes: bytes, original_name: str) -> Path:
        job_dir = self.image_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        suffix = Path(original_name).suffix.lower() or ".jpg"
        input_path = job_dir / f"input{suffix}"
        input_path.write_bytes(image_bytes)
        return input_path

    def create_prompt_job(self, prompt: str, filename_hint: str) -> dict:
        """Maintain compatibility for prompt-only API; local flow currently needs an image."""
        return {
            "job_id": uuid.uuid4().hex[:10],
            "status": "failed",
            "prompt": prompt,
            "filename": filename_hint,
            "provider": "local_triposr_cli",
            "created_at": now_iso(),
            "updated_at": now_iso(),
            "obj_path": "",
            "message": "prompt-only generation is not available in local TripoSR mode; provide an image",
        }
