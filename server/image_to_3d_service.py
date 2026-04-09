import asyncio
import os
import shutil
from urllib.parse import urlparse
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Awaitable, Callable

import requests


NotifyFn = Callable[[dict], Awaitable[None]]

TRIPOSG_SPACE_URL = os.getenv("TRIPOSG_SPACE_URL", os.getenv("IMAGE_3D_HF_SPACE_URL", "VAST-AI/TripoSG")).strip() or "VAST-AI/TripoSG"
TRIPOSG_HF_TOKEN = os.getenv("TRIPOSG_HF_TOKEN", os.getenv("IMAGE_3D_HF_TOKEN", os.getenv("HUGGINGFACE_API_KEY", ""))).strip()
TRIPOSG_TIMEOUT_SECONDS = int(os.getenv("TRIPOSG_TIMEOUT_SECONDS", os.getenv("IMAGE_3D_HF_SPACE_TIMEOUT_SECONDS", "180")))

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sanitize_name(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in "-_" else "-" for char in value.lower())
    cleaned = cleaned.strip("-")
    return cleaned or "model"


def _flatten_cloud_artifact_values(value):
    if value is None:
        return
    if isinstance(value, (list, tuple, set)):
        for item in value:
            yield from _flatten_cloud_artifact_values(item)
        return
    if isinstance(value, dict):
        for key in ("obj_path", "path", "output", "result", "file", "url", "download_url"):
            if key in value:
                yield from _flatten_cloud_artifact_values(value[key])
        return
    yield value


def _download_cloud_artifact(url: str, output_dir: Path, filename_hint: str) -> Path:
    parsed = urlparse(url)
    suffix = Path(parsed.path).suffix.lower() or ".obj"
    target_path = output_dir / f"{sanitize_name(filename_hint)}_{uuid.uuid4().hex[:10]}{suffix}"
    response = requests.get(url, timeout=TRIPOSG_TIMEOUT_SECONDS)
    response.raise_for_status()
    target_path.write_bytes(response.content)
    return target_path


def _convert_cloud_artifact_to_obj(source_path: Path, output_dir: Path, filename_hint: str) -> Path:
    if source_path.suffix.lower() == ".obj":
        target_path = output_dir / f"{sanitize_name(filename_hint)}_{uuid.uuid4().hex[:10]}.obj"
        shutil.copyfile(source_path, target_path)
        return target_path

    try:
        import trimesh  # type: ignore[import-not-found]
    except Exception as exc:
        raise RuntimeError(f"trimesh is required to convert cloud mesh output: {exc}") from exc

    mesh = trimesh.load(str(source_path), force="scene")
    target_path = output_dir / f"{sanitize_name(filename_hint)}_{uuid.uuid4().hex[:10]}.obj"
    mesh.export(str(target_path))
    return target_path


def _resolve_cloud_artifact(result, output_dir: Path, filename_hint: str) -> Path | None:
    for candidate in _flatten_cloud_artifact_values(result):
        if candidate is None:
            continue
        if isinstance(candidate, Path):
            if candidate.exists():
                return _convert_cloud_artifact_to_obj(candidate, output_dir, filename_hint)
            continue

        candidate_text = str(candidate).strip()
        if not candidate_text:
            continue

        if candidate_text.startswith(("http://", "https://")):
            downloaded = _download_cloud_artifact(candidate_text, output_dir, filename_hint)
            return _convert_cloud_artifact_to_obj(downloaded, output_dir, filename_hint)

        local_candidate = Path(candidate_text)
        if local_candidate.exists():
            return _convert_cloud_artifact_to_obj(local_candidate, output_dir, filename_hint)

        if candidate_text.lower().endswith((".obj", ".glb", ".gltf")):
            maybe_relative = Path.cwd() / candidate_text
            if maybe_relative.exists():
                return _convert_cloud_artifact_to_obj(maybe_relative, output_dir, filename_hint)

    return None


BACKUP_MODEL_SOURCE = Path(__file__).resolve().parent / "generated" / "objs" / "backup_model.glb"


def _materialize_backup_model(output_dir: Path, filename_hint: str) -> Path:
    if not BACKUP_MODEL_SOURCE.exists():
        raise RuntimeError(f"backup model not found: {BACKUP_MODEL_SOURCE}")
    return _convert_cloud_artifact_to_obj(BACKUP_MODEL_SOURCE, output_dir, filename_hint)


class TripoSGSpaceProvider:
    name = "triposg"

    def __init__(self, space_url: str, timeout_seconds: int, hf_token: str) -> None:
        self.space_url = space_url.strip()
        self.timeout_seconds = timeout_seconds
        self.hf_token = hf_token.strip()
        self.available = bool(self.space_url)

    def describe(self, output_root: Path) -> dict:
        return {
            "mode": self.name,
            "available": self.available,
            "configured": bool(self.space_url),
            "details": self.space_url,
            "hf_token_configured": bool(self.hf_token),
            "api_names": ["/start_session", "/run_segmentation", "/get_random_seed", "/image_to_3d"],
            "output_root": str(output_root).replace("\\", "/"),
        }

    def _get_client(self):
        try:
            from gradio_client import Client, handle_file  # type: ignore[import-not-found]
        except Exception as exc:
            raise RuntimeError(f"gradio_client is required for TripoSG image generation: {exc}") from exc

        client = Client(self.space_url, hf_token=self.hf_token or None)
        return client, handle_file

    def _start_session(self, client) -> None:
        try:
            client.predict(api_name="/start_session")
        except Exception:
            pass

    def _run_segmentation(self, client, handle_file, image_path: Path):
        try:
            return client.predict(image=handle_file(str(image_path)), api_name="/run_segmentation")
        except Exception as exc:
            raise RuntimeError(f"TripoSG segmentation failed: {exc}") from exc

    def _get_seed(self, client) -> float:
        try:
            seed_value = client.predict(randomize_seed=True, seed=0, api_name="/get_random_seed")
            return float(seed_value)
        except Exception:
            return 0.0

    def _image_to_3d(self, client, handle_file, segmented_image, seed: float):
        if isinstance(segmented_image, dict):
            image_input = segmented_image
        else:
            image_input = handle_file(str(segmented_image))

        return client.predict(
            image=image_input,
            seed=seed,
            num_inference_steps=50,
            guidance_scale=7,
            simplify=True,
            target_face_num=100000,
            api_name="/image_to_3d",
        )

    async def generate(self, image_path: Path, prompt: str, job_id: str, filename_hint: str, output_dir: Path) -> Path:
        client, handle_file = self._get_client()
        await asyncio.to_thread(self._start_session, client)
        segmented_image = await asyncio.to_thread(self._run_segmentation, client, handle_file, image_path)
        seed = await asyncio.to_thread(self._get_seed, client)
        result = await asyncio.to_thread(self._image_to_3d, client, handle_file, segmented_image, seed)
        artifact = _resolve_cloud_artifact(result, output_dir, filename_hint)
        if not artifact:
            raise RuntimeError("TripoSG did not return a usable GLB artifact")
        return artifact


def build_image_provider() -> object:
    return TripoSGSpaceProvider(
        space_url=TRIPOSG_SPACE_URL,
        timeout_seconds=TRIPOSG_TIMEOUT_SECONDS,
        hf_token=TRIPOSG_HF_TOKEN,
    )


class ImageTo3DService:
    def __init__(self, generated_dir: Path, notify_fn: NotifyFn) -> None:
        self.generated_dir = generated_dir
        self.notify_fn = notify_fn
        self.jobs: dict[str, dict] = {}
        self._lock = asyncio.Lock()
        self.provider = build_image_provider()
        self.provider_name = getattr(self.provider, "name", "triposg")
        self.provider_available = bool(getattr(self.provider, "available", False))

        self.image_dir = self.generated_dir / "image_jobs"
        self.obj_dir = self.generated_dir / "objs"
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.obj_dir.mkdir(parents=True, exist_ok=True)

    def describe_provider(self) -> dict:
        describe = getattr(self.provider, "describe", None)
        if callable(describe):
            return describe(self.obj_dir)
        return {
            "mode": self.provider_name,
            "available": self.provider_available,
            "configured": self.provider_available,
            "output_root": str(self.obj_dir).replace("\\", "/"),
        }

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
            base_name = Path(original_name or "image").stem.replace("_", " ").replace("-", " ").strip() or "object"
            prompt = f"Create a detailed 3D reconstruction of {base_name}."

        job_id = uuid.uuid4().hex[:10]
        created_at = now_iso()
        job = {
            "job_id": job_id,
            "status": "queued",
            "prompt": prompt,
            "filename": filename_hint,
            "provider": self.provider_name,
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
            "provider": self.provider_name,
        })

        if not self.provider_available:
            backup_obj = _materialize_backup_model(self.obj_dir, filename_hint)
            obj_rel = str(backup_obj).replace("\\", "/")
            message = "TripoSG unavailable; using bundled backup model"
            await self._set_job(job_id, status="success", obj_path=obj_rel, message=message, provider="backup")
            await self.notify_fn({
                "type": "generation_complete",
                "job_id": job_id,
                "obj_path": obj_rel,
                "message": message,
                "provider": "backup",
            })
            await self.notify_fn({
                "type": "image_job_complete",
                "job_id": job_id,
                "status": "success",
                "obj_path": obj_rel,
                "message": message,
                "provider": "backup",
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
                "provider": self.provider_name,
            })

            final_obj = await self.provider.generate(input_path, prompt, job_id, filename_hint, self.obj_dir)
            if not final_obj.exists() or final_obj.stat().st_size < 64:
                raise RuntimeError(f"mesh obj was not generated correctly: {final_obj}")

            obj_rel = str(final_obj).replace("\\", "/")
            await self._set_job(job_id, status="success", obj_path=obj_rel, message="obj ready", provider=self.provider_name)
            await self.notify_fn({
                "type": "generation_complete",
                "job_id": job_id,
                "obj_path": obj_rel,
                "message": "image-to-3d completed",
                "provider": self.provider_name,
            })
            await self.notify_fn({
                "type": "image_job_complete",
                "job_id": job_id,
                "status": "success",
                "obj_path": obj_rel,
                "provider": self.provider_name,
            })
        except Exception as e:
            try:
                backup_obj = _materialize_backup_model(self.obj_dir, filename_hint)
                obj_rel = str(backup_obj).replace("\\", "/")
                message = f"TripoSG failed; using bundled backup model: {e}"
                await self._set_job(job_id, status="success", obj_path=obj_rel, message=message, provider="backup")
                await self.notify_fn({
                    "type": "generation_complete",
                    "job_id": job_id,
                    "obj_path": obj_rel,
                    "message": message,
                    "provider": "backup",
                })
                await self.notify_fn({
                    "type": "image_job_complete",
                    "job_id": job_id,
                    "status": "success",
                    "obj_path": obj_rel,
                    "message": message,
                    "provider": "backup",
                })
                return
            except Exception as backup_error:
                await self._set_job(job_id, status="failed", message=str(backup_error), provider=self.provider_name)
            await self.notify_fn({
                "type": "generation_failed",
                "job_id": job_id,
                "message": str(backup_error) if 'backup_error' in locals() else str(e),
                "provider": self.provider_name,
            })
            await self.notify_fn({
                "type": "image_job_complete",
                "job_id": job_id,
                "status": "failed",
                "message": str(backup_error) if 'backup_error' in locals() else str(e),
                "provider": self.provider_name,
            })

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
            "provider": self.provider_name,
            "created_at": now_iso(),
            "updated_at": now_iso(),
            "obj_path": "",
            "message": "prompt-only generation is not available for this provider; provide an image",
        }
