import asyncio
import os
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Awaitable, Callable


NotifyFn = Callable[[dict], Awaitable[None]]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sanitize_name(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in "-_" else "-" for char in value.lower())
    cleaned = cleaned.strip("-")
    return cleaned or "model"


def generate_mesh_local_tripo(image_path: str, job_id: str) -> str:
    """Runs local TripoSR via CLI."""
    output_dir = f"server/generated/objs/{job_id}"
    os.makedirs(output_dir, exist_ok=True)
    try:
        subprocess.run(["python", "-m", "tsr.cli", image_path, "--output-dir", output_dir], check=True)
        return os.path.join(output_dir, "0", "mesh.obj")
    except Exception as e:
        print(f"TripoSR Error: {e}")
        return ""


class ImageTo3DService:
    def __init__(self, generated_dir: Path, notify_fn: NotifyFn) -> None:
        self.generated_dir = generated_dir
        self.notify_fn = notify_fn
        self.jobs: dict[str, dict] = {}
        self._lock = asyncio.Lock()
        self.cli_available = True

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

        asyncio.create_task(self._process_job(job_id, prompt, image_bytes, filename_hint, original_name))
        return job

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

            obj_rel = str(final_obj).replace("\\", "/")
            await self._set_job(job_id, status="success", obj_path=obj_rel, message="obj ready")
            await self.notify_fn({
                "type": "image_job_complete",
                "job_id": job_id,
                "status": "success",
                "obj_path": obj_rel,
            })
        except Exception as e:
            await self._set_job(job_id, status="failed", message=str(e))
            await self.notify_fn({
                "type": "image_job_complete",
                "job_id": job_id,
                "status": "failed",
                "message": str(e),
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
            "provider": "local_triposr_cli",
            "created_at": now_iso(),
            "updated_at": now_iso(),
            "obj_path": "",
            "message": "prompt-only generation is not available in local TripoSR mode; provide an image",
        }
