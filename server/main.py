"""
A.R.K. - Augmented Reality Kinetic Interface
FastAPI Backend Server
"""

import json
import os
import re
import shutil
import uuid
import asyncio
import sys
import requests
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi import HTTPException, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from voice_service import VoiceCommandService
from image_to_3d_service import ImageTo3DService

ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)

LOCAL_CODER_URL = os.getenv("LOCAL_CODER_URL", "http://localhost:11434/api/generate")
LOCAL_CODER_MODEL = os.getenv("LOCAL_CODER_MODEL", "qwen2.5-coder")

ARK_COMMAND_SYSTEM_PROMPT = (
    "You are A.R.K., an Augmented Reality Kinetic Interface assistant. "
    "You help users manipulate 3D models through natural language. "
    "When a user gives a command, respond with a short JSON object containing: "
    '"action" (one of: rotate, scale, move, texture, color, explode, reset, info), '
    '"params" (a dict of parameters like axis, value, color hex, etc.), '
    'and "description" (a short human-readable summary of what you did). '
    "Keep responses concise. If the command is conversational, set action to info."
)

BLENDER_SCRIPT_SYSTEM_PROMPT = (
    "You are an expert Python Blender Developer. "
    "Write a script to generate the requested 3D model. "
    "CRITICAL: Do NOT join meshes (bpy.ops.object.join()). "
    "Keep distinct parts as separate objects in the hierarchy. "
    "Ensure the script exports an OBJ."
)

app = FastAPI(title="A.R.K. Server", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Track connected Unity clients
connected_clients: list[WebSocket] = []

GENERATED_DIR = Path(__file__).resolve().parent / "generated"
GENERATED_SCRIPTS_DIR = GENERATED_DIR / "scripts"
GENERATED_OBJS_DIR = GENERATED_DIR / "objs"
MAX_PROMPT_LEN = 300
MAX_JOB_HISTORY = 20
BLENDER_TIMEOUT_SECONDS = 90
GENERATE_COOLDOWN_SECONDS = 4
PROMPT_DENYLIST = [
    "malware",
    "ransomware",
    "exploit",
    "weapon",
    "bomb",
]

generation_jobs = deque(maxlen=MAX_JOB_HISTORY)
last_generation_status = {
    "status": "idle",
    "job_id": "",
    "message": "no generation has run yet",
}

blender_path = ""
generation_last_seen: dict[str, float] = {}
voice_service: VoiceCommandService | None = None
image_to_3d_service: ImageTo3DService | None = None


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_generated_dirs() -> None:
    GENERATED_SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    GENERATED_OBJS_DIR.mkdir(parents=True, exist_ok=True)


def resolve_blender_path() -> str:
    configured_path = os.getenv("BLENDER_PATH", "").strip()
    if configured_path:
        return configured_path
    discovered_path = shutil.which("blender")
    if discovered_path:
        return discovered_path

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


def query_local_coder(prompt: str) -> str:
    """Queries the local Ollama instance running Qwen2.5-Coder."""
    payload = {"model": LOCAL_CODER_MODEL, "prompt": prompt, "stream": False}
    try:
        response = requests.post(LOCAL_CODER_URL, json=payload, timeout=60)
        response.raise_for_status()
        return response.json().get("response", "")
    except Exception as e:
        print(f"Local LLM Error: {e}")
        return "{}"


def to_relative_artifact_path(path: Path) -> str:
    try:
        return str(path.relative_to(Path(__file__).resolve().parent)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def sanitize_filename(value: str) -> str:
    base = re.sub(r"[^a-zA-Z0-9_-]+", "-", value.strip().lower())
    base = re.sub(r"-{2,}", "-", base).strip("-")
    return base or "model"


def strip_code_fence(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else ""
        cleaned = cleaned.rsplit("```", 1)[0].strip()
    return cleaned


def validate_blender_script(script_text: str) -> tuple[bool, str]:
    lower_script = script_text.lower()
    blocked_tokens = [
        "import os",
        "import sys",
        "import socket",
        "import requests",
        "import subprocess",
        "from os",
        "from sys",
        "subprocess",
        "eval(",
        "exec(",
        "__import__(",
        "open(",
        "bpy.ops.script",
        "bpy.ops.wm.open_mainfile",
    ]

    if not script_text.strip():
        return False, "generated script was empty"

    if "import bpy" not in lower_script and "bpy." not in lower_script:
        return False, "script must use bpy operations"

    has_legacy_export = "export_scene.obj" in lower_script
    has_modern_export = "wm.obj_export" in lower_script
    if not has_legacy_export and not has_modern_export:
        return False, "script must export an obj using bpy.ops.export_scene.obj or bpy.ops.wm.obj_export"

    for token in blocked_tokens:
        if token in lower_script:
            return False, f"script contains blocked token: {token}"

    return True, ""


def normalize_obj_export(script_text: str) -> str:
    export_block = (
        "\n"
        "# export obj across blender versions\n"
        "output_filepath = OUTPUT_OBJ_PATH\n"
        "if hasattr(bpy.ops.wm, 'obj_export'):\n"
        "    bpy.ops.wm.obj_export(filepath=output_filepath, export_selected_objects=False)\n"
        "elif hasattr(bpy.ops.export_scene, 'obj'):\n"
        "    bpy.ops.export_scene.obj(filepath=output_filepath, use_selection=False)\n"
        "else:\n"
        "    raise RuntimeError('obj export operator not found in this blender build')\n"
    )

    # remove any existing export calls so we control compatibility in one place
    script_lines = script_text.splitlines()
    filtered_lines = []
    for line in script_lines:
        lower = line.lower()
        if "bpy.ops.export_scene.obj" in lower or "bpy.ops.wm.obj_export" in lower:
            continue
        filtered_lines.append(line)

    base = "\n".join(filtered_lines).strip()
    header = (
        "import bpy\n"
        "import math\n"
        "OUTPUT_OBJ_PATH = r\"{OBJ_PATH}\"\n"
    )

    # avoid duplicate import headers if model already produced them
    cleaned = base
    cleaned = cleaned.replace("import bpy\n", "")
    cleaned = cleaned.replace("import math\n", "")
    return f"{header}{cleaned}\n{export_block}"


def contains_blocked_prompt_text(user_prompt: str) -> tuple[bool, str]:
    lower_prompt = user_prompt.lower()
    for token in PROMPT_DENYLIST:
        if token in lower_prompt:
            return True, token
    return False, ""


def enforce_generate_cooldown(client_key: str) -> tuple[bool, int]:
    now_ts = datetime.now(timezone.utc).timestamp()
    last_ts = generation_last_seen.get(client_key, 0)
    elapsed = now_ts - last_ts
    if elapsed < GENERATE_COOLDOWN_SECONDS:
        wait_seconds = int(GENERATE_COOLDOWN_SECONDS - elapsed + 0.999)
        return False, wait_seconds

    generation_last_seen[client_key] = now_ts
    return True, 0


async def query_local_command(prompt: str) -> dict:
    """Send a prompt to local coder and parse the JSON action response."""
    try:
        full_prompt = f"{ARK_COMMAND_SYSTEM_PROMPT}\n\nUser command: {prompt}\n\nRespond with ONLY valid JSON."
        text = await asyncio.to_thread(query_local_coder, full_prompt)
        text = text.strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
            text = text.rsplit("```", 1)[0].strip()
        return json.loads(text)
    except json.JSONDecodeError:
        return {"action": "info", "params": {}, "description": text}
    except Exception as e:
        return {"action": "error", "params": {}, "description": str(e)}


async def query_local_blender_script(user_prompt: str, obj_output_path: Path) -> str:
    prompt = (
        f"{BLENDER_SCRIPT_SYSTEM_PROMPT}\n"
        "Output only Python code with no markdown and no explanation.\n"
        "Use bpy only. Clear default scene objects, build geometry that matches the request, and keep parts modular.\n"
        "Do not import os/sys/socket/subprocess/requests.\n"
        f"Output path: {obj_output_path.as_posix()}\n"
        f"User request: {user_prompt}\n"
    )

    raw_text = await asyncio.to_thread(query_local_coder, prompt)
    script_text = strip_code_fence(raw_text)
    script_text = normalize_obj_export(script_text)
    script_text = script_text.replace("{OBJ_PATH}", obj_output_path.as_posix())
    is_valid, reason = validate_blender_script(script_text)
    if not is_valid:
        raise ValueError(reason)
    return script_text


async def query_voice_intent(text: str) -> dict:
    prompt = (
        f"Analyze this user command: '{text}'. Return ONLY valid JSON. "
        "If making a new 3D model, return {\"intent\": \"generate_blender\", \"prompt\": \"<command>\"}. "
        "If modifying color/material of an existing part, return {\"intent\": \"edit_unity\", \"target\": \"<part_name>\", \"color\": \"<color_name>\"}."
    )
    raw = await asyncio.to_thread(query_local_coder, prompt)
    try:
        return json.loads(strip_code_fence(raw))
    except Exception:
        return {"intent": "generate_blender", "prompt": text}


async def run_blender_script(script_path: Path) -> tuple[bool, int, str, str, str]:
    if not blender_path:
        return False, -1, "", "", "blender cli was not found; set BLENDER_PATH or add blender to PATH"

    process = await asyncio.create_subprocess_exec(
        blender_path,
        "-b",
        "-P",
        str(script_path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=BLENDER_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        process.kill()
        await process.communicate()
        return False, -1, "", "", "blender process timed out"

    stdout_text = stdout.decode("utf-8", errors="replace")
    stderr_text = stderr.decode("utf-8", errors="replace")
    if process.returncode != 0:
        return False, process.returncode, stdout_text, stderr_text, "blender returned a non-zero exit code"
    return True, process.returncode, stdout_text, stderr_text, ""


async def check_blender_runtime() -> dict:
    if not blender_path:
        return {
            "available": False,
            "path": "",
            "version": "",
            "message": "blender cli was not found",
        }

    process = await asyncio.create_subprocess_exec(
        blender_path,
        "--version",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()
    stdout_text = stdout.decode("utf-8", errors="replace")
    stderr_text = stderr.decode("utf-8", errors="replace")
    first_line = stdout_text.splitlines()[0] if stdout_text.splitlines() else ""

    if process.returncode != 0:
        return {
            "available": False,
            "path": blender_path,
            "version": "",
            "message": stderr_text.strip() or "blender returned non-zero from --version",
        }

    return {
        "available": True,
        "path": blender_path,
        "version": first_line,
        "message": "ok",
    }


async def send_generation_event(ws: WebSocket | None, payload: dict) -> None:
    if not ws:
        return
    await ws.send_json(payload)


async def broadcast_unity_event(payload: dict, skip_ws: WebSocket | None = None) -> None:
    stale_clients: list[WebSocket] = []
    for client in connected_clients.copy():
        if skip_ws and client is skip_ws:
            continue
        try:
            await client.send_json(payload)
        except Exception:
            stale_clients.append(client)

    for client in stale_clients:
        if client in connected_clients:
            connected_clients.remove(client)


def validate_generate_request(prompt: str, client_key: str) -> tuple[bool, str]:
    if not prompt:
        return False, "prompt is required"
    if len(prompt) > MAX_PROMPT_LEN:
        return False, f"prompt must be <= {MAX_PROMPT_LEN} chars"
    blocked, token = contains_blocked_prompt_text(prompt)
    if blocked:
        return False, f"prompt contains blocked token: {token}"

    allowed, wait_seconds = enforce_generate_cooldown(client_key)
    if not allowed:
        return False, f"wait {wait_seconds}s before generating again"

    return True, ""


async def process_command_text(command_text: str) -> dict:
    result = await query_local_command(command_text)
    return {
        "command": command_text,
        "action": result.get("action", "info"),
        "params": result.get("params", {}),
        "description": result.get("description", ""),
    }


async def generate_obj_from_prompt(
    user_prompt: str,
    filename_hint: str,
    ws: WebSocket | None = None,
    source: str = "api",
) -> dict:
    global last_generation_status

    job_id = uuid.uuid4().hex[:10]
    safe_name = sanitize_filename(filename_hint or "model")
    script_path = GENERATED_SCRIPTS_DIR / f"{safe_name}_{job_id}.py"
    obj_path = GENERATED_OBJS_DIR / f"{safe_name}_{job_id}.obj"
    started_at = now_iso()

    await send_generation_event(ws, {
        "type": "generation_started",
        "job_id": job_id,
        "prompt": user_prompt,
        "source": source,
    })

    if not ws:
        await broadcast_unity_event({
            "type": "generation_started",
            "job_id": job_id,
            "prompt": user_prompt,
            "source": source,
        })

    try:
        script_text = await query_local_blender_script(user_prompt, obj_path)
        script_path.write_text(script_text, encoding="utf-8")

        if not blender_path:
            completed_at = now_iso()
            result = {
                "status": "script_ready",
                "job_id": job_id,
                "prompt": user_prompt,
                "obj_path": to_relative_artifact_path(obj_path),
                "script_path": to_relative_artifact_path(script_path),
                "started_at": started_at,
                "completed_at": completed_at,
                "message": "script generated but blender is not available",
            }
            generation_jobs.appendleft(result)
            last_generation_status = {
                "status": "script_ready",
                "job_id": job_id,
                "message": "script generated while blender was unavailable",
            }

            await send_generation_event(ws, {
                "type": "generation_script_ready",
                "job_id": job_id,
                "script_path": result["script_path"],
                "message": result["message"],
                "source": source,
            })
            await broadcast_unity_event({
                "type": "generation_script_ready",
                "job_id": job_id,
                "script_path": result["script_path"],
                "message": result["message"],
                "source": source,
            }, skip_ws=ws)
            return result

        ok, exit_code, stdout_text, stderr_text, error_message = await run_blender_script(script_path)
        if not ok:
            raise RuntimeError(error_message or f"blender exited with {exit_code}")

        if not obj_path.exists() or obj_path.stat().st_size < 64:
            raise RuntimeError("obj file was not created correctly")

        completed_at = now_iso()
        result = {
            "status": "success",
            "job_id": job_id,
            "prompt": user_prompt,
            "obj_path": to_relative_artifact_path(obj_path),
            "script_path": to_relative_artifact_path(script_path),
            "started_at": started_at,
            "completed_at": completed_at,
            "exit_code": exit_code,
            "stdout_tail": stdout_text[-1200:],
            "stderr_tail": stderr_text[-1200:],
            "message": "obj generated",
        }
        generation_jobs.appendleft(result)
        last_generation_status = {
            "status": "success",
            "job_id": job_id,
            "message": "last generation completed",
        }

        await send_generation_event(ws, {
            "type": "generation_complete",
            "job_id": job_id,
            "obj_path": result["obj_path"],
            "script_path": result["script_path"],
            "message": result["message"],
            "source": source,
        })
        await broadcast_unity_event({
            "type": "generation_complete",
            "job_id": job_id,
            "obj_path": result["obj_path"],
            "script_path": result["script_path"],
            "message": result["message"],
            "source": source,
        }, skip_ws=ws)
        return result

    except Exception as e:
        failed_at = now_iso()
        result = {
            "status": "failed",
            "job_id": job_id,
            "prompt": user_prompt,
            "obj_path": to_relative_artifact_path(obj_path),
            "script_path": to_relative_artifact_path(script_path),
            "started_at": started_at,
            "completed_at": failed_at,
            "message": str(e),
        }
        generation_jobs.appendleft(result)
        last_generation_status = {
            "status": "failed",
            "job_id": job_id,
            "message": str(e),
        }

        await send_generation_event(ws, {
            "type": "generation_failed",
            "job_id": job_id,
            "message": str(e),
            "source": source,
        })
        await broadcast_unity_event({
            "type": "generation_failed",
            "job_id": job_id,
            "message": str(e),
            "source": source,
        }, skip_ws=ws)
        return result


@app.on_event("startup")
async def startup_event():
    global blender_path, voice_service, image_to_3d_service
    ensure_generated_dirs()
    blender_path = resolve_blender_path()

    voice_service = VoiceCommandService(
        generate_fn=lambda prompt, filename: generate_obj_from_prompt(prompt, filename, source="voice"),
        command_fn=lambda command: process_command_text(command),
        notify_fn=lambda payload: broadcast_unity_event(payload),
        intent_fn=lambda text: query_voice_intent(text),
        wake_phrase=os.getenv("ARK_WAKE_PHRASE", "hello ark"),
    )

    image_to_3d_service = ImageTo3DService(
        generated_dir=GENERATED_DIR,
        notify_fn=lambda payload: broadcast_unity_event(payload),
    )


@app.on_event("shutdown")
async def shutdown_event():
    if voice_service:
        voice_service.stop()


# --- REST Endpoints ---

@app.get("/")
async def root():
    return {"status": "online", "project": "A.R.K.", "version": "0.1.0"}


@app.get("/health")
async def health():
    image_provider = {
        "configured": False,
        "mode": "local_triposr_cli",
        "python_executable": sys.executable,
    }
    if image_to_3d_service:
        image_provider = {
            "configured": image_to_3d_service.cli_available,
            "mode": "local_triposr_cli",
            "python_executable": sys.executable,
            "output_root": str(image_to_3d_service.obj_dir).replace("\\", "/"),
        }

    return {
        "status": "healthy",
        "local_coder_url": LOCAL_CODER_URL,
        "local_coder_model": LOCAL_CODER_MODEL,
        "connected_clients": len(connected_clients),
        "blender_available": bool(blender_path),
        "blender_path": blender_path,
        "generated_dir": str(GENERATED_DIR),
        "last_generation_status": last_generation_status,
        "image_to_3d": image_provider,
    }


@app.post("/api/command")
async def rest_command(body: dict):
    """REST endpoint so you can test commands without WebSocket."""
    prompt = body.get("command", "")
    result = await process_command_text(prompt)
    return {"input": prompt, "result": result}


@app.post("/api/generate-obj")
async def rest_generate_obj(body: dict):
    prompt = str(body.get("prompt", "")).strip()
    filename_hint = str(body.get("filename", "model")).strip()

    ok, message = validate_generate_request(prompt, "rest")
    if not ok:
        if message.startswith("wait "):
            raise HTTPException(status_code=429, detail=message)
        raise HTTPException(status_code=400, detail=message)

    result = await generate_obj_from_prompt(prompt, filename_hint, source="rest")
    if result.get("status") not in ["success", "script_ready"]:
        raise HTTPException(status_code=500, detail=result)
    return result


@app.get("/api/generation-jobs")
async def get_generation_jobs():
    return {"jobs": list(generation_jobs)}


@app.get("/api/blender-check")
async def blender_check():
    return await check_blender_runtime()


@app.post("/api/image-to-3d")
async def create_image_to_3d_job(
    prompt: str = Form(...),
    filename: str = Form("image-model"),
    image: UploadFile = File(...),
):
    if not image_to_3d_service:
        raise HTTPException(status_code=500, detail="image-to-3d service unavailable")

    content_type = (image.content_type or "").lower()
    if content_type and not content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="uploaded file must be an image")

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="image file is empty")

    job = await image_to_3d_service.create_job(
        prompt=prompt.strip(),
        image_bytes=image_bytes,
        filename_hint=filename.strip() or "image-model",
        original_name=image.filename or "input.jpg",
    )
    return {"status": "accepted", "job": job}


@app.get("/api/image-to-3d/jobs")
async def list_image_to_3d_jobs():
    if not image_to_3d_service:
        raise HTTPException(status_code=500, detail="image-to-3d service unavailable")
    return {"jobs": await image_to_3d_service.list_jobs()}


@app.get("/api/image-to-3d/jobs/{job_id}")
async def get_image_to_3d_job(job_id: str):
    if not image_to_3d_service:
        raise HTTPException(status_code=500, detail="image-to-3d service unavailable")
    job = await image_to_3d_service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    return {"job": job}


@app.post("/api/prompt-to-3d")
async def create_prompt_to_3d_job(
    prompt: str = Form(...),
    filename_hint: str = Form(default="generated"),
):
    """Generate a 3D model from a text prompt using local coder-assisted mesh fallback."""
    if not image_to_3d_service:
        raise HTTPException(status_code=500, detail="image-to-3d service unavailable")
    job = image_to_3d_service.create_prompt_job(prompt, filename_hint)
    return {"job": job}


@app.get("/api/image-to-3d/provider-check")
async def image_to_3d_provider_check():
    if not image_to_3d_service:
        raise HTTPException(status_code=500, detail="image-to-3d service unavailable")
    return {
        "mode": "local_triposr_cli",
        "configured": image_to_3d_service.cli_available,
        "output_root": str(image_to_3d_service.obj_dir).replace("\\", "/"),
        "python_executable": sys.executable,
    }


@app.get("/api/voice/status")
async def voice_status():
    if not voice_service:
        return {"running": False, "message": "voice service unavailable"}
    return voice_service.status()


@app.post("/api/voice/start")
async def voice_start():
    if not voice_service:
        raise HTTPException(status_code=500, detail="voice service unavailable")

    loop = asyncio.get_running_loop()
    ok, message = voice_service.start(loop)
    if not ok:
        raise HTTPException(status_code=500, detail=message)
    return {"status": "ok", "message": message, "voice": voice_service.status()}


@app.post("/api/voice/stop")
async def voice_stop():
    if not voice_service:
        raise HTTPException(status_code=500, detail="voice service unavailable")
    ok, message = voice_service.stop()
    if not ok:
        raise HTTPException(status_code=500, detail=message)
    return {"status": "ok", "message": message, "voice": voice_service.status()}


# --- WebSocket for Unity Communication ---

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    connected_clients.append(ws)
    print(f"[ARK] Unity client connected. Total clients: {len(connected_clients)}")

    try:
        await ws.send_json({"type": "connection", "message": "Connected to A.R.K. server"})

        while True:
            data = await ws.receive_text()
            print(f"[ARK] Received: {data}")

            try:
                msg = json.loads(data)
            except json.JSONDecodeError:
                await ws.send_json({"type": "error", "message": "Invalid JSON"})
                continue

            msg_type = msg.get("type", "")

            if msg_type == "ping":
                await ws.send_json({"type": "pong"})

            elif msg_type == "command":
                command_text = msg.get("command", "")
                print(f"[ARK] Command: {command_text}")
                result = await process_command_text(command_text)
                print(f"[ARK] Local coder response: {result}")
                await ws.send_json({
                    "type": "command_result",
                    **result,
                })

            elif msg_type == "generate":
                prompt = msg.get("prompt", "")
                filename_hint = msg.get("filename", "model")
                ok, message = validate_generate_request(prompt, f"ws:{id(ws)}")
                if not ok:
                    await ws.send_json({"type": "error", "message": message})
                    continue

                print(f"[ARK] Generate request: {prompt}")
                result = await generate_obj_from_prompt(prompt, filename_hint, ws=ws, source="ws")
                await ws.send_json({
                    "type": "generate_result",
                    "prompt": prompt,
                    "result": result,
                })

            elif msg_type == "voice_start":
                if not voice_service:
                    await ws.send_json({"type": "error", "message": "voice service unavailable"})
                    continue
                ok, message = voice_service.start(asyncio.get_running_loop())
                if not ok:
                    await ws.send_json({"type": "error", "message": message})
                    continue
                await ws.send_json({"type": "voice_status", "status": "running", "message": message})

            elif msg_type == "voice_stop":
                if not voice_service:
                    await ws.send_json({"type": "error", "message": "voice service unavailable"})
                    continue
                ok, message = voice_service.stop()
                if not ok:
                    await ws.send_json({"type": "error", "message": message})
                    continue
                await ws.send_json({"type": "voice_status", "status": "stopped", "message": message})

            else:
                await ws.send_json({"type": "echo", "received": msg})

    except WebSocketDisconnect:
        connected_clients.remove(ws)
        print(f"[ARK] Client disconnected. Total clients: {len(connected_clients)}")


# --- Run ---
if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    ensure_generated_dirs()
    blender_path = resolve_blender_path()
    print(f"[ARK] Starting server on {host}:{port}")
    print(f"[ARK] Local coder URL: {LOCAL_CODER_URL}")
    print(f"[ARK] Local coder model: {LOCAL_CODER_MODEL}")
    print(f"[ARK] Blender path: {blender_path if blender_path else 'MISSING'}")
    uvicorn.run(app, host=host, port=port)
