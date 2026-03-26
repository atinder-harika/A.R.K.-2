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
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

# --- Gemini Setup ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
genai.configure(api_key=GEMINI_API_KEY)
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
model = None
active_model_name = ""

ARK_SYSTEM_PROMPT = (
    "You are A.R.K., an Augmented Reality Kinetic Interface assistant. "
    "You help users manipulate 3D models through natural language. "
    "When a user gives a command, respond with a short JSON object containing: "
    '"action" (one of: rotate, scale, move, texture, color, explode, reset, info), '
    '"params" (a dict of parameters like axis, value, color hex, etc.), '
    'and "description" (a short human-readable summary of what you did). '
    "Keep responses concise. If the command is conversational, set action to info."
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


def select_gemini_model_name() -> str:
    if GEMINI_MODEL:
        return GEMINI_MODEL

    try:
        for item in genai.list_models():
            methods = getattr(item, "supported_generation_methods", []) or []
            if "generateContent" in methods:
                name = getattr(item, "name", "")
                if name.startswith("models/"):
                    return name.replace("models/", "", 1)
                return name
    except Exception:
        return ""

    return ""


def initialize_gemini_model() -> None:
    global model, active_model_name
    selected = select_gemini_model_name()
    active_model_name = selected
    model = genai.GenerativeModel(selected) if selected else None


def refresh_model_if_missing(error_text: str) -> bool:
    global model, active_model_name
    lower_error = error_text.lower()
    if "not found" not in lower_error and "unsupported" not in lower_error:
        return False

    try:
        for item in genai.list_models():
            methods = getattr(item, "supported_generation_methods", []) or []
            if "generateContent" not in methods:
                continue
            name = getattr(item, "name", "")
            candidate = name.replace("models/", "", 1) if name.startswith("models/") else name
            if not candidate or candidate == active_model_name:
                continue
            active_model_name = candidate
            model = genai.GenerativeModel(candidate)
            return True
    except Exception:
        return False

    return False


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


async def ask_gemini(prompt: str) -> dict:
    """Send a prompt to Gemini and parse the JSON action response."""
    try:
        if not model:
            return {"action": "error", "params": {}, "description": "no gemini model is configured"}

        full_prompt = f"{ARK_SYSTEM_PROMPT}\n\nUser command: {prompt}\n\nRespond with ONLY valid JSON."
        try:
            response = model.generate_content(full_prompt)
        except Exception as first_error:
            if refresh_model_if_missing(str(first_error)) and model:
                response = model.generate_content(full_prompt)
            else:
                raise

        text = response.text.strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
            text = text.rsplit("```", 1)[0].strip()
        return json.loads(text)
    except json.JSONDecodeError:
        return {"action": "info", "params": {}, "description": response.text.strip()}
    except Exception as e:
        return {"action": "error", "params": {}, "description": str(e)}


async def ask_gemini_blender_script(user_prompt: str, obj_output_path: Path) -> str:
    if not model:
        raise ValueError("no gemini model is configured")

    prompt = (
        "you write safe blender python scripts for blender headless mode. "
        "output only python code with no markdown and no explanation. "
        "constraints: use bpy only, clear default scene objects, build geometry that matches the request, "
        "set basic transforms if needed. do not call any export operator directly. "
        "just build geometry. export is handled by the server wrapper. "
        "do not import os/sys/socket/subprocess/requests. do not read or write any files except this obj export. "
        "output path: "
        f"{obj_output_path.as_posix()}\n"
        f"user request: {user_prompt}\n"
    )

    try:
        response = model.generate_content(prompt)
    except Exception as first_error:
        if refresh_model_if_missing(str(first_error)) and model:
            response = model.generate_content(prompt)
        else:
            raise

    script_text = strip_code_fence(response.text)
    script_text = normalize_obj_export(script_text)
    script_text = script_text.replace("{OBJ_PATH}", obj_output_path.as_posix())
    is_valid, reason = validate_blender_script(script_text)
    if not is_valid:
        raise ValueError(reason)
    return script_text


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


async def generate_obj_from_prompt(user_prompt: str, filename_hint: str, ws: WebSocket | None = None) -> dict:
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
    })

    try:
        script_text = await ask_gemini_blender_script(user_prompt, obj_path)
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
            })
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
        })
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
        })
        return result


@app.on_event("startup")
async def startup_event():
    global blender_path
    ensure_generated_dirs()
    blender_path = resolve_blender_path()
    initialize_gemini_model()


# --- REST Endpoints ---

@app.get("/")
async def root():
    return {"status": "online", "project": "A.R.K.", "version": "0.1.0"}


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "gemini_key_set": bool(GEMINI_API_KEY),
        "connected_clients": len(connected_clients),
        "gemini_model": active_model_name,
        "blender_available": bool(blender_path),
        "blender_path": blender_path,
        "generated_dir": str(GENERATED_DIR),
        "last_generation_status": last_generation_status,
    }


@app.post("/api/command")
async def rest_command(body: dict):
    """REST endpoint so you can test commands without WebSocket."""
    prompt = body.get("command", "")
    result = await ask_gemini(prompt)
    return {"input": prompt, "result": result}


@app.post("/api/generate-obj")
async def rest_generate_obj(body: dict):
    prompt = str(body.get("prompt", "")).strip()
    filename_hint = str(body.get("filename", "model")).strip()

    if not prompt:
        raise HTTPException(status_code=400, detail="prompt is required")
    if len(prompt) > MAX_PROMPT_LEN:
        raise HTTPException(status_code=400, detail=f"prompt must be <= {MAX_PROMPT_LEN} chars")
    blocked, token = contains_blocked_prompt_text(prompt)
    if blocked:
        raise HTTPException(status_code=400, detail=f"prompt contains blocked token: {token}")

    allowed, wait_seconds = enforce_generate_cooldown("rest")
    if not allowed:
        raise HTTPException(status_code=429, detail=f"wait {wait_seconds}s before generating again")

    result = await generate_obj_from_prompt(prompt, filename_hint)
    if result.get("status") not in ["success", "script_ready"]:
        raise HTTPException(status_code=500, detail=result)
    return result


@app.get("/api/generation-jobs")
async def get_generation_jobs():
    return {"jobs": list(generation_jobs)}


@app.get("/api/blender-check")
async def blender_check():
    return await check_blender_runtime()


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
                # Send to Gemini for processing
                result = await ask_gemini(command_text)
                print(f"[ARK] Gemini response: {result}")
                await ws.send_json({
                    "type": "command_result",
                    "command": command_text,
                    "action": result.get("action", "info"),
                    "params": result.get("params", {}),
                    "description": result.get("description", ""),
                })

            elif msg_type == "generate":
                prompt = msg.get("prompt", "")
                filename_hint = msg.get("filename", "model")
                if not prompt:
                    await ws.send_json({"type": "error", "message": "prompt is required"})
                    continue
                if len(prompt) > MAX_PROMPT_LEN:
                    await ws.send_json({"type": "error", "message": f"prompt must be <= {MAX_PROMPT_LEN} chars"})
                    continue
                blocked, token = contains_blocked_prompt_text(prompt)
                if blocked:
                    await ws.send_json({"type": "error", "message": f"prompt contains blocked token: {token}"})
                    continue

                allowed, wait_seconds = enforce_generate_cooldown(f"ws:{id(ws)}")
                if not allowed:
                    await ws.send_json({"type": "error", "message": f"wait {wait_seconds}s before generating again"})
                    continue

                print(f"[ARK] Generate request: {prompt}")
                result = await generate_obj_from_prompt(prompt, filename_hint, ws=ws)
                await ws.send_json({
                    "type": "generate_result",
                    "prompt": prompt,
                    "result": result,
                })

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
    initialize_gemini_model()
    print(f"[ARK] Starting server on {host}:{port}")
    print(f"[ARK] Gemini API key: {'configured' if GEMINI_API_KEY else 'MISSING'}")
    print(f"[ARK] Gemini model: {active_model_name if active_model_name else 'MISSING'}")
    print(f"[ARK] Blender path: {blender_path if blender_path else 'MISSING'}")
    uvicorn.run(app, host=host, port=port)
