# A.R.K.-2

Local-first FastAPI backend for the A.R.K. capstone prototype.

## Source Of Truth

All execution rules and migration phases are centralized in:

- [Agents/Master_Execution_Prompt.txt](Agents/Master_Execution_Prompt.txt)

## Current Scope

Implemented now:

- Local LLM command and Blender script generation through Ollama Qwen2.5-Coder
- Local image-to-3D flow through TripoSR CLI hook
- Voice intent routing for `generate_blender` and `edit_unity`
- WebSocket broadcast to Unity clients
- Health smoke testing

Deferred for later:

- Webcam job pipeline

## Quick Start

1. Create and activate a Python virtual environment.
2. Install dependencies:

```bash
cd server
pip install -r requirements.txt
```

3. Ensure Ollama is running and has the configured model.
4. Ensure Blender CLI is installed and accessible.
5. Start API server:

```bash
cd server
python main.py
```

## Key Environment Variables

Set in [server/.env](server/.env):

- `LOCAL_CODER_URL` (default: `http://localhost:11434/api/generate`)
- `LOCAL_CODER_MODEL` (default: `qwen2.5-coder:7b`)
- `BLENDER_PATH` (optional if Blender is in PATH)
- `ARK_WAKE_PHRASE` (optional, default `hello ark`)

## API Endpoints

Core:

- `GET /health`
- `GET /api/blender-check`
- `POST /api/command`
- `POST /api/generate-obj`
- `GET /api/generation-jobs`

Image-to-3D:

- `POST /api/image-to-3d`
- `GET /api/image-to-3d/jobs`
- `GET /api/image-to-3d/jobs/{job_id}`
- `GET /api/image-to-3d/provider-check`

Voice:

- `GET /api/voice/status`
- `POST /api/voice/start`
- `POST /api/voice/stop`

WebSocket:

- `WS /ws`

## Run Tests

From repo root:

```bash
.\.venv\Scripts\python.exe -m unittest discover -s server/tests -p "test_*.py"
```

## Notes

- Generated runtime artifacts are written under [server/generated](server/generated).
- Meshes must remain modular; do not join objects during generation.