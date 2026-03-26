# A.R.K.-2

fastapi backend for ark prototype.

## quick start

1. install python deps in `server/requirements.txt`
2. set `GEMINI_API_KEY` in `server/.env`
3. make sure blender cli is available in `PATH` or set `BLENDER_PATH`
4. run server:

```bash
cd server
python main.py
```

## health check

`GET /health` returns runtime status including blender availability and last generation result.

## generate obj with rest

endpoint: `POST /api/generate-obj`

request body:

```json
{
	"prompt": "make a simple tennis racket handle grip with small grooves",
	"filename": "racket-grip"
}
```

success response includes:

- `job_id`
- `obj_path` (relative path like `generated/objs/...`)
- `script_path` (saved generated blender script)
- `status` set to `success`

notes:

- prompt max length is 300 chars
- generation has a short cooldown (429 if called too fast)
- some blocked tokens are rejected for safety

## generate obj with websocket

connect to `ws://localhost:8000/ws`

send:

```json
{
	"type": "generate",
	"prompt": "make a low poly gear",
	"filename": "gear"
}
```

events:

- `generation_started`
- `generation_complete` or `generation_failed`
- `generate_result` (final payload wrapper)

## generated files

- scripts: `server/generated/scripts`
- objs: `server/generated/objs`

## debug endpoints

- `GET /api/generation-jobs` returns recent generation jobs (in-memory history)