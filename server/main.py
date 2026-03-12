"""
A.R.K. - Augmented Reality Kinetic Interface
FastAPI Backend Server
"""

import json
import os
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

# --- Gemini Setup ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

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


async def ask_gemini(prompt: str) -> dict:
    """Send a prompt to Gemini and parse the JSON action response."""
    try:
        full_prompt = f"{ARK_SYSTEM_PROMPT}\n\nUser command: {prompt}\n\nRespond with ONLY valid JSON."
        response = model.generate_content(full_prompt)
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
    }


@app.post("/api/command")
async def rest_command(body: dict):
    """REST endpoint so you can test commands without WebSocket."""
    prompt = body.get("command", "")
    result = await ask_gemini(prompt)
    return {"input": prompt, "result": result}


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
                print(f"[ARK] Generate request: {prompt}")
                result = await ask_gemini(f"Generate a 3D model description for: {prompt}")
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
    print(f"[ARK] Starting server on {host}:{port}")
    print(f"[ARK] Gemini API key: {'configured' if GEMINI_API_KEY else 'MISSING'}")
    uvicorn.run(app, host=host, port=port)
