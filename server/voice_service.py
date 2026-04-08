import asyncio
import threading
import time
from datetime import datetime, timezone
from typing import Awaitable, Callable

try:
    import speech_recognition as sr  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - optional dependency
    sr = None


GenerateFn = Callable[[str, str], Awaitable[dict]]
CommandFn = Callable[[str], Awaitable[dict]]
NotifyFn = Callable[[dict], Awaitable[None]]
IntentFn = Callable[[str], Awaitable[dict]]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class VoiceCommandService:
    def __init__(
        self,
        generate_fn: GenerateFn,
        command_fn: CommandFn,
        notify_fn: NotifyFn,
        intent_fn: IntentFn | None = None,
        wake_phrase: str = "hello ark",
        phrase_limit_seconds: int = 8,
    ) -> None:
        self.generate_fn = generate_fn
        self.command_fn = command_fn
        self.notify_fn = notify_fn
        self.intent_fn = intent_fn
        self.wake_phrase = wake_phrase.strip().lower()
        self.phrase_limit_seconds = phrase_limit_seconds
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._running = False
        self._last_error = ""
        self._last_transcript = ""
        self._commands_handled = 0

    def status(self) -> dict:
        return {
            "running": self._running,
            "wake_phrase": self.wake_phrase,
            "last_error": self._last_error,
            "last_transcript": self._last_transcript,
            "commands_handled": self._commands_handled,
            "speech_recognition_available": bool(sr),
        }

    def start(self, loop: asyncio.AbstractEventLoop) -> tuple[bool, str]:
        if self._running:
            return True, "voice service already running"
        if not sr:
            self._last_error = "SpeechRecognition is not installed"
            return False, self._last_error

        self._loop = loop
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._thread_loop, daemon=True)
        self._thread.start()
        self._running = True
        self._last_error = ""
        return True, "voice service started"

    def stop(self) -> tuple[bool, str]:
        if not self._running:
            return True, "voice service already stopped"

        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3)

        self._running = False
        return True, "voice service stopped"

    def _submit_async(self, coro: Awaitable[None]) -> None:
        if not self._loop:
            return
        asyncio.run_coroutine_threadsafe(coro, self._loop)

    def _thread_loop(self) -> None:
        recognizer = sr.Recognizer()
        try:
            with sr.Microphone() as source:
                recognizer.adjust_for_ambient_noise(source, duration=0.6)
                self._submit_async(self.notify_fn({
                    "type": "voice_status",
                    "status": "ready",
                    "wake_phrase": self.wake_phrase,
                    "timestamp": now_iso(),
                }))

                while not self._stop_event.is_set():
                    try:
                        audio = recognizer.listen(source, timeout=1, phrase_time_limit=self.phrase_limit_seconds)
                    except sr.WaitTimeoutError:
                        continue

                    try:
                        transcript = recognizer.recognize_google(audio)
                    except sr.UnknownValueError:
                        continue
                    except Exception as e:  # pragma: no cover - depends on external stt service
                        self._last_error = str(e)
                        self._submit_async(self.notify_fn({
                            "type": "voice_status",
                            "status": "stt_error",
                            "message": self._last_error,
                            "timestamp": now_iso(),
                        }))
                        time.sleep(0.4)
                        continue

                    self._last_transcript = transcript
                    lower = transcript.strip().lower()
                    if not lower.startswith(self.wake_phrase):
                        continue

                    spoken_command = lower[len(self.wake_phrase):].strip(" ,.:;")
                    if not spoken_command:
                        self._submit_async(self.notify_fn({
                            "type": "voice_status",
                            "status": "wake_detected",
                            "message": "wake phrase heard; no command after wake phrase",
                            "timestamp": now_iso(),
                        }))
                        continue

                    self._commands_handled += 1
                    self._submit_async(self._handle_command(spoken_command, transcript))
        except Exception as e:  # pragma: no cover - hardware dependent
            self._last_error = str(e)
            self._submit_async(self.notify_fn({
                "type": "voice_status",
                "status": "fatal_error",
                "message": self._last_error,
                "timestamp": now_iso(),
            }))
        finally:
            self._running = False

    async def _handle_command(self, normalized_command: str, original_transcript: str) -> None:
        await self.notify_fn({
            "type": "voice_command_received",
            "command": normalized_command,
            "transcript": original_transcript,
            "timestamp": now_iso(),
        })

        if self.intent_fn:
            intent_json = await self.intent_fn(normalized_command)
            intent = str(intent_json.get("intent", "")).strip().lower()

            if intent == "generate_blender":
                prompt = str(intent_json.get("prompt", "")).strip()
                if not prompt:
                    await self.notify_fn({
                        "type": "voice_command_result",
                        "status": "ignored",
                        "message": "generate_blender intent missing prompt",
                        "timestamp": now_iso(),
                    })
                    return

                result = await self.generate_fn(prompt, "voice-model")
                await self.notify_fn({
                    "type": "voice_command_result",
                    "status": "ok",
                    "mode": "generate",
                    "intent": intent_json,
                    "result": result,
                    "timestamp": now_iso(),
                })
                return

            if intent == "edit_unity":
                await self.notify_fn({
                    "type": "edit_unity",
                    **intent_json,
                    "timestamp": now_iso(),
                })
                await self.notify_fn({
                    "type": "voice_command_result",
                    "status": "ok",
                    "mode": "edit_unity",
                    "intent": intent_json,
                    "timestamp": now_iso(),
                })
                return

        if normalized_command.startswith("generate a model of"):
            prompt = normalized_command.replace("generate a model of", "", 1).strip()
            if not prompt:
                await self.notify_fn({
                    "type": "voice_command_result",
                    "status": "ignored",
                    "message": "generation command missing prompt",
                    "timestamp": now_iso(),
                })
                return
            result = await self.generate_fn(prompt, "voice-model")
            await self.notify_fn({
                "type": "voice_command_result",
                "status": "ok",
                "mode": "generate",
                "result": result,
                "timestamp": now_iso(),
            })
            return

        if normalized_command.startswith("generate"):
            prompt = normalized_command.replace("generate", "", 1).strip()
            if prompt.startswith("a model of"):
                prompt = prompt.replace("a model of", "", 1).strip()
            if not prompt:
                await self.notify_fn({
                    "type": "voice_command_result",
                    "status": "ignored",
                    "message": "generation command missing prompt",
                    "timestamp": now_iso(),
                })
                return
            result = await self.generate_fn(prompt, "voice-model")
            await self.notify_fn({
                "type": "voice_command_result",
                "status": "ok",
                "mode": "generate",
                "result": result,
                "timestamp": now_iso(),
            })
            return

        result = await self.command_fn(normalized_command)
        await self.notify_fn({
            "type": "voice_command_result",
            "status": "ok",
            "mode": "command",
            "result": result,
            "timestamp": now_iso(),
        })
