"""Compatibility shim for Python 3.13.

SpeechRecognition still imports the removed stdlib `aifc` module on newer
Python versions. The backend only needs that import to succeed so the WAV
fallback path can load.
"""

from __future__ import annotations

import wave


Error = wave.Error


def open(*args, **kwargs):
    raise NotImplementedError("AIFF/AIFC audio support is not available in this compatibility shim")
