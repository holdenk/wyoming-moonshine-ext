"""Code for transcription using the sherpa-onnx library."""

import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np

from .const import Transcriber

import os

from moonshine_voice import Transcriber as MT
from moonshine_voice import (
    load_wav_file,
    get_model_for_language,
    ModelArch
)


_LOGGER = logging.getLogger(__name__)



class MoonshineTranscriber(Transcriber):
    """Wrapper for moonshine model."""

    def __init__(self, model_id: str, language: str, cache_dir: Union[str, Path]) -> None:
        """Initialize model."""
        cache_dir = Path(cache_dir)
        # Hack: moonshine has it's own loader and it uses the env var MOONSHINE_VOICE_CACHE to resolve it
        os.environ
        if language is None:
            language = "en"
        model_size = ModelArch.SMALL_STREAMING
        if model_id == "medium":
            model_size = ModelArch.MEDIUM_STREAMING
        elif model_id == "tiny":
            model_size = ModelArch.TINY
        model_path, model_arch = get_model_for_language(language, model_size)
        self.recognizer = MT(model_path=model_path, model_arch=model_arch)


        # Prime model so that the first transcription will be fast
        self.recognizer.start()
        self.recognizer.transcribe_without_streaming(np.zeros(shape=(128), dtype=np.float32), 32)

    def transcribe(
        self,
        wav_path: Union[str, Path],
        language: Optional[str],
        beam_size: int = 5,
        initial_prompt: Optional[str] = None,
    ) -> str:
        """Returns transcription for WAV file."""
        audio_data, sample_rate = load_wav_file(wav_path)

        result = self.recognizer.transcribe_without_streaming(audio_data, sample_rate)
        _LOGGER.debug(f"Got {result}")
        return result
