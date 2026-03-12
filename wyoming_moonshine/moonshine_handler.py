"""Code for transcription using the moonshine-voice library."""

import logging
import os
from pathlib import Path
from typing import Optional, Union

import numpy as np
from moonshine_voice import ModelArch  # pylint: disable=no-name-in-module
from moonshine_voice import Transcriber as MT  # pylint: disable=no-name-in-module
from moonshine_voice import (  # pylint: disable=no-name-in-module
    get_model_for_language,
    load_wav_file,
)

from .const import Transcriber

_LOGGER = logging.getLogger(__name__)


class MoonshineTranscriber(Transcriber):
    """Wrapper for moonshine model."""

    def __init__(
        self, model_id: str, language: str, cache_dir: Union[str, Path]
    ) -> None:
        """Initialize model."""
        cache_dir = Path(cache_dir)
        os.environ["MOONSHINE_VOICE_CACHE"] = str(cache_dir)

        if language is None:
            language = "en"

        model_size = ModelArch.SMALL_STREAMING
        if model_id == "medium":
            model_size = ModelArch.MEDIUM_STREAMING
        elif model_id == "tiny":
            model_size = ModelArch.TINY

        _LOGGER.debug("Starting moonshine model %s %s", language, model_size)
        model_path, model_arch = get_model_for_language(language, model_size)
        self.recognizer = MT(model_path=model_path, model_arch=model_arch)

        # Prime model so that the first transcription will be fast
        _LOGGER.debug("Priming model with zeros...")
        try:
            self.recognizer.transcribe_without_streaming(
                np.zeros(shape=(128,), dtype=np.float32), 32
            )
        except Exception as e:
            _LOGGER.debug("Error priming model: %s", e)
        _LOGGER.debug("Model ready.")

    def transcribe(
        self,
        wav_path: Union[str, Path],
        language: Optional[str],
    ) -> str:
        """Return transcription for WAV file."""
        audio_data, sample_rate = load_wav_file(wav_path)

        transcript = self.recognizer.transcribe_without_streaming(
            audio_data, sample_rate
        )
        _LOGGER.debug("Got %s", transcript)
        return "\n".join(line.text for line in transcript.lines)
