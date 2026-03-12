"""Logic for model selection, loading, and transcription."""

import asyncio
import logging
import platform
from pathlib import Path
from typing import Optional, Union

from .const import Transcriber
from .moonshine_handler import MoonshineTranscriber

_LOGGER = logging.getLogger(__name__)


class ModelLoader:
    """Load and cache moonshine transcriber."""

    def __init__(
        self,
        preferred_language: Optional[str],
        download_dir: Union[str, Path],
        model: Optional[str],
    ) -> None:
        self.preferred_language = preferred_language
        self.download_dir = Path(download_dir)
        self.model = model

        self._transcriber: Optional[Transcriber] = None
        self._transcriber_lock = asyncio.Lock()

    async def load_transcriber(self, language: Optional[str] = None) -> Transcriber:
        """Load or get cached transcriber."""
        async with self._transcriber_lock:
            if self._transcriber is not None:
                return self._transcriber

            language = language or self.preferred_language or "en"
            model = self.model
            if model is None:
                machine = platform.machine().lower()
                is_arm = ("arm" in machine) or ("aarch" in machine)
                model = "small" if is_arm else "medium"

            _LOGGER.debug(
                "Loading moonshine model '%s' for language '%s'", model, language
            )

            self._transcriber = await asyncio.to_thread(
                MoonshineTranscriber,
                model,
                language=language,
                cache_dir=self.download_dir,
            )

        return self._transcriber

    async def transcribe(
        self, wav_path: Union[str, Path], language: Optional[str]
    ) -> str:
        """Transcribe WAV file using moonshine transcriber.

        Assumes WAV file is 16kHz 16-bit mono PCM.
        """
        transcriber = await self.load_transcriber(language)
        text = await asyncio.to_thread(
            transcriber.transcribe,
            wav_path,
            language,
        )
        _LOGGER.debug("Transcribed audio: %s", text)
        return text
