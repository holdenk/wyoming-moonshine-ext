"""Logic for model loading."""

import asyncio
import logging
import platform
from pathlib import Path
from typing import Optional, Union

from .const import Transcriber
from .moonshine_handler import MoonshineTranscriber

_LOGGER = logging.getLogger(__name__)


class ModelLoader:
    """Load moonshine transcriber."""

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
        self._lock = asyncio.Lock()

    async def load_transcriber(self, language: Optional[str] = None) -> Transcriber:
        """Load or get cached transcriber."""
        async with self._lock:
            if self._transcriber is not None:
                return self._transcriber

            model = self.model
            if model is None:
                machine = platform.machine().lower()
                is_arm = ("arm" in machine) or ("aarch" in machine)
                model = "small" if is_arm else "medium"
                _LOGGER.debug("Auto-selected model: %s", model)

            lang = language or self.preferred_language or "en"

            self._transcriber = MoonshineTranscriber(
                model_id=model,
                language=lang,
                cache_dir=self.download_dir,
            )

            _LOGGER.debug("Loaded moonshine transcriber with model %s", model)
            return self._transcriber
