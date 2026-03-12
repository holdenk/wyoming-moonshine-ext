"""Constants."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Union

AUTO_LANGUAGE = "auto"
AUTO_MODEL = "auto"


class Transcriber(ABC):
    """Base class for transcribers."""

    @abstractmethod
    def transcribe(self, wav_path: Union[str, Path], language: Optional[str]) -> str:
        pass
