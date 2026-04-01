"""Event handler for clients of the server."""

import asyncio
import logging
import os
import tempfile
import wave
from typing import Any, Optional

from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioChunkConverter, AudioStop, AudioStart
from wyoming.event import Event
from wyoming.info import Describe, Info
from wyoming.server import AsyncEventHandler

from .moonshine_handler import MoonshineTranscriber

_LOGGER = logging.getLogger(__name__)


class DispatchEventHandler(AsyncEventHandler):
    """Dispatches to moonshine transcriber."""

    def __init__(
        self,
        wyoming_info: Info,
        transcriber: MoonshineTranscriber,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.wyoming_info_event = wyoming_info.event()

        self._language: Optional[str] = None
        self._transcriber = transcriber

        self._audio_converter = AudioChunkConverter(rate=16000, width=2, channels=1)

    async def handle_event(self, event: Event) -> bool:
        _LOGGER.debug("Received event: %s", event.type)
        if AudioStart.is_type(event.type):
            _LOGGER.debug("Start of audio received, starting session")
            await self._transcriber.start_transcription()
            return True

        if AudioChunk.is_type(event.type):
            _LOGGER.debug("Audio chunk received")
            chunk = self._audio_converter.convert(AudioChunk.from_event(event))
            await self._transcriber.queue_chunk(chunk.audio, chunk.rate)
            return True

        if AudioStop.is_type(event.type):
            _LOGGER.debug("Audio stopped")

            text = await self._transcriber.get_and_clear_transcription()

            _LOGGER.info(text)

            await self.write_event(Transcript(text=text).event())
            _LOGGER.debug("Completed request")

            return False

        if Transcribe.is_type(event.type):
            transcribe = Transcribe.from_event(event)
            self._language = transcribe.language or self._loader.preferred_language
            _LOGGER.debug("Language set to %s", self._language)

            return True

        if Describe.is_type(event.type):
            await self.write_event(self.wyoming_info_event)
            _LOGGER.debug("Sent info")
            return True

        _LOGGER.warning("Unknown event type: %s", event.type)

        return True
