"""Code for transcription using the moonshine library."""

import logging
import os
from pathlib import Path
from typing import Optional, Union

import numpy as np
from moonshine_voice import ModelArch
from moonshine_voice import Transcriber as MT
from moonshine_voice import TranscriptEventListener, LineStarted, LineTextChanged, LineCompleted, Error
from moonshine_voice import get_model_for_language, load_wav_file

_LOGGER = logging.getLogger(__name__)

class AccumulatingListener(TranscriptEventListener):
    def __init__(self):
        _LOGGER.debug("Initializing AccumulatingListener")
        self.lines = None
        self.offset = 0

    def on_line_started(self, event: LineStarted) -> None:
        """Called when a new transcription line starts."""
        _LOGGER.debug("Starting new line with id %s", event.line.line_id)
        if not self.lines:
            self.lines = [""]
            self.offset = event.line.line_id
        if len(self.lines) + 1 < event.line.line_id-self.offset:
            raise Exception(
                "Got unexpected line id %s, expected at most %s",
                event.line.line_id+self.offset,
                len(self.lines) + 1,
            )

    def on_line_text_changed(self, event: LineTextChanged) -> None:
        """Called when a transcription line is completed."""
        _LOGGER.debug("Text changed: Setting line to %s", event.line.text)
        self.lines[event.line.line_id-self.offset] = event.line.text

    def on_line_completed(self, event: LineCompleted) -> None:
        """Called when a transcription line is completed."""
        _LOGGER.debug("Line completed: Setting line to %s", event.line.text)
        self.lines[event.line.line_id-self.offset] = event.line.text

    def on_error(self, event: Error) -> None:
        """Called when an error occurs."""
        _LOGGER.error("Error during transcription: %s", event.error)

    def get_text(self) -> str:
        """Get the current accumulated text."""
        return "\n".join(self.lines)


class MoonshineTranscriber:
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
        self.listener: Optional[TranscriptEventListener] = None

    async def get_and_clear_transcription(self) -> str:
        # Indicate we have no more data coming
        _LOGGER.debug("Stopping recognizer to finalize transcription")
        self.recognizer.stop()
        text = ""
        if not self.listener:
            raise Exception("No transcription listener found")
        text = self.listener.get_text()
        # Remove the current listener at the end of the transcription
        self.recognizer.remove_all_listeners()
        _LOGGER.debug("Got %s", text)
        return text

    async def start_transcription(self):
        if self.listener:
            _LOGGER.debug("Transcription already in progress, not starting new one")
            return
        self.sample_rate = None
        _LOGGER.debug("Starting new transcription")
        self.listener = AccumulatingListener()
        self.recognizer.start()
        _LOGGER.debug("Recognizer started, clearing listeners")
        self.recognizer.remove_all_listeners()
        _LOGGER.debug("Creating new listener for transcription")
        self.recognizer.add_listener(self.listener)
        _LOGGER.debug("Listener added")

    async def queue_chunk(self, raw_audio_data, sample_rate):
        """Queue a chunk for transcription"""
        if not self.listener:
            _LOGGER.debug("No transcription service on first chunk, starting")
            await self.start_transcription()
        if not self.sample_rate:
            self.sample_rate = sample_rate
        # raw_bytes: your bytes object or bytearray
        samples = np.frombuffer(raw_audio_data, dtype=np.int16)

        # normalize to [-1.0, 1.0)
        audio_data = samples.astype(np.float32) / 32768.0
        stream = self.recognizer._default_stream
        _LOGGER.debug(f"Adding chunk of size {len(audio_data)} at rate {sample_rate}")
        _LOGGER.debug(f"Listener is {self.listener} configured listeners are {stream._listeners}")
        add_result = self.recognizer.add_audio(audio_data, sample_rate)
        _LOGGER.debug(f"Chunk added, result: {add_result}?")
        _LOGGER.debug(f"Current stream time {stream._stream_time}, last updated at {stream._last_update_time}")
