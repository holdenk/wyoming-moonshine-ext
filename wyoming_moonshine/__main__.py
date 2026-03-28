#!/usr/bin/env python3
import argparse
import asyncio
import logging
import platform
from functools import partial

from wyoming.info import AsrModel, AsrProgram, Attribution, Info
from wyoming.server import AsyncServer, AsyncTcpServer

from . import __version__
from .const import AUTO_MODEL
from .dispatch_handler import DispatchEventHandler
from .moonshine_handler import MoonshineTranscriber

_LOGGER = logging.getLogger(__name__)


async def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--uri", required=True, help="unix:// or tcp://")
    parser.add_argument(
        "--zeroconf",
        nargs="?",
        const="moonshine",
        help="Enable discovery over zeroconf with optional name (default: moonshine)",
    )
    parser.add_argument(
        "--model", default=AUTO_MODEL, help=f"Name of model to use (or {AUTO_MODEL})"
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        action="append",
        help="Data directory to check for downloaded models",
    )
    parser.add_argument(
        "--download-dir",
        help="Directory to download models into (default: first data dir)",
    )
    parser.add_argument(
        "--language",
        default="en",
        help=f"Default language to set for transcription (default: en)",
    )
    parser.add_argument("--debug", action="store_true", help="Log DEBUG messages")
    parser.add_argument(
        "--log-format", default=logging.BASIC_FORMAT, help="Format for log messages"
    )
    parser.add_argument(
        "--version",
        action="version",
        version=__version__,
        help="Print version and exit",
    )

    args = parser.parse_args()

    if not args.download_dir:
        args.download_dir = args.data_dir[0]

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO, format=args.log_format
    )
    _LOGGER.debug(args)

    model = args.model if args.model != AUTO_MODEL else None

    wyoming_info = Info(
        asr=[
            AsrProgram(
                name="moonshine",
                description="moonshine",
                attribution=Attribution(
                    name="Useful Sensors",
                    url="https://github.com/useful-sensors/moonshine",
                ),
                installed=True,
                version=__version__,
                models=[
                    AsrModel(
                        name="moonshine",
                        description="moonshine",
                        attribution=Attribution(
                            name="Useful Sensors",
                            url="https://github.com/useful-sensors/moonshine",
                        ),
                        installed=True,
                        languages=["en"],
                        version=__version__,
                    )
                ],
            )
        ],
    )

    # Load model
    _LOGGER.debug("Loading transcriber")
    model = args.model
    if not model:
        machine = platform.machine().lower()
        is_arm = ("arm" in machine) or ("aarch" in machine)
        model = "small" if is_arm else "medium"
        _LOGGER.debug("Auto-Selected model: %s", model)
    else:
        _LOGGER.debug("Selected model: %s", model)
    _transcriber = MoonshineTranscriber(
        model_id=model, language=args.language, cache_dir=args.download_dir
    )

    lang = args.language or "en"

    handler_factory = partial(
        DispatchEventHandler,
        wyoming_info,
        _transcriber,
    )

    server = AsyncServer.from_uri(args.uri)

    if args.zeroconf:
        if not isinstance(server, AsyncTcpServer):
            raise ValueError("Zeroconf requires tcp:// uri")

        from wyoming.zeroconf import HomeAssistantZeroconf

        tcp_server: AsyncTcpServer = server

        # Start server first so the port is bound before zeroconf advertises it
        await tcp_server.start(handler_factory)

        hass_zeroconf = HomeAssistantZeroconf(
            name=args.zeroconf, port=tcp_server.port, host=tcp_server.host
        )
        await hass_zeroconf.register_server()
        _LOGGER.debug("Zeroconf discovery enabled")

        _LOGGER.info("Ready")
        assert tcp_server._server is not None
        await tcp_server._server.serve_forever()
    else:
        _LOGGER.info("Ready")
        await server.run(handler_factory)


# -----------------------------------------------------------------------------


def run() -> None:
    asyncio.run(main())


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        pass
