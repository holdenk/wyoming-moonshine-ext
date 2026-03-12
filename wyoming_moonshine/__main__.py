#!/usr/bin/env python3
"""Wyoming server for moonshine speech-to-text."""

import argparse
import asyncio
import logging
from functools import partial

from wyoming.info import AsrModel, AsrProgram, Attribution, Info
from wyoming.server import AsyncServer, AsyncTcpServer

from . import __version__
from .const import AUTO_MODEL
from .dispatch_handler import DispatchEventHandler
from .models import ModelLoader

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
        "--model",
        default=AUTO_MODEL,
        help="Name of model to use (or auto)",
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
        help="Default language to set for transcription (default: en)",
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

    if args.model == AUTO_MODEL:
        args.model = None

    model_name = "moonshine"

    wyoming_info = Info(
        asr=[
            AsrProgram(
                name="moonshine",
                description="Moonshine speech-to-text",
                attribution=Attribution(
                    name="Useful Sensors",
                    url="https://github.com/useful-sensors/moonshine",
                ),
                installed=True,
                version=__version__,
                models=[
                    AsrModel(
                        name=model_name,
                        description=model_name,
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

    loader = ModelLoader(
        preferred_language=args.language,
        download_dir=args.download_dir,
        model=args.model,
    )

    # Pre-load model
    _LOGGER.debug("Pre-loading transcriber")
    await loader.load_transcriber()

    server = AsyncServer.from_uri(args.uri)
    handler_factory = partial(DispatchEventHandler, wyoming_info, loader)

    if args.zeroconf:
        if not isinstance(server, AsyncTcpServer):
            raise ValueError("Zeroconf requires tcp:// uri")

        tcp_server: AsyncTcpServer = server

        # Bind port first so it's ready for connections
        await tcp_server.start(handler_factory)

        # Now register zeroconf — HA will discover and connect
        from wyoming.zeroconf import HomeAssistantZeroconf

        hass_zeroconf = HomeAssistantZeroconf(
            name=args.zeroconf, port=tcp_server.port, host=tcp_server.host
        )
        await hass_zeroconf.register_server()
        _LOGGER.info("Ready (zeroconf)")

        # Block on the already-started server
        assert tcp_server._server is not None  # pylint: disable=protected-access
        await tcp_server._server.serve_forever()  # pylint: disable=protected-access
    else:
        _LOGGER.info("Ready")
        await server.run(handler_factory)


# -----------------------------------------------------------------------------


def run() -> None:
    """Run the server."""
    asyncio.run(main())


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        pass
