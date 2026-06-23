import asyncio
import sys

import uvicorn


APP = "app.main:app"
HOST = "127.0.0.1"
PORT = 8010


def main() -> None:
    config = uvicorn.Config(
        APP,
        host=HOST,
        port=PORT,
        loop="asyncio",
        http="h11",
        backlog=1,
        reload=False,
        log_level="info",
    )
    server = uvicorn.Server(config)

    if sys.platform.startswith("win"):
        loop = asyncio.SelectorEventLoop()
        asyncio.set_event_loop(loop)

        try:
            loop.run_until_complete(server.serve())
        finally:
            loop.close()
    else:
        uvicorn.run(
            APP,
            host=HOST,
            port=PORT,
            loop="asyncio",
            http="h11",
            backlog=1,
            reload=False,
        )


if __name__ == "__main__":
    main()