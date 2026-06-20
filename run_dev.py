import asyncio

import uvicorn


if hasattr(asyncio, "WindowsSelectorEventLoopPolicy"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=8010,
        loop="asyncio",
        http="h11",
        backlog=1,
        reload=False,
    )