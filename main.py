# main.py
import asyncio
import sys

# ✅ Fix event loop issue on Windows
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import uvicorn

if __name__ == "__main__":
    # ✅ THIS GUARD is required on Windows when using multiprocessing / reload
    uvicorn.run("agentZeroQdrant:app", host="127.0.0.1", port=8000, reload=True)
