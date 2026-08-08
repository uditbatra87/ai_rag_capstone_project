from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import uvicorn
from pydantic import BaseModel
import logging, json, time
import asyncio

app = FastAPI(title="My First API Service")

class Question(BaseModel):
    question: str

async def get_streaming_answer(question: Question):
    answer = "This is just a test , so RELAX"
    for word in answer.split(" "):
        yield word + " "
        await asyncio.sleep(1)

@app.post("/streaming-response")
async def get_streaming_response(question: Question):
    return StreamingResponse(
        get_streaming_answer(question),
        media_type='text/plain'
    )

if __name__ == "__main__":
    uvicorn.run(
        "fastapi_streaming_response:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )