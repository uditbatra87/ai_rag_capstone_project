from fastapi import FastAPI

app = FastAPI(title="My First API Service")


@app.get("/first_endpoint")
async def first_method():
    return { "response" : "Hello World, How are you ?"}


