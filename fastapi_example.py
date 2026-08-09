from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
from custom_logging import setup_logger


app = FastAPI(title="My First API Service")

class User(BaseModel):
    username : str
    emailid : str
    age : int


log = setup_logger()


@app.get("/first_endpoint")
async def first_method():
    return { "response" : "Hello Udit, How are you ?"}

@app.post("/createuser")
async def register_user(user_details : User):
    log.info(f"User details received for {user_details}")
    log.warning(f"User details warning for {user_details}")
    log.error(f"User detailds error for {user_details}")

    return (f"User {user_details.username} is created Successfully.")



# to run from command line 
# Run : uvicorn fastapi_example:app --reload --port 8000

if __name__ == "__main__":
    uvicorn.run(
        "fastapi_example:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
