from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
import logging, json, time

app = FastAPI(title="My First API Service")

class User(BaseModel):
    username : str
    emailid : str
    age : int

# ==============================================================================
# HOW PYTHON LOG LEVELS FILTER MESSAGES
# ==============================================================================
# Rule: Setting a level acts as a MINIMUM threshold.
#       It captures that specific level and anything HIGHER (more severe).
#       It completely BLOCKS anything LOWER (less severe).
#
# Severity Scale & Behavior:
# -------------------------
# [50] CRITICAL --> Captured if level is CRITICAL, ERROR, WARNING, INFO, or DEBUG
# [40] ERROR    --> Captured if level is ERROR, WARNING, INFO, or DEBUG
# [30] WARNING  --> Captured if level is WARNING, INFO, or DEBUG (Default Level)
# ------------------------------------------------------------------------------
# [20] INFO     --> Captured ONLY if level is INFO or DEBUG
# [10] DEBUG    --> Captured ONLY if level is DEBUG
# ==============================================================================
# EXAMPLES:
# ------------------------------------------------------------------------------
# 1. If you set level to WARNING:
#    -> Captured: WARNING, ERROR, CRITICAL
#    -> Hidden:   INFO, DEBUG (Too low severity!)
#
# 2. If you set level to INFO:
#    -> Captured: INFO, WARNING, ERROR, CRITICAL
#    -> Hidden:   DEBUG (Too low severity!)
#
# 3. If you want to see absolutely EVERYTHING, set the level to DEBUG.
# ==============================================================================


class JsonLogger(logging.Formatter):
    def format(self,record):
        log_entry = {
            'timestamp' : time.strftime('%Y-%m-%d %H:%M:%S'),
            'level' : record.levelname,
            'message' : record.getMessage()
        }

        return json.dumps(log_entry)


# log = logging.getLogger('my-sample-webservice')
# log.setLevel(logging.INFO)

# # Adding the logs to a File
# fileLogger=logging.FileHandler('webservice.log',mode='a')
# fileLogger.setFormatter(JsonLogger())
# log.addHandler(fileLogger)

# # Adding the logs to a Stream
# streamLogger=logging.StreamHandler()
# streamLogger.setFormatter(JsonLogger())
# log.addHandler(streamLogger)



log = logging.getLogger('my-sample-webservice')
log.setLevel(logging.INFO)

if not log.handlers:  # <-- only add handlers if none exist yet
    fileLogger = logging.FileHandler('webservice.log', mode='a')
    fileLogger.setFormatter(JsonLogger())
    log.addHandler(fileLogger)

    streamLogger = logging.StreamHandler()
    streamLogger.setFormatter(JsonLogger())
    log.addHandler(streamLogger)


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
