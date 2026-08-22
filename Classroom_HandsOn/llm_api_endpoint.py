from fastapi import FastAPI
from openai import AsyncOpenAI
import asyncio, time
from settings import my_settings
from pydantic import BaseModel
from schema import input_structure,output_structure
import warnings
warnings.filterwarnings("ignore")


client = AsyncOpenAI(
    api_key=my_settings.OPEN_AI_KEY,
    base_url=my_settings.OPEN_AI_URL,
)

app = FastAPI()


@app.post('/chat', response_model=output_structure)
async def call_open_ai(payload: input_structure) -> output_structure:
    start_time = time.perf_counter()

    for attempt in range(my_settings.OPEN_AI_RETRIES):  # 0, 1, 2
        try:
            response = await client.chat.completions.create(
                model='gpt-4o-mini',
                timeout=my_settings.OPEN_AI_TIMEOUT,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant. Respond to the user in a polite way."},
                    {"role": "user", "content": payload.query}
                ]
            )

            end_time = time.perf_counter()

            final_response = {
                "model": response.model,
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
                "llm_response": response.choices[0].message.content,
                "time_taken": end_time - start_time
            }

            return output_structure(**final_response)

        except Exception as e:
            if attempt == my_settings.OPEN_AI_RETRIES - 1:
                raise e
            await asyncio.sleep(2)

# uvicorn llm_api_endpoint:app --reload --port 8000
# ==============================================================================
# EXPLANATION: output_structure(**final_response)
# ==============================================================================
#
# This line does two things at once: dictionary unpacking + Pydantic model
# construction.
#
# ------------------------------------------------------------------------------
# STEP 1: What final_response is
# ------------------------------------------------------------------------------
# final_response = {
#     "model": response.model,
#     "prompt_tokens": response.usage.prompt_tokens,
#     "completion_tokens": response.usage.completion_tokens,
#     "total_tokens": response.usage.total_tokens,
#     "llm_response": response.choices[0].message.content,
#     "time_taken": end_time - start_time
# }
#
# This is just a plain Python dictionary - key/value pairs. It has NO
# relationship to the output_structure class yet. It's just data sitting
# in a dict.
#
# ------------------------------------------------------------------------------
# STEP 2: What output_structure(...) normally expects
# ------------------------------------------------------------------------------
# class output_structure(BaseModel):
#     model: str
#     prompt_tokens: int
#     completion_tokens: int
#     total_tokens: int
#     llm_response: str
#     time_taken: float
#
# To build this the "long way," you'd write out every argument by name:
#
#   output_structure(
#       model=response.model,
#       prompt_tokens=response.usage.prompt_tokens,
#       completion_tokens=response.usage.completion_tokens,
#       total_tokens=response.usage.total_tokens,
#       llm_response=response.choices[0].message.content,
#       time_taken=end_time - start_time
#   )
#
# Verbose - and we already built a dict with the exact same keys.
#
# ------------------------------------------------------------------------------
# STEP 3: What ** actually does - dictionary unpacking
# ------------------------------------------------------------------------------
# The ** operator, applied to a dict inside a function/constructor call,
# UNPACKS the dictionary into keyword arguments automatically.
#
#   final_response = {"model": "gpt-4o-mini", "prompt_tokens": 10}
#
#   output_structure(**final_response)
#
#   # Python expands this to exactly:
#   output_structure(model="gpt-4o-mini", prompt_tokens=10)
#
# So **final_response takes every key in the dict and turns it into
# key=value syntax, matching each dict key to the corresponding
# parameter name in output_structure.__init__.
#
# ------------------------------------------------------------------------------
# STEP 4: Why this works specifically here
# ------------------------------------------------------------------------------
# This only works because the dict's keys EXACTLY MATCH the field names
# in output_structure:
#
#   Dict key                 Model field
#   -----------------------  --------------------------
#   "model"                  model: str
#   "prompt_tokens"          prompt_tokens: int
#   "completion_tokens"      completion_tokens: int
#   "total_tokens"           total_tokens: int
#   "llm_response"           llm_response: str
#   "time_taken"             time_taken: float
#
# Since every key lines up with a real field name, **final_response
# successfully fills in all six required arguments.
#
# ------------------------------------------------------------------------------
# STEP 5: What Pydantic then does with those values
# ------------------------------------------------------------------------------
# Once output_structure(**final_response) is called, Pydantic:
#   1. Receives the keyword arguments (model=..., prompt_tokens=..., etc.)
#   2. VALIDATES each value against its declared type
#      (e.g. checks prompt_tokens is really an int, time_taken is a float)
#   3. If valid -> builds and returns an actual output_structure OBJECT
#      (not a dict) with .model, .prompt_tokens, etc. as real attributes
#      accessible via dot notation
#   4. If invalid -> raises a ValidationError immediately
#
# ------------------------------------------------------------------------------
# What happens WITHOUT **
# ------------------------------------------------------------------------------
# output_structure(final_response)   # <-- no ** = this FAILS
#
# Pydantic models don't accept a single positional dict argument like that;
# they expect individual keyword arguments. That's the whole reason ** is
# needed - it converts "one dict" into "several keyword arguments."
#
# ------------------------------------------------------------------------------
# Why this matters for a FastAPI endpoint
# ------------------------------------------------------------------------------
# Function signature: -> output_structure
#
# FastAPI (via response_model=output_structure) expects an ACTUAL
# output_structure instance returned - not a raw dict - so it can:
#   - Validate the response shape before sending it
#   - Auto-generate accurate API docs in /docs
#   - Serialize it correctly to JSON in the HTTP response
#
# Returning the plain dict (final_response) would often still work, since
# FastAPI can coerce a matching dict into the response model automatically.
# But explicitly building output_structure(**final_response) makes the
# validation happen right here, in our own code, at the exact point the
# response is built - rather than relying on FastAPI to catch mismatches
# later. It's the more explicit, safer pattern.
# ==============================================================================