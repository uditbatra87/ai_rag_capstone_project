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