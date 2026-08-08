from openai import AsyncOpenAI
import asyncio,json,time
from settings import my_settings

import warnings
warnings.filterwarnings("ignore")

client = AsyncOpenAI(
        api_key=my_settings.OPEN_AI_KEY,
        base_url=my_settings.OPEN_AI_URL,
    )

async def call_open_ai(user_query : str) -> dict :

    for attempt in range(my_settings.OPEN_AI_RETRIES): # 0,1,2
        try:
            start_time = time.perf_counter()

            response = await client.chat.completions.create(
            model='gpt-4o-mini',
            timeout=my_settings.OPEN_AI_TIMEOUT,
            messages=[
                        {"role" : "system","content" : "You are a helpful assistant. Respond to the user in a polite way."},
                        {"role" : "user","content" : user_query}
                    ]
            )   

            end_time = time.perf_counter()

            final_response = {
                "model" : response.model,
                "prompt_tokens" : response.usage.prompt_tokens,
                "completion_tokens" : response.usage.completion_tokens,
                "total_tokens" : response.usage.total_tokens,
                "llm_response" : response.choices[0].message.content,
                "time_taken" : end_time - start_time
            }

            return final_response

        except Exception as e:
            end_time = time.perf_counter()
            time_taken = end_time - start_time
            if attempt == my_settings.OPEN_AI_RETRIES - 1:
                raise e
            await asyncio.sleep(2)

result = asyncio.run(call_open_ai("How is the weather today ?"))
print(json.dumps(result, indent=2))