from openai import AsyncOpenAI
import asyncio
from settings import my_settings
import json

import warnings
warnings.filterwarnings("ignore")

client = AsyncOpenAI(
        api_key=my_settings.OPEN_AI_KEY,
        base_url=my_settings.OPEN_AI_URL,
    )

async def call_open_ai(user_query : str) -> dict :
    response = await client.chat.completions.create(
    model='gpt-4o-mini',
    messages=[
                {"role" : "system","content" : "You are a helpful assistant. Respond to the user in a polite way."},
                {"role" : "user","content" : user_query}
            ]
    )   

    final_response = {
        "model" : response.model,
        "prompt_tokens" : response.usage.prompt_tokens,
        "completion_tokens" : response.usage.completion_tokens,
        "total_tokens" : response.usage.total_tokens,
        "llm_response" : response.choices[0].message.content
    }

    return final_response

result = asyncio.run(call_open_ai("How is the weather today ?"))
print(json.dumps(result, indent=2))