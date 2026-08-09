from openai import OpenAI
from pydantic import BaseModel, Field
from typing import List
import json
from settings import my_settings

# --------------------------------------------------------------------
# Client setup - use your existing settings (Vocareum key needs
# base_url pointed at their proxy, not OpenAI's default endpoint)
# --------------------------------------------------------------------
client = OpenAI(
    api_key=my_settings.OPEN_AI_KEY,
    base_url=my_settings.OPEN_AI_URL,
)


# --------------------------------------------------------------------
# The structure we want the LLM's answer validated against
# --------------------------------------------------------------------
class Answer(BaseModel):
    content: str = Field(description="The main answer content.")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0.")
    sources: List[str] = Field(description="List of sources or references used.")


# ======================================================================
# APPROACH 1: Prompt-based JSON mode
# We ASK the model (via the system prompt) to return JSON in a specific
# shape, then manually parse + validate it ourselves.
# Less reliable than Approach 2, since the model could still deviate
# from the requested shape even with response_format enabled.
# ======================================================================
message = [
    {
        "role": "system",
        "content": """Answer the user's question in JSON format with these keys:
         content, confidence (score between 0-1), sources (list of references).

Sample output:
{
  "content": "<the real response>",
  "confidence": 0.8,
  "sources": ["source1", "source2"]
}
"""
    },
    {"role": "user", "content": "How is the growth rate of India ?"}
]

response = client.chat.completions.create(
    model='gpt-3.5-turbo',
    messages=message,
    response_format={"type": "json_object"}  # tells the model to return valid JSON
)

output = response.choices[0].message.content
#print(output)

json_output = json.loads(output)          # turn the JSON string into a Python dict
ans = Answer.model_validate(json_output)  # validate/parse it into an Answer object
#print(ans)


# ======================================================================
# APPROACH 2: Tool/function calling
# We define a strict schema as a "tool" and FORCE the model to call it.
# This is generally more reliable than Approach 1 for structured output,
# since the model is constrained to fill in exactly these fields.
# ======================================================================
tool = [
    {
        "type": "function",
        "function": {
            "name": "answer_question",
            "description": "Provide a structured answer to the user's question",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The main answer content."
                    },
                    "confidence": {
                        "type": "number",
                        "description": "Confidence score between 0.0 and 1.0."
                    },
                    "sources": {
                        "type": "array",
                        "description": "List of sources or references used",
                        "items": {"type": "string"}
                    }
                },
                "required": ["content", "confidence", "sources"]
            }
        }
    }
]

message = [{"role": "user", "content": "How is the growth rate of India ?"}]

response = client.chat.completions.create(
    model='gpt-3.5-turbo',
    messages=message,
    tools=tool,
    tool_choice={"type": "function", "function": {"name": "answer_question"}}
    # forces the model to ALWAYS call this specific function
)

# response.choices[0].message.content is empty here -
# the model responded via a tool call instead of plain text
tool_call = response.choices[0].message.tool_calls[0]
output = tool_call.function.arguments   # this is a JSON string of the arguments

print(output)

json_output = json.loads(output)
ans = Answer.model_validate(json_output)
#print(ans.content)