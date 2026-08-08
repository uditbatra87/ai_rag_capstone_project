import instructor
from openai import OpenAI
from pydantic import BaseModel,Field
import os
from dotenv import load_dotenv

import warnings
warnings.filterwarnings("ignore")


class User(BaseModel):
    name : str = Field(min_length=3,max_length=100)
    age : int = Field(gt=0,min=1,max=125,description="Age of the person")
    occupation : str = Field(default=None)
    hobbies : list[str] = Field(default=[])
    email : str = Field(default=None)
    phone : str = Field(default=None)


# Load the variables
load_dotenv('.env.local')

# Access them using os.getenv or os.environ
OPEN_AI_KEY = os.getenv("OPEN_AI_KEY")
OPEN_AI_URL = os.getenv("OPEN_AI_URL")

client = instructor.from_openai(
    OpenAI(
        api_key=OPEN_AI_KEY,
        base_url=OPEN_AI_URL

    )
)

input = """Hello, My name is Udit and i am 39 years old. I like reading and sleeping. I work as an IT Engineer. you can reach out to me uditbatra87@test.org
or call me on 9158650792"""


user:User = client.chat.completions.create(
    model='gpt-4o-mini',
    response_model=User,
    messages=[
        {"role" : "system","content" : "Extract the User information from text"},
        {"role" : "user","content" : input}
        ]
)

print(user.model_dump())
