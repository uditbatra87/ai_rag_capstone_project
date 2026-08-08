from pydantic import BaseModel

class input_structure(BaseModel):
    query : str


class output_structure(BaseModel):
    model : str
    prompt_tokens : int
    completion_tokens : int
    total_tokens : int
    llm_response : str
    time_taken : float