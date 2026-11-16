from fastapi import FastAPI
from pydantic import BaseModel
from src.generator import generate_response

app = FastAPI()

class Query(BaseModel):
    prompt: str
    k: int = 3
    temperature: float = 0.6

@app.post("/generate")
def generate(q: Query):
    text = generate_response(q.prompt, top_k=q.k, temperature=q.temperature)
    return {"output": text}
