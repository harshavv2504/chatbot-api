from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
import os

# Use environment variable for API key
API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=API_KEY)

app = FastAPI()

# Store conversation history per session (in memory)
conversations = {}

class ChatRequest(BaseModel):
    session_id: str
    message: str

@app.post("/chat")
def chat(req: ChatRequest):
    if req.session_id not in conversations:
        conversations[req.session_id] = []

    conversations[req.session_id].append({"role": "user", "content": req.message})

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=conversations[req.session_id],
        temperature=0.3,
    )

    reply = response.choices[0].message.content
    conversations[req.session_id].append({"role": "assistant", "content": reply})

    return {"reply": reply}
