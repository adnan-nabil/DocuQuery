from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional


from src.brain import sys 


app = FastAPI(title="CSN RAG Bot Backend")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str
    thread_id: Optional[str] = "default_session"

class ChatResponse(BaseModel):
    answer: str
    history: List


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        
        config = {"configurable": {"thread_id": request.thread_id}}
        result = sys.invoke({
            "question": request.question}, 
            config=config
        )
        
        
        return {
            "answer": result["answer"],
            "history": result["history"]
        }
    except Exception as e:
        
        print(f"Error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def read_root():
    return {"status": "Online", "message": "CSN RAG Bot is running"}