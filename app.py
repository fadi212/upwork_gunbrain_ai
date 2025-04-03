from fastapi import FastAPI, Form, HTTPException
from enum import Enum
import uuid
from src.pipeline.query import internal_query_index, external_query_index

class PromptType(str, Enum):
    general = "General"

app = FastAPI()

@app.get("/new-session/")
async def new_session():
    """
    Generate a new session ID to isolate chat memory.
    """
    return {"session_id": str(uuid.uuid4())}

@app.post("/internal-query/")
async def internal_query(
    prompt_type: PromptType,
    query: str = Form(...),
    session_id: str = Form(...),
    selected_model: str = Form("gpt-4o-mini")  
):
    try:
        response, citations, prompt_tokens, completion_tokens, total_tokens, chat_history, model_used = await internal_query_index(
            prompt_type=prompt_type, query=query, session_id=session_id, selected_model=selected_model
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {
        "response": response,
        "citations": citations,
        "history": chat_history,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "model_used": model_used
    }

@app.post("/external-query/")
async def external_query(
    prompt_type: PromptType,
    query: str = Form(...),
    session_id: str = Form(...),
    selected_model: str = Form("gpt-4o-mini")
):
    try:
        response, model_used, prompt_tokens, completion_tokens, total_tokens, history = await external_query_index(
            prompt_type=prompt_type, query=query, session_id=session_id, selected_model=selected_model
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {
        "response": response,
        "history": history,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "model_used": model_used
    }
