from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from rag import add_pdf
from pydantic import BaseModel
from agent import run_agent
from memory import init_db, get_history, save_message
import json
import shutil
import os
from rag import save_memory  # or memory_vector

# -------- INIT --------
init_db()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------- REQUEST MODEL --------
class AskRequest(BaseModel):
    question: str
    user_id: str = "default"


# -------- ASK ENDPOINT --------
@app.post("/ask")
def ask(req: AskRequest):
    user_id = req.user_id
    question = req.question

    history = get_history(user_id)
    print("📜 HISTORY:", history)

    result = run_agent(question, history, user_id)
    
    try:
        parsed = json.loads(result)
        clean_answer = parsed.get("answer", result)
    except:
        clean_answer = result

    save_memory(question, user_id)
    save_memory(clean_answer, user_id)
    
    save_message(user_id, "assistant", clean_answer)

    return {"success": True, "data": result}

# -------- UPLOAD --------
@app.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...), background_tasks: BackgroundTasks = None
):
    try:
        file_path = f"uploads/{file.filename}"
        os.makedirs("uploads", exist_ok=True)

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        background_tasks.add_task(add_pdf, file_path)

        return {"success": True, "message": f"{file.filename} uploaded. Processing..."}

    except Exception as e:
        return {"success": False, "error": str(e)}
