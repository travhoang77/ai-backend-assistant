from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from rag import add_pdf
from pydantic import BaseModel
from agent import run_agent
import json
import shutil
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all for now
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request body
class AskRequest(BaseModel):
    question: str

# Response endpoint
@app.post("/ask")
def ask_ai(request: AskRequest):
    try:
        result = run_agent(request.question)

        # Try to parse JSON output
        try:
            parsed = json.loads(result)
            return {"success": True, "data": parsed}
        except:
            return {"success": True, "data": result}

    except Exception as e:
        return {"success": False, "error": str(e)}
    
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    try:
        file_path = f"uploads/{file.filename}"
        os.makedirs("uploads", exist_ok=True)

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 🔥 Run indexing in background
        background_tasks.add_task(add_pdf, file_path)

        return {
            "success": True,
            "message": f"{file.filename} uploaded. Processing... wait a few seconds before querying."
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }