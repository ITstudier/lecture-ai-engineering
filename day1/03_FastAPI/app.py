import os
import torch
from transformers import pipeline
import time
import traceback
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import uvicorn
import nest_asyncio
from pyngrok import ngrok

# --- 設定 ---
MODEL_NAME = "google/gemma-2-2b-jpn-it"
print(f"\u30e2\u30c7\u30eb\u540d\u3092\u8a2d\u5b9a: {MODEL_NAME}")

# --- モデル設定クラス ---
class Config:
    def __init__(self, model_name=MODEL_NAME):
        self.MODEL_NAME = model_name

config = Config(MODEL_NAME)

# --- FastAPIアプリケーション定義 ---
app = FastAPI(
    title="\u30ed\u30fc\u30ab\u30ebLLM API\u30b5\u30fc\u30d3\u30b9",
    description="transformers\u30e2\u30c7\u30eb\u3092\u4f7f\u7528\u3057\u305f\u30c6\u30ad\u30b9\u30c8\u751f\u6210\u306e\u305f\u3081\u306eAPI",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- データモデル定義 ---
class Message(BaseModel):
    role: str
    content: str

class SimpleGenerationRequest(BaseModel):
    prompt: str
    max_new_tokens: Optional[int] = 512
    do_sample: Optional[bool] = True
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9

class GenerationResponse(BaseModel):
    generated_text: str
    response_time: float

# --- モデル関連の関数 ---
model = None

def load_model():
    global model
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\u4f7f\u7528\u30c7\u30d0\u30a4\u30b9: {device}")
        pipe = pipeline(
            "text-generation",
            model=config.MODEL_NAME,
            model_kwargs={
                "torch_dtype": torch.bfloat16,
                "device_map": "auto" if device == "cuda" else None
            }
        )
        model = pipe
        print(f"\u2705 \u30e2\u30c7\u30eb '{config.MODEL_NAME}' \u306e\u8aad\u307f\u8fbc\u307f\u306b\u6210\u529f\u3057\u307e\u3057\u305f")
        return pipe
    except Exception as e:
        print(f"\u274c \u30e2\u30c7\u30eb\u306e\u8aad\u307f\u8fbc\u307f\u5931\u6557: {e}")
        traceback.print_exc()
        return None

def extract_assistant_response(outputs, user_prompt):
    assistant_response = ""
    try:
        if outputs and isinstance(outputs, list) and len(outputs) > 0 and outputs[0].get("generated_text"):
            generated_output = outputs[0]["generated_text"]
            if isinstance(generated_output, str):
                if user_prompt:
                    prompt_end_index = generated_output.find(user_prompt)
                    if prompt_end_index != -1:
                        prompt_end_pos = prompt_end_index + len(user_prompt)
                        assistant_response = generated_output[prompt_end_pos:].strip()
                    else:
                        assistant_response = generated_output
                else:
                    assistant_response = generated_output
            elif isinstance(generated_output, list):
                last_message = generated_output[-1]
                if isinstance(last_message, dict) and last_message.get("role") == "assistant":
                    assistant_response = last_message.get("content", "").strip()
                else:
                    assistant_response = str(last_message).strip()
            else:
                assistant_response = str(generated_output).strip()
    except Exception as e:
        print(f"\u5fdc\u7b54\u306e\u62bd\u51fa\u4e2d\u306b\u30a8\u30e9\u30fc\u304c\u767a\u751f\u3057\u307e\u3057\u305f: {e}")
        traceback.print_exc()
        assistant_response = "\u5fdc\u7b54\u306e\u62bd\u51fa\u306b\u5931\u6557\u3057\u307e\u3057\u305f\u3002"

    if not assistant_response:
        print("\u8b66\u544a: \u30a2\u30b7\u30b9\u30bf\u30f3\u30c8\u306e\u5fdc\u7b54\u3092\u62bd\u51fa\u3067\u304d\u307e\u305b\u3093\u3067\u3057\u305f\u3002\u5b8c\u5168\u306a\u51fa\u529b:", outputs)
        assistant_response = "\u5fdc\u7b54\u3092\u751f\u6210\u3067\u304d\u307e\u305b\u3093\u3067\u3057\u305f\u3002"

    return assistant_response

@app.on_event("startup")
async def startup_event():
    print("\ud83d\udce6 \u30e2\u30c7\u30eb\u3092\u8d77\u52d5\u6642\u306b\u521d\u671f\u5316\u4e2d...")
    load_model_task()
    if model is None:
        print("\u26a0\ufe0f \u8d77\u52d5\u6642\u306b\u30e2\u30c7\u30eb\u521d\u671f\u5316\u306b\u5931\u6557\u3057\u307e\u3057\u305f")
    else:
        print("\u2705 \u8d77\u52d5\u6642\u306b\u30e2\u30c7\u30eb\u306e\u521d\u671f\u5316\u304c\u5b8c\u4e86\u3057\u307e\u3057\u305f")

@app.get("/")
async def root():
    return {"status": "ok", "message": "Local LLM API is runnning"}

@app.get("/health")
async def health_check():
    global model
    if model is None:
        return {"status": "error", "message": "No model loaded"}
    return {"status": "ok", "model": config.MODEL_NAME}

@app.post("/generate", response_model=GenerationResponse)
async def generate_simple(request: SimpleGenerationRequest):
    global model
    if model is None:
        print("generate\u30a8\u30f3\u30c9\u30dd\u30a4\u30f3\u30c8: \u30e2\u30c7\u30eb\u304c\u8aad\u307f\u8fbc\u307e\u308c\u3066\u3044\u307e\u305b\u3093\u3002\u8aad\u307f\u8fbc\u307f\u3092\u8a66\u307f\u307e\u3059...")
        load_model_task()
        if model is None:
            raise HTTPException(status_code=503, detail="\u30e2\u30c7\u30eb\u304c\u5229\u7528\u3067\u304d\u307e\u305b\u3093\u3002")
    try:
        start_time = time.time()
        outputs = model(
            request.prompt,
            max_new_tokens=request.max_new_tokens,
            do_sample=request.do_sample,
            temperature=request.temperature,
            top_p=request.top_p,
        )
        assistant_response = extract_assistant_response(outputs, request.prompt)
        response_time = time.time() - start_time
        return GenerationResponse(
            generated_text=assistant_response,
            response_time=response_time
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"\u5fdc\u7b54\u306e\u751f\u6210\u4e2d\u306b\u30a8\u30e9\u30fc\u304c\u767a\u751f\u3057\u307e\u3057\u305f: {str(e)}")

def load_model_task():
    global model
    print("load_model_task: \u30e2\u30c7\u30eb\u306e\u8aad\u307f\u8fbc\u307f\u3092\u958b\u59cb...")
    loaded_pipe = load_model()
    if loaded_pipe:
        model = loaded_pipe
        print("load_model_task: \u30e2\u30c7\u30eb\u306e\u8aad\u307f\u8fbc\u307f\u304c\u5b8c\u4e86\u3057\u307e\u3057\u305f")
    else:
        print("load_model_task: \u30e2\u30c7\u30eb\u306e\u8aad\u307f\u8fbc\u307f\u306b\u5931\u6557\u3057\u307e\u3057\u305f")

print("FastAPI\u30a8\u30f3\u30c9\u30dd\u30a4\u30f3\u30c8\u3092\u5b9a\u7fa9\u3057\u307e\u3057\u305f\u3002")

def run_with_ngrok(port=8501):
    nest_asyncio.apply()
    ngrok_token = os.environ.get("NGROK_TOKEN")
    if not ngrok_token:
        print("Ngrok\u8a8d\u8a3c\u30c8\u30fc\u30af\u30f3\u304c'NGROK_TOKEN'\u74b0\u5883\u5909\u6570\u306b\u8a2d\u5b9a\u3055\u308c\u3066\u3044\u307e\u305b\u3093\u3002")
        return
    try:
        ngrok.set_auth_token(ngrok_token)
        tunnels = ngrok.get_tunnels()
        for tunnel in tunnels:
            ngrok.disconnect(tunnel.public_url)
        print(f"\u30dd\u30fc\u30c8{port}\u306b\u65b0\u3057\u3044ngrok\u30c8\u30f3\u30cd\u30eb\u3092\u958b\u3044\u3066\u3044\u307e\u3059...")
        public_url = ngrok.connect(port).public_url
        print("-" * 69)
        print(f"\u2705 \u516c\u958bURL:   {public_url}")
        print(f"\ud83d\udcd6 API\u30c9\u30ad\u30e5\u30e1\u30f3\u30c8: {public_url}/docs")
        print("-" * 69)
        uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
    except Exception as e:
        print(f"\n ngrok\u307e\u305f\u306fUvicorn\u306e\u8d77\u52d5\u4e2d\u306b\u30a8\u30e9\u30fc: {e}")
        traceback.print_exc()
        try:
            tunnels = ngrok.get_tunnels()
            for tunnel in tunnels:
                ngrok.disconnect(tunnel.public_url)
        except Exception as ne:
            print(f"ngrok\u30af\u30ea\u30fc\u30f3\u30a2\u30c3\u30d7\u4e2d\u306b\u30a8\u30e9\u30fc: {ne}")

if __name__ == "__main__":
    run_with_ngrok(port=8501)
    print("\n\u30b5\u30fc\u30d0\u30fc\u30d7\u30ed\u30bb\u30b9\u304c\u7d42\u4e86\u3057\u307e\u3057\u305f\u3002")
