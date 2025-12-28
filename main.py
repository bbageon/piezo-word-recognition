import torch
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from fastapi.middleware.cors import CORSMiddleware
import numpy as np


class SensorData(BaseModel):
    time: List[int]
    value1: List[int]
    value2: List[int]


# ============================================
# 1) LLaMA2 + LoRA 로드
# ============================================

BASE_MODEL_PATH = "/Users/bbageon/Desktop/projects/Finetuning/llama2_local"
LORA_MODEL_PATH = (
    "/Users/bbageon/Desktop/projects/Finetuning/motionQA_finetuned_lora_mps"
)

print("🚀 Loading tokenizer & model...")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, local_files_only=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH, local_files_only=True)
model = PeftModel.from_pretrained(model, LORA_MODEL_PATH)

# numeric token 확장된 tokenizer 반영 (이미 학습된 tokenizer와 동일해야 함)
# tokenizer.add_tokens([...])  <-- 여기서는 재정의 X (미리 확장된 tokenizer 사용)

device = "mps" if torch.backends.mps.is_available() else "cpu"
model = model.to(device).to(torch.float32)
model.eval()

print("✅ Model + LoRA Loaded Successfully!")


# ============================================
# 2) FastAPI 초기화
# ============================================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================
# 3) Request Body 정의
# ============================================
class SensorData(BaseModel):
    time: List[int]
    value1: List[int]
    value2: List[int]


# ============================================
# 4) Feature 기반 Prompt 생성
# ============================================
word_list = ["Hi", "Bye", "Thanks"]


def extract_features(x: List[int]) -> dict:
    arr = np.asarray(x, dtype=np.float32)

    # 기본 통계
    mean = float(arr.mean())
    std = float(arr.std())
    vmin = float(arr.min())
    vmax = float(arr.max())
    p2p = float(vmax - vmin)

    # 에너지(크기)
    energy = float(np.mean(arr * arr))

    # 1차 차분 기반 (변화량)
    diff = np.diff(arr)
    diff_mean = float(diff.mean()) if diff.size else 0.0
    diff_std = float(diff.std()) if diff.size else 0.0
    diff_abs_mean = float(np.mean(np.abs(diff))) if diff.size else 0.0

    # zero crossing(부호 변화) - DC 제거 후 계산
    centered = arr - mean
    zc = (
        int(
            np.sum((centered[:-1] >= 0) & (centered[1:] < 0))
            + np.sum((centered[:-1] < 0) & (centered[1:] >= 0))
        )
        if arr.size > 1
        else 0
    )

    return {
        "mean": mean,
        "std": std,
        "min": vmin,
        "max": vmax,
        "p2p": p2p,
        "energy": energy,
        "diff_mean": diff_mean,
        "diff_std": diff_std,
        "diff_abs_mean": diff_abs_mean,
        "zero_crossings": zc,
        "n": int(arr.size),
    }


VALID_WORDS = ["Hi", "Hello", "Bye"]


def normalize_prediction(text: str) -> str:
    t = text.strip().lower()

    if "hi" in t:
        return "Hi"
    if "hello" in t:
        return "Hello"
    if "bye" in t:
        return "Bye"

    # fallback (절대 None 안 나오게)
    return "Hi"


def make_llm_prompt(
    time_list: List[int], v1_list: List[int], v2_list: List[int]
) -> str:
    assert len(time_list) == len(v1_list) == len(v2_list)

    # duration(ms) 추정: 마지막 time 값(ESP32가 0~3000ms)
    duration_ms = int(time_list[-1]) if len(time_list) > 0 else 0
    n = len(time_list)

    f_left = extract_features(v1_list)
    f_right = extract_features(v2_list)

    choices = ", ".join(word_list)

    # ✅ LLM에 "숫자 600줄"을 넣지 말고, 특징만 넣는다
    prompt = f"""
    You are a classifier for piezo vibration signals recorded from two channels (left/right).

    Meta:
    - samples: {n}
    - duration_ms: {duration_ms}

    Left channel features:
    {f_left}

    Right channel features:
    {f_right}

    Task:
    Predict the spoken word among: {choices}.
    Answer with only ONE of the choices (no extra text).
    Answer:
    """.strip()

    return prompt


# ============================================
# 5) LLM 추론 함수
# ============================================
def run_llm(prompt: str) -> str:
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048,  # 특징 기반이면 2k도 충분
    ).to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
            temperature=0.0,
        )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    last_line = decoded.strip().split("\n")[-1]
    return normalize_prediction(last_line)
    # return last_line.strip()


# ============================================
# 6) Predict API
# ============================================
@app.post("/predict_llm")
def predict_llm(data: SensorData):

    # 1) 프롬프트 생성
    prompt = make_llm_prompt(data.time, data.value1, data.value2)

    # 2) 모델 예측
    prediction = run_llm(prompt)

    return {"prediction": prediction, "prompt_used": prompt}


# ============================================
# 7) Test Root
# ============================================
@app.get("/")
def root():
    return {"message": "Piezo LLM Recognition API Running!"}
