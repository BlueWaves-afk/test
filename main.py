from __future__ import annotations

import base64
import json
import os

import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Image QA ──────────────────────────────────────────────────────────────────

class ImageQARequest(BaseModel):
    image_base64: str
    question: str


class ImageQAResponse(BaseModel):
    answer: str


@app.post("/answer-image", response_model=ImageQAResponse)
def answer_image(req: ImageQARequest) -> ImageQAResponse:
    try:
        image_bytes = base64.b64decode(req.image_base64, validate=True)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 image data")

    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content([
        {
            "mime_type": "image/png",
            "data": image_bytes,
        },
        (
            f"{req.question}\n\n"
            "Reply with only the raw answer value — no units, no extra text, "
            "no explanation. For numbers, use digits only (e.g. '4089.35')."
        ),
    ])

    return ImageQAResponse(answer=response.text.strip())


# ── Invoice Extract ───────────────────────────────────────────────────────────

INVOICE_PROMPT = """Extract these 6 fields from the invoice text below and return valid JSON with no other text:
- invoice_no: string (null if not found)
- date: string in YYYY-MM-DD format (null if not found)
- vendor: string — the seller/vendor name (null if not found)
- amount: number — subtotal BEFORE tax (null if not found)
- tax: number — tax amount only, not grand total (null if not found)
- currency: string e.g. INR, USD (null if not found)

Return ONLY the JSON object, no markdown, no explanation.

Invoice text:
"""


class ExtractRequest(BaseModel):
    invoice_text: str


@app.post("/extract")
def extract(req: ExtractRequest) -> dict:
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(INVOICE_PROMPT + req.invoice_text)
    text = response.text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return json.loads(text)
