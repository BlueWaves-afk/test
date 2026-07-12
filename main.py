from __future__ import annotations

import base64
import json
import os

import anthropic
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


# ── Image QA ──────────────────────────────────────────────────────────────────

class ImageQARequest(BaseModel):
    image_base64: str
    question: str


class ImageQAResponse(BaseModel):
    answer: str


@app.post("/answer-image", response_model=ImageQAResponse)
def answer_image(req: ImageQARequest) -> ImageQAResponse:
    try:
        base64.b64decode(req.image_base64, validate=True)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 image data")

    message = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=256,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": req.image_base64,
                        },
                    },
                    {
                        "type": "text",
                        "text": (
                            f"{req.question}\n\n"
                            "Reply with only the raw answer value — no units, no extra text, "
                            "no explanation. For numbers, use digits only (e.g. '4089.35')."
                        ),
                    },
                ],
            }
        ],
    )

    return ImageQAResponse(answer=message.content[0].text.strip())


# ── Invoice Extract ───────────────────────────────────────────────────────────

INVOICE_SYSTEM = """You are an invoice data extractor. Given raw invoice text, extract exactly these 6 fields and return valid JSON with no other text:
- invoice_no: string (null if not found)
- date: string in YYYY-MM-DD format (null if not found)
- vendor: string — the seller/vendor name (null if not found)
- amount: number — subtotal BEFORE tax (null if not found)
- tax: number — tax amount only, not grand total (null if not found)
- currency: string e.g. INR, USD (null if not found)

Return ONLY the JSON object, no markdown, no explanation."""


class ExtractRequest(BaseModel):
    invoice_text: str


@app.post("/extract")
def extract(req: ExtractRequest) -> dict:
    message = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=512,
        system=INVOICE_SYSTEM,
        messages=[{"role": "user", "content": req.invoice_text}],
    )
    return json.loads(message.content[0].text.strip())
