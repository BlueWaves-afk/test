from __future__ import annotations

import base64
import json
import os

import boto3
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

bedrock = boto3.client(
    "bedrock-runtime",
    region_name=os.environ.get("AWS_REGION", "us-east-1"),
    aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
)

MODEL_ID = "amazon.nova-lite-v1:0"  # supports vision, no access request needed


def invoke(messages: list, system: str | None = None) -> str:
    body: dict = {
        "messages": messages,
        "inferenceConfig": {"max_new_tokens": 512},
    }
    if system:
        body["system"] = [{"text": system}]
    response = bedrock.invoke_model(
        modelId=MODEL_ID,
        body=json.dumps(body),
    )
    result = json.loads(response["body"].read())
    return result["output"]["message"]["content"][0]["text"].strip()


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

    answer = invoke([
        {
            "role": "user",
            "content": [
                {
                    "image": {
                        "format": "png",
                        "source": {"bytes": req.image_base64},
                    },
                },
                {
                    "text": (
                        f"{req.question}\n\n"
                        "Reply with only the raw answer value — no units, no extra text, "
                        "no explanation. For numbers, use digits only (e.g. '4089.35')."
                    ),
                },
            ],
        }
    ])
    return ImageQAResponse(answer=answer)


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
    text = invoke(
        [{"role": "user", "content": [{"text": req.invoice_text}]}],
        system=INVOICE_SYSTEM,
    )
    text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return json.loads(text)
