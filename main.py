from __future__ import annotations

import base64
import json
import os

import boto3
import numpy as np
import openai
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

MODEL_ID = "amazon.nova-lite-v1:0"

oai = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def invoke(messages: list, system: str | None = None, max_tokens: int = 1024) -> str:
    body: dict = {
        "messages": messages,
        "inferenceConfig": {"max_new_tokens": max_tokens},
    }
    if system:
        body["system"] = [{"text": system}]
    response = bedrock.invoke_model(
        modelId=MODEL_ID,
        body=json.dumps(body),
    )
    result = json.loads(response["body"].read())
    return result["output"]["message"]["content"][0]["text"].strip()


def clean_json(text: str) -> str:
    return text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()


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
                {"image": {"format": "png", "source": {"bytes": req.image_base64}}},
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
    text = invoke([{"role": "user", "content": [{"text": req.invoice_text}]}], system=INVOICE_SYSTEM)
    return json.loads(clean_json(text))


# ── Dynamic Extract ───────────────────────────────────────────────────────────

class DynamicExtractRequest(BaseModel):
    text: str
    schema: dict[str, str]


@app.post("/dynamic-extract")
def dynamic_extract(req: DynamicExtractRequest) -> dict:
    schema_lines = "\n".join(f"  - {k}: {v}" for k, v in req.schema.items())
    system = f"""You are a structured data extractor. Extract fields from the given text according to the schema and return a single JSON object with no other text.

Schema (field: type):
{schema_lines}

Rules:
- Return EXACTLY the keys listed — no extras, no missing keys.
- Use null for fields not found in the text.
- Dates must be ISO format YYYY-MM-DD.
- integer and float fields must be JSON numbers, not strings.
- Return ONLY the JSON object, no markdown, no explanation."""

    text = invoke([{"role": "user", "content": [{"text": req.text}]}], system=system)
    return json.loads(clean_json(text))


# ── Rich Invoice Parse ────────────────────────────────────────────────────────

RICH_INVOICE_SYSTEM = """You are a precise invoice parser. Extract structured data from the invoice text and return ONLY a valid JSON object matching this exact schema — no markdown, no explanation:

{
  "vendor": string (biller name exactly as written),
  "currency": ISO 4217 code (USD/EUR/GBP/INR/JPY — infer from symbols like ₹=INR, £=GBP, €=EUR),
  "total_amount": integer (main unit only, no decimals — convert spelled-out numbers, K suffix, Indian grouping),
  "invoice_date": YYYY-MM-DD string,
  "due_in_days": integer (Net 30 → 30, "two weeks" → 14, etc.),
  "is_paid": boolean (true if paid, false if awaiting payment),
  "priority": one of "low"/"normal"/"high"/"urgent",
  "contact_email": string lowercased,
  "line_items": array of {"sku": string, "quantity": integer, "unit_price": integer} in order of appearance,
  "item_count": integer (number of line items)
}"""


class RichInvoiceRequest(BaseModel):
    document_id: str
    text: str
    schema: dict | None = None


@app.post("/parse-invoice")
def parse_invoice(req: RichInvoiceRequest) -> dict:
    text = invoke(
        [{"role": "user", "content": [{"text": req.text}]}],
        system=RICH_INVOICE_SYSTEM,
        max_tokens=1024,
    )
    return json.loads(clean_json(text))


# ── Semantic Search (text-embedding-3-small) ──────────────────────────────────

class SemanticSearchRequest(BaseModel):
    query_id: str
    query: str
    candidates: list[str]


@app.post("/semantic-search")
def semantic_search(req: SemanticSearchRequest) -> dict:
    texts = [req.query] + req.candidates
    resp = oai.embeddings.create(model="text-embedding-3-small", input=texts)
    embs = np.array([e.embedding for e in resp.data])
    q_emb = embs[0]
    c_embs = embs[1:]
    sims = c_embs @ q_emb  # unit-normalised by default
    top3 = np.argsort(-sims)[:3].tolist()
    return {"ranking": top3}
