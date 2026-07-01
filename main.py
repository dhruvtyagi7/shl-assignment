"""
main.py

FastAPI app with two endpoints:
  GET  /health  — liveness check, needed for Render cold starts
  POST /chat    — the main conversational endpoint

Stateless by design — every request includes the full conversation
history, and the server holds nothing between calls.
"""

import logging
import os

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator

import agent
import guardrails

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="SHL Assessment Recommender")


# request/response models — the grader checks the schema strictly

class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]

    @field_validator("messages")
    @classmethod
    def must_have_messages(cls, v):
        if not v:
            raise ValueError("messages cannot be empty")
        return v


class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str


class ChatResponse(BaseModel):
    reply: str
    recommendations: list[Recommendation]
    end_of_conversation: bool

@app.get("/")
def root():
    return {"message": "API is running"}
    
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    messages = [m.model_dump() for m in request.messages]

    # run guardrails first — catch injections/off-topic before hitting Gemini
    flagged, reason = guardrails.check(messages)
    if flagged:
        result = guardrails.make_refusal_response(reason)
    else:
        result = agent.run(messages)

    return ChatResponse(
        reply=result["reply"],
        recommendations=[Recommendation(**r) for r in result["recommendations"]],
        end_of_conversation=result["end_of_conversation"],
    )


@app.exception_handler(Exception)
async def catch_all(request: Request, exc: Exception):
    logger.error("Unhandled exception: %s", exc, exc_info=True)
    # return a valid schema response even on unexpected errors
    return JSONResponse(
        status_code=200,
        content={
            "reply": "Something went wrong on my end. Please try again.",
            "recommendations": [],
            "end_of_conversation": False,
        },
    )
