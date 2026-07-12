"""
backend/schemas.py
==================
Pydantic request/response schemas used by the API layer.
"""

from typing import Optional
from pydantic import BaseModel


class AdminCreate(BaseModel):
    username: str
    password: str
    setup_token: str


class VerifyRequest(BaseModel):
    category: Optional[str] = None
    status: str = "Verified"
    note: Optional[str] = None


class VoteRequest(BaseModel):
    voter_fingerprint: str
