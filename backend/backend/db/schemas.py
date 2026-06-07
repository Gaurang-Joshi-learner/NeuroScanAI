from pydantic import BaseModel, EmailStr, field_validator
from datetime import datetime
from typing import Optional
import re

class RegisterRequest(BaseModel):
    email: EmailStr
    password: str
    full_name: str
    org_name: str

    @field_validator("password")
    @classmethod
    def password_strength(cls, v):
        if len(v) < 8: raise ValueError("Min 8 characters")
        if not re.search(r"[A-Z]", v): raise ValueError("Need uppercase letter")
        if not re.search(r"\d", v): raise ValueError("Need a number")
        return v

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    user: "UserResponse"

class RefreshRequest(BaseModel):
    refresh_token: str

class UserResponse(BaseModel):
    id: str
    email: str
    full_name: Optional[str]
    avatar_url: Optional[str]
    role: str
    org_id: str
    created_at: datetime
    class Config: from_attributes = True

class AnalysisResponse(BaseModel):
    id: str
    analysis_type: str
    status: str
    filename: str
    file_size_bytes: int
    result: Optional[dict]
    error_message: Optional[str]
    model_version: Optional[str]
    created_at: datetime
    completed_at: Optional[datetime]
    class Config: from_attributes = True

TokenResponse.model_rebuild()
