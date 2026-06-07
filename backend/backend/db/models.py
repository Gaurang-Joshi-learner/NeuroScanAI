import uuid, enum
from datetime import datetime
from sqlalchemy import String, Boolean, DateTime, ForeignKey, Integer, Float, Text, Enum as SAEnum
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func
from db.database import Base

def gen_id(): return str(uuid.uuid4())

class AnalysisType(str, enum.Enum):
    SEIZURE = "seizure"
    SPEECH  = "speech"

class AnalysisStatus(str, enum.Enum):
    QUEUED     = "queued"
    PROCESSING = "processing"
    DONE       = "done"
    FAILED     = "failed"

class UserRole(str, enum.Enum):
    ADMIN  = "admin"
    MEMBER = "member"

class Organization(Base):
    __tablename__ = "organizations"
    id:         Mapped[str]      = mapped_column(String(36), primary_key=True, default=gen_id)
    name:       Mapped[str]      = mapped_column(String(255), nullable=False)
    is_active:  Mapped[bool]     = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    users:    Mapped[list["User"]]     = relationship("User",    back_populates="organization")
    analyses: Mapped[list["Analysis"]] = relationship("Analysis", back_populates="organization")

class User(Base):
    __tablename__ = "users"
    id:              Mapped[str]           = mapped_column(String(36), primary_key=True, default=gen_id)
    org_id:          Mapped[str]           = mapped_column(String(36), ForeignKey("organizations.id"))
    email:           Mapped[str]           = mapped_column(String(255), unique=True, nullable=False, index=True)
    hashed_password: Mapped[str | None]    = mapped_column(String(255), nullable=True)
    full_name:       Mapped[str | None]    = mapped_column(String(255), nullable=True)
    avatar_url:      Mapped[str | None]    = mapped_column(String(500), nullable=True)
    role:            Mapped[UserRole]      = mapped_column(SAEnum(UserRole), default=UserRole.MEMBER)
    is_active:       Mapped[bool]          = mapped_column(Boolean, default=True)
    google_id:       Mapped[str | None]    = mapped_column(String(255), nullable=True, unique=True)
    last_login:      Mapped[datetime|None] = mapped_column(DateTime, nullable=True)
    created_at:      Mapped[datetime]      = mapped_column(DateTime, server_default=func.now())
    organization: Mapped["Organization"]  = relationship("Organization", back_populates="users")
    analyses:     Mapped[list["Analysis"]] = relationship("Analysis", back_populates="user")

class Analysis(Base):
    __tablename__ = "analyses"
    id:             Mapped[str]             = mapped_column(String(36), primary_key=True, default=gen_id)
    org_id:         Mapped[str]             = mapped_column(String(36), ForeignKey("organizations.id"))
    user_id:        Mapped[str]             = mapped_column(String(36), ForeignKey("users.id"))
    analysis_type:  Mapped[AnalysisType]    = mapped_column(SAEnum(AnalysisType))
    status:         Mapped[AnalysisStatus]  = mapped_column(SAEnum(AnalysisStatus), default=AnalysisStatus.QUEUED)
    filename:       Mapped[str]             = mapped_column(String(500), nullable=False)
    file_path:      Mapped[str]             = mapped_column(String(1000), nullable=False)
    file_size_bytes:Mapped[int]             = mapped_column(Integer, default=0)
    result:         Mapped[dict | None]     = mapped_column(JSONB, nullable=True)
    error_message:  Mapped[str | None]      = mapped_column(Text, nullable=True)
    model_version:  Mapped[str | None]      = mapped_column(String(50), nullable=True)
    created_at:     Mapped[datetime]        = mapped_column(DateTime, server_default=func.now())
    completed_at:   Mapped[datetime|None]   = mapped_column(DateTime, nullable=True)
    organization: Mapped["Organization"] = relationship("Organization", back_populates="analyses")
    user:         Mapped["User"]         = relationship("User",         back_populates="analyses")
