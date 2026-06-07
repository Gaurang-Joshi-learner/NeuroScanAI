"""
routers/analysis.py
--------------------
Endpoints for uploading EEG files and running analysis.
POST /analysis/seizure  — upload EDF, detect seizures
POST /analysis/speech   — upload MAT, decode speech
GET  /analysis/         — list user's analyses
GET  /analysis/{id}     — get single analysis result
DELETE /analysis/{id}   — delete analysis
"""
import os, uuid, aiofiles
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from db.database import get_db, settings
from db.models import User, Analysis, AnalysisType, AnalysisStatus
from db.schemas import AnalysisResponse
from auth.dependencies import get_current_user
from inference.seizure_engine import get_seizure_engine
from inference.speech_engine import get_speech_engine
from fastapi.responses import FileResponse
from utils.pdf_report import generate_seizure_report
import tempfile

router = APIRouter(prefix="/analysis", tags=["analysis"])

def gen_id(): return str(uuid.uuid4())

ALLOWED_SEIZURE = {".edf"}
ALLOWED_SPEECH  = {".mat"}


async def run_analysis_bg(analysis_id: str, file_path: str, analysis_type: str, models_dir: str):
    from db.database import AsyncSessionLocal
    async with AsyncSessionLocal() as db:
        try:
            result = await db.execute(select(Analysis).where(Analysis.id == analysis_id))
            analysis = result.scalar_one()
            analysis.status = AnalysisStatus.PROCESSING
            await db.commit()

            if analysis_type == "seizure":
                engine = get_seizure_engine(os.path.join(models_dir, "seizure"))
                output = engine.predict(file_path)
            else:
                engine = get_speech_engine(os.path.join(models_dir, "speech"))
                output = engine.predict(file_path)

            result2 = await db.execute(select(Analysis).where(Analysis.id == analysis_id))
            analysis = result2.scalar_one()
            analysis.status       = AnalysisStatus.DONE
            analysis.result       = output
            analysis.model_version = output.get("model_version")
            analysis.completed_at = datetime.now()
            await db.commit()

        except Exception as e:
            result3 = await db.execute(select(Analysis).where(Analysis.id == analysis_id))
            analysis = result3.scalar_one_or_none()
            if analysis:
                analysis.status        = AnalysisStatus.FAILED
                analysis.error_message = str(e)
                await db.commit()


async def _upload(
    file: UploadFile, analysis_type: str, allowed_exts: set,
    background_tasks: BackgroundTasks,
    current_user: User, db: AsyncSession
) -> AnalysisResponse:
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed_exts:
        raise HTTPException(400, f"Only {allowed_exts} files supported for {analysis_type} analysis")

    content = await file.read()
    max_bytes = settings.MAX_FILE_SIZE_MB * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(413, f"File exceeds {settings.MAX_FILE_SIZE_MB}MB")

    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    fid = gen_id()
    file_path = os.path.join(settings.UPLOAD_DIR, f"{fid}{ext}")
    async with aiofiles.open(file_path, "wb") as f:
        await f.write(content)

    analysis = Analysis(
        id=fid, org_id=current_user.org_id, user_id=current_user.id,
        analysis_type=AnalysisType(analysis_type),
        status=AnalysisStatus.QUEUED,
        filename=file.filename, file_path=file_path,
        file_size_bytes=len(content),
    )
    db.add(analysis)
    await db.flush()

    background_tasks.add_task(
        run_analysis_bg, analysis.id, file_path, analysis_type, settings.MODELS_DIR
    )
    return AnalysisResponse.model_validate(analysis)


@router.post("/seizure", response_model=AnalysisResponse, status_code=201)
async def analyze_seizure(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await _upload(file, "seizure", ALLOWED_SEIZURE, background_tasks, current_user, db)


@router.post("/speech", response_model=AnalysisResponse, status_code=201)
async def analyze_speech(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await _upload(file, "speech", ALLOWED_SPEECH, background_tasks, current_user, db)


@router.get("/", response_model=list[AnalysisResponse])
async def list_analyses(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    skip: int = 0, limit: int = 50,
):
    result = await db.execute(
        select(Analysis)
        .where(Analysis.org_id == current_user.org_id)
        .order_by(desc(Analysis.created_at))
        .offset(skip).limit(limit)
    )
    return [AnalysisResponse.model_validate(a) for a in result.scalars().all()]


@router.get("/{analysis_id}", response_model=AnalysisResponse)
async def get_analysis(
    analysis_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Analysis).where(Analysis.id == analysis_id,
                               Analysis.org_id == current_user.org_id)
    )
    a = result.scalar_one_or_none()
    if not a: raise HTTPException(404, "Analysis not found")
    return AnalysisResponse.model_validate(a)


@router.delete("/{analysis_id}", status_code=204)
async def delete_analysis(
    analysis_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Analysis).where(Analysis.id == analysis_id,
                               Analysis.org_id == current_user.org_id)
    )
    a = result.scalar_one_or_none()
    if not a: raise HTTPException(404, "Analysis not found")
    if os.path.exists(a.file_path): os.remove(a.file_path)
    await db.delete(a)
    await db.commit()
    return
@router.get("/{analysis_id}/pdf")
async def download_pdf(
    analysis_id: str,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Analysis).where(
            Analysis.id == analysis_id,
            Analysis.org_id == current_user.org_id
        )
    )

    analysis = result.scalar_one_or_none()

    if not analysis:
        raise HTTPException(
            status_code=404,
            detail="Analysis not found"
        )

    pdf_path = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ).name
    print("ANALYSIS RESULT:")
    print(analysis.result)

    generate_seizure_report(pdf_path, analysis)

    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename=f"{analysis.filename}_report.pdf"
    )
