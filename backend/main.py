from pathlib import Path
from typing import Optional
import secrets
import json
from datetime import datetime


from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import sqlite3

from db import init_db, login_user, register_user, save_prediction, get_user_history
from backend.ai_pipeline import analyze, image_to_data_url
from backend.reporting import report_html

ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "frontend"
REPORTS_DIR = ROOT / "data" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="OcuVisionAI", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

init_db()
# Lightweight support inbox stored alongside the existing SQLite database.
try:
    conn = sqlite3.connect("retino_db.sqlite")
    conn.execute("CREATE TABLE IF NOT EXISTS support_requests (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, name TEXT, email TEXT, page TEXT, message TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)")
    conn.commit(); conn.close()
except Exception as exc:
    print("Support table setup warning:", exc)
TOKENS = {}

class AuthRequest(BaseModel):
    email: str
    password: str

class RegisterRequest(BaseModel):
    name: str
    email: str
    password: str

class SupportRequest(BaseModel):
    message: str
    page: str = "OcuVisionAI workspace"

def current_user(authorization: Optional[str]):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authentication required.")
    token = authorization.split(" ", 1)[1].strip()
    user = TOKENS.get(token)
    if not user:
        raise HTTPException(status_code=401, detail="Session expired. Please sign in again.")
    return user

@app.get("/")
def home():
    return FileResponse(FRONTEND / "index.html")

@app.get("/assets/logo")
def logo():
    path = FRONTEND / "assets" / "ocuv_isionai_logo.svg"
    if not path.exists():
        raise HTTPException(status_code=404)
    return FileResponse(path, media_type="image/svg+xml")

@app.get("/assets/sidebar-logo")
def sidebar_logo():
    path = FRONTEND / "assets" / "sidebar_logo.svg"
    if not path.exists():
        raise HTTPException(status_code=404)
    return FileResponse(path, media_type="image/svg+xml")

@app.get("/assets/{filename}")
def asset(filename: str):
    path = FRONTEND / "css" / filename
    if path.exists():
        return FileResponse(path)
    raise HTTPException(status_code=404)

@app.get("/js/{filename}")
def javascript(filename: str):
    path = FRONTEND / "js" / filename
    if path.exists():
        return FileResponse(path, media_type="application/javascript")
    raise HTTPException(status_code=404)

@app.get("/api/health")
def health():
    return {"status": "ok", "service": "OcuVisionAI"}

@app.post("/api/register")
def register(payload: RegisterRequest):
    if not payload.name.strip() or not payload.email.strip() or len(payload.password) < 6:
        raise HTTPException(
            status_code=400,
            detail="Name, email and a 6+ character password are required.",
        )
    if not register_user(payload.name.strip(), payload.email.strip(), payload.password):
        raise HTTPException(status_code=409, detail="An account with this email already exists.")
    return {"ok": True}

@app.post("/api/login")
def login(payload: AuthRequest):
    user = login_user(payload.email.strip(), payload.password)
    if not user:
        raise HTTPException(status_code=401, detail="Incorrect email or password.")

    token = secrets.token_urlsafe(32)
    TOKENS[token] = {
        "id": user["id"],
        "name": user["name"],
        "email": user["email"],
    }
    return {"token": token, "user": TOKENS[token]}

@app.post("/api/logout")
def logout(authorization: Optional[str] = Header(default=None)):
    if authorization and authorization.startswith("Bearer "):
        TOKENS.pop(authorization.split(" ", 1)[1].strip(), None)
    return {"ok": True}

def _report_path(user_id, report_id):
    return REPORTS_DIR / f"{int(user_id)}_{int(report_id)}.html"

def _history_with_report_flag(user_id):
    rows=[]
    for row in get_user_history(user_id):
        item=dict(row)
        rid=item.get("id")
        item["report_available"] = bool(rid and _report_path(user_id, rid).exists())
        rows.append(item)
    return rows

@app.get("/api/history")
def history(authorization: Optional[str] = Header(default=None)):
    user = current_user(authorization)
    return {"history": _history_with_report_flag(user["id"])}

@app.get("/api/history/{history_id}")
def history_detail(history_id:int, authorization: Optional[str] = Header(default=None)):
    user = current_user(authorization)
    rows = get_user_history(user["id"])
    item = next((dict(r) for r in rows if int(r.get("id", -1)) == history_id), None)
    if not item:
        raise HTTPException(status_code=404, detail="Screening not found.")
    payload_path = REPORTS_DIR / f"{int(user['id'])}_{history_id}.json"
    analysis = {}
    if payload_path.exists():
        try: analysis=json.loads(payload_path.read_text(encoding="utf-8"))
        except Exception: analysis={}
    return {"history": item, "analysis": analysis, "report_available": _report_path(user["id"],history_id).exists()}

@app.get("/api/reports")
def reports(authorization: Optional[str] = Header(default=None)):
    user=current_user(authorization)
    rows=_history_with_report_flag(user["id"])
    available=[]
    for row in rows:
        if row.get("report_available"):
            available.append({"id":row.get("id"),"prediction":row.get("prediction"),"confidence":row.get("confidence"),"created_at":row.get("created_at"),"image_path":row.get("image_path")})
    return {"reports":available}

@app.get("/api/reports/{report_id}")
def get_report(report_id:int, authorization: Optional[str] = Header(default=None)):
    user=current_user(authorization)
    allowed={int(row.get("id")) for row in get_user_history(user["id"]) if row.get("id") is not None}
    if report_id not in allowed:
        raise HTTPException(status_code=404, detail="Report not found.")
    path=_report_path(user["id"],report_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Report is not available for this screening.")
    return {"report_html":path.read_text(encoding="utf-8"),"id":report_id}

@app.get("/api/reports/{report_id}/download")
def download_report(report_id:int, authorization: Optional[str] = Header(default=None)):
    user = current_user(authorization)
    allowed={int(row.get("id")) for row in get_user_history(user["id"]) if row.get("id") is not None}
    if report_id not in allowed:
        raise HTTPException(status_code=404, detail="Report not found.")
    path=_report_path(user["id"],report_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Report is not available for this screening.")
    return FileResponse(path, media_type="text/html", filename=f"OcuVisionAI_Screening_Report_{report_id}.html")

@app.post("/api/analyze")
async def analyze_image(
    file: UploadFile = File(...),
    authorization: Optional[str] = Header(default=None),
):
    user = current_user(authorization)

    allowed = {"image/jpeg", "image/png", "image/jpg"}
    if file.content_type not in allowed:
        raise HTTPException(status_code=400, detail="Please upload JPG, JPEG or PNG.")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="The uploaded image is empty.")
    if len(data) > 200 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Maximum image size is 200 MB.")

    try:
        result = analyze(data)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}") from exc

    if not result["ok"]:
        return result

    save_prediction(
        user["id"],
        file.filename or "retinal_image",
        result["prediction"],
        result["confidence"],
    )

    images = {
        "original": image_to_data_url(result["original"]),
        "enhanced": image_to_data_url(result["enhanced"]),
        "gradcam": image_to_data_url(result["gradcam"]),
        "vessel": image_to_data_url(result["vessel"]),
        "lesion": image_to_data_url(result["lesion"]),
        "optic_disc": image_to_data_url(result["optic_disc"]),
        "microaneurysm": image_to_data_url(result["clinical_maps"]["microaneurysm"]),
        "exudate": image_to_data_url(result["clinical_maps"]["exudate"]),
        "hemorrhage": image_to_data_url(result["clinical_maps"]["hemorrhage"]),
        "neovascularization": image_to_data_url(result["clinical_maps"]["neovascularization"]),
        "fovea": image_to_data_url(result["clinical_maps"]["fovea"]),
    }
    report = report_html(result, file.filename or "retinal_image", images=images)
    try:
        latest_rows = get_user_history(user["id"])
        report_id = int(latest_rows[0]["id"]) if latest_rows else 0
        if report_id:
            _report_path(user["id"], report_id).write_text(report, encoding="utf-8")
            payload = {
                "filename": file.filename or "retinal_image",
                "quality": result["quality"], "enhanced_quality": result["enhanced_quality"],
                "prediction": result["prediction"], "confidence": result["confidence"],
                "class_probabilities": result["class_probabilities"], "images": images,
                "optic_disc_location": result.get("optic_disc_location"), "fovea_location": result.get("fovea_location"),
                "evidence_summary": result.get("evidence_summary", {}), "confidence_calibration": result.get("confidence_calibration", {}),
                "referable_flag": result.get("referable_flag", False),
            }
            (REPORTS_DIR / f"{int(user['id'])}_{report_id}.json").write_text(json.dumps(payload), encoding="utf-8")
    except Exception as exc:
        print("Report persistence warning:", exc)
        report_id = 0

    return {
        "ok": True,
        "filename": file.filename,
        "quality": result["quality"],
        "enhanced_quality": result["enhanced_quality"],
        "prediction": result["prediction"],
        "confidence": result["confidence"],
        "class_probabilities": result["class_probabilities"],
        "images": images,
        "optic_disc_location": result.get("optic_disc_location"),
        "fovea_location": result.get("fovea_location"),
        "evidence_summary": result.get("evidence_summary", {}),
        "confidence_calibration": result.get("confidence_calibration", {}),
        "report_html": report,
        "report_id": report_id,
        "referable_flag": result.get("referable_flag", False),
        "disclaimer": "AI screening support only. Model evidence is not a confirmed clinical diagnosis.",
    }

@app.post("/api/support")
def create_support_request(payload: SupportRequest, authorization: Optional[str] = Header(default=None)):
    user = current_user(authorization)
    message = payload.message.strip()
    if len(message) < 3:
        raise HTTPException(status_code=400, detail="Please describe the issue or question.")
    conn = sqlite3.connect("retino_db.sqlite")
    cur = conn.cursor()
    cur.execute("INSERT INTO support_requests (user_id,name,email,page,message) VALUES (?,?,?,?,?)", (user["id"], user["name"], user["email"], payload.page[:200], message[:5000]))
    conn.commit(); request_id=cur.lastrowid; conn.close()
    return {"ok": True, "id": request_id, "message": "Support request recorded."}

@app.get("/api/validation/requirements")
def validation_requirements():
    return {
        "referable_definition": "International Clinical DR severity levels 2+ (Moderate, Severe, Proliferative DR)",
        "required_metrics": ["sensitivity", "specificity", "accuracy", "confusion_matrix", "calibration"],
        "status": "Requires independent labelled validation/test data; no clinical performance claim is made by the prototype.",
    }
