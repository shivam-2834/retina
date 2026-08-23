import io
import csv
import time
import streamlit as st
import torch
import torch.nn as nn
import cv2
from torchvision import models, transforms
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
from db import init_db, login_user, register_user, save_prediction, get_user_history
from styles import load_css

# ─── PAGE CONFIG ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="OcuVisionAI",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_db()
st.markdown(load_css(), unsafe_allow_html=True)

# ─── SESSION DEFAULTS ────────────────────────────────────────────────────────
for key, val in {
    "logged_in": False,
    "user_id": None,
    "user_name": None,
    "active_page": "Dashboard",
    "analysis_result": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = val


# ─── CONSTANTS & HELPERS ─────────────────────────────────────────────────────
SEVERITY_ORDER = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
MODEL_PATH = "model.pth"

@st.cache_resource
def load_model():
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 5)

    model.load_state_dict(
        torch.load(MODEL_PATH, map_location="cpu")
    )

    model.eval()
    return model

model = load_model()

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

CLASS_NAMES = [
     "Mild",
    "Moderate",
    "No DR",
    "Proliferative DR",
    "Severe"
]
# ─── EXPLAINABLE AI (Grad-CAM) ─────────────────────────────────────────────

def generate_gradcam(model, input_tensor, target_class):
    """
    Generate a Grad-CAM heatmap for the existing ResNet-50 model.
    This does NOT change the model or its prediction.
    """

    model.eval()

    # ResNet-50 final convolutional layer
    target_layer = model.layer4[-1]

    activations = []
    gradients = []

    def forward_hook(module, inputs, output):
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)


    try:
        # Forward pass
        output = model(input_tensor)

        # Clear previous gradients
        model.zero_grad()

        # Get score for the predicted class
        score = output[0, target_class]

        # Backpropagate
        score.backward()

        # Get saved activations and gradients
        activation = activations[0]
        gradient = gradients[0]

        # Average gradients across spatial dimensions
        weights = gradient.mean(dim=(2, 3), keepdim=True)

        # Weighted combination of feature maps
        cam = (weights * activation).sum(dim=1, keepdim=True)

        # Remove negative influence
        cam = torch.relu(cam)

        # Resize heatmap to image size
        cam = torch.nn.functional.interpolate(
            cam,
            size=(256, 256),
            mode="bilinear",
            align_corners=False
        )

        # Normalize between 0 and 1
        cam = cam.squeeze().detach().cpu()

        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())

        return cam.numpy()

    finally:
        # Remove hooks after use
        forward_handle.remove()
        backward_handle.remove()

def logout():
    st.session_state.clear()
    st.rerun()


def severity_badge(prediction: str) -> str:
    colors = {
        "No DR":            ("#22c55e", "✅"),
        "Mild":             ("#f59e0b", "⚠️"),
        "Moderate":         ("#f97316", "🔶"),
        "Severe":           ("#ef4444", "🔴"),
        "Proliferative DR": ("#dc2626", "🚨"),
    }
    color, icon = colors.get(prediction, ("#94a3b8", "❓"))
    return (
        f'<span style="background:{color};color:white;padding:3px 10px;'
        f'border-radius:20px;font-size:0.8rem;font-weight:600;">{icon} {prediction}</span>'
    )


def history_to_csv(history: list) -> bytes:
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["ID", "Date", "File", "Prediction", "Confidence (%)"])
    for row in history:
        raw_date = row.get("created_at")
        if raw_date:
            if hasattr(raw_date, 'strftime'):
                date = raw_date.strftime("%Y-%m-%d %H:%M:%S")
            else:
                try:
                    date = str(raw_date).replace("-", "/")[0:19]
                except:
                    date = str(raw_date)
        else:
            date = ""
        writer.writerow([
            row.get("id", ""),
            date,
            row.get("image_path", ""),
            row.get("prediction", ""),
            f"{row.get('confidence') or 0:.1f}",
        ])
    return output.getvalue().encode("utf-8")


def high_risk_banner(history: list):
    if not history:
        return
    latest_pred = history[0].get("prediction", "")
    if latest_pred in ("Severe", "Proliferative DR"):
        st.markdown(f"""
        <div style="background:#fef2f2;border:2px solid #dc2626;border-radius:10px;
                    padding:0.9rem 1.2rem;margin-bottom:1.2rem;display:flex;
                    align-items:center;gap:0.8rem;">
            <span style="font-size:1.5rem;">🚨</span>
            <div>
                <strong style="color:#dc2626;">High-Risk Result Detected</strong><br>
                <span style="font-size:0.88rem;color:#7f1d1d;">
                    Your most recent scan was classified as <strong>{latest_pred}</strong>.
                    Please seek immediate ophthalmology consultation.
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
#  AUTH PAGE
# ═══════════════════════════════════════════════════════════════════════════
def page_login():
    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        st.markdown("""
        <div style="text-align:center;padding:2rem 0 1rem;">
            <h1 style="font-size:2.2rem;margin-bottom:0;">👁️ OcuVisionAI</h1>
            <p style="color:#64748b;margin-top:0.3rem;">Detect Early. Save Vision.</p>
        </div>
        """, unsafe_allow_html=True)

        tab_login, tab_signup = st.tabs(["🔐 Login", "📝 Create Account"])

        with tab_login:
            st.markdown("<br>", unsafe_allow_html=True)
            email = st.text_input("Email address", key="login_email", placeholder="you@example.com")
            password = st.text_input("Password", type="password", key="login_pw", placeholder="••••••••")
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Sign In", use_container_width=True, type="primary"):
                if not email or not password:
                    st.error("Please fill in all fields.")
                else:
                    user = login_user(email, password)
                    if user:
                        st.session_state.logged_in = True
                        st.session_state.user_id = user["id"]
                        st.session_state.user_name = user["name"]
                        st.success(f"Welcome back, {user['name']}!")
                        st.rerun()
                    else:
                        st.error("Incorrect email or password.")

        with tab_signup:
            st.markdown("<br>", unsafe_allow_html=True)
            name = st.text_input("Full name", key="reg_name", placeholder="Dr. Priya Sharma")
            reg_email = st.text_input("Email address", key="reg_email", placeholder="you@example.com")
            reg_pw = st.text_input("Password", type="password", key="reg_pw", placeholder="Min. 6 characters")
            reg_pw2 = st.text_input("Confirm password", type="password", key="reg_pw2", placeholder="••••••••")
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Create Account", use_container_width=True, type="primary"):
                if not name or not reg_email or not reg_pw:
                    st.error("All fields are required.")
                elif reg_pw != reg_pw2:
                    st.error("Passwords do not match.")
                elif len(reg_pw) < 6:
                    st.error("Password must be at least 6 characters.")
                else:
                    if register_user(name, reg_email, reg_pw):
                        st.success("Account created! Please sign in.")
                    else:
                        st.error("An account with this email already exists.")


# ═══════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════
def render_sidebar():
    with st.sidebar:
        st.markdown(f"""
        <div style="padding:1rem 0 0.5rem;">
            <h2 style="margin:0;">👁️ OcuVisionAI</h2>
            <p style="font-size:0.8rem;opacity:0.75;margin:0.2rem 0 1rem;">Retinal Screening System</p>
            <div style="background:rgba(255,255,255,0.1);border-radius:8px;
                        padding:0.6rem 0.8rem;margin-bottom:1rem;">
                <strong>👤 {st.session_state.user_name}</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

        pages = {
            "📊 Dashboard": "Dashboard",
            "🔬 New Scan":  "New Scan",
            "📜 History":   "History",
        }
        for label, page in pages.items():
            active = st.session_state.active_page == page
            if st.button(
                label,
                use_container_width=True,
                type="primary" if active else "secondary",
                key=f"nav_{page}",
            ):
                st.session_state.active_page = page
                st.rerun()

        st.markdown("---")
        if st.button("🚪 Logout", use_container_width=True):
            logout()


# ═══════════════════════════════════════════════════════════════════════════
#  PAGE: DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════
def page_dashboard():
    import pandas as pd
    from collections import Counter
    last_scan_str = "—" 
    last_scan = None

    user_name = st.session_state.user_name

    st.html(
    """
    <div style="
        background: linear-gradient(135deg, #0f172a, #1e3a5f);
        padding: 24px;
        border-radius: 16px;
        margin-bottom: 20px;
    ">
        <div style="
            font-size: 14px;
            color: #ffffff;
            margin-bottom: 8px;
        ">
            OcuVisionAI • Retinal Screening
        </div>

        <div style="
            font-size: 32px;
            font-weight: 700;
            color: #ffffff;
            margin-bottom: 6px;
        ">
            👋 Welcome back, """ + user_name + """
        </div>

        <div style="
            font-size: 15px;
            color: #ffffff;
        ">
            Here's your retinal screening summary
        </div>
    </div>
    """
)
    

    history = get_user_history(st.session_state.user_id)

    # ── High-risk banner ──
    high_risk_banner(history)

    # ── Metrics (no avg confidence) ──
    total = len(history)
    last_scan = history[0]["created_at"] if history else None
    if last_scan:
        if hasattr(last_scan, 'strftime'):
            last_scan_str = last_scan.strftime("%b %d, %Y")
        else:
            try:
                last_scan_str = str(last_scan).replace("-", "/")[0:10]
            except:
                last_scan_str = str(last_scan)
    last_pred = history[0].get("prediction", "—") if history else "—"

    c1, c2, c3 = st.columns(3)
    

    with c1:
     st.markdown(f"""
    <div style="
        background: white;
        padding: 1.2rem;
        border-radius: 14px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 8px rgba(15,23,42,0.06);
    ">
        <div style="font-size:14px; color:#64748b;">
            🔬 Total Scans
        </div>
        <div style="
            font-size:32px;
            font-weight:700;
            color:#0f172a;
            margin-top:6px;
        ">
            {total}
        </div>
    </div>
    """, unsafe_allow_html=True)

    with c2:
     st.markdown(f"""
    <div style="
        background: white;
        padding: 1.2rem;
        border-radius: 14px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 8px rgba(15,23,42,0.06);
    ">
        <div style="font-size:14px; color:#64748b;">
            📅 Last Scan
        </div>
        <div style="
            font-size:28px;
            font-weight:700;
            color:#0f172a;
            margin-top:6px;
        ">
            {last_scan_str}
        </div>
    </div>
    """, unsafe_allow_html=True)

    with c3:
     st.markdown(f"""
    <div style="
        background: white;
        padding: 1.2rem;
        border-radius: 14px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 8px rgba(15,23,42,0.06);
    ">
        <div style="font-size:14px; color:#64748b;">
            🏷️ Last Diagnosis
        </div>
        <div style="
            font-size:28px;
            font-weight:700;
            color:#0f172a;
            margin-top:6px;
        ">
            {last_pred}
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    col_a, col_b = st.columns([2, 1])

    with col_a:
        # ── Confidence over time chart ──
        if history:
            st.markdown("### 📈 Confidence Over Time")
            chart_rows = [
                {"Scan": f"#{i+1} {r.get('image_path','')[:10]}", "Confidence (%)": round(r.get("confidence") or 0, 1)}
                for i, r in enumerate(reversed(history))
            ]
            df_chart = pd.DataFrame(chart_rows).set_index("Scan")
            st.line_chart(df_chart, use_container_width=True)

        # ── Recent scans ──
        st.markdown("### 📋 Recent Scans")
        if not history:
            st.info("No scans yet. Start a new scan to see results here.")
        else:
            for row in history[:5]:
                conf = row.get("confidence") or 0
                pred = row.get("prediction", "Unknown")
                raw_date = row.get("created_at")
                if raw_date:
                    if hasattr(raw_date, 'strftime'):
                        date = raw_date.strftime("%b %d, %Y %H:%M")
                    else:
                        try:
                            date = str(raw_date).replace("-", "/")[0:16]
                        except:
                            date = str(raw_date)
                st.markdown(f"""
                <div style="display:flex;justify-content:space-between;align-items:center;
                            padding:0.7rem 1rem;border-radius:8px;margin-bottom:0.5rem;
                            background:#f8fafc;border:1px solid #e2e8f0;">
                    <div>
                        <strong>{date}</strong><br>
                        <span style="font-size:0.8rem;color:#64748b;">{row.get('image_path','—')}</span>
                    </div>
                    <div style="text-align:right;">
                        {severity_badge(pred)}<br>
                        <span style="font-size:0.8rem;color:#64748b;">Conf: {conf:.1f}%</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    with col_b:
        # ── Diagnosis breakdown bar chart ──
        if history:
            st.markdown("### 🏷️ Diagnosis Breakdown")
            counts = Counter(r.get("prediction", "Unknown") for r in history)
            # Order by severity
            ordered = {k: counts.get(k, 0) for k in SEVERITY_ORDER if k in counts}
            df_bar = pd.DataFrame({"Count": list(ordered.values())}, index=list(ordered.keys()))
            st.bar_chart(df_bar, use_container_width=True)

        st.markdown("### ⚡ Quick Start")
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔬 Start New Scan", use_container_width=True, type="primary"):
            st.session_state.active_page = "New Scan"
            st.rerun()
        if st.button("📜 View Full History", use_container_width=True):
            st.session_state.active_page = "History"
            st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div style="background:#eff6ff;border-left:4px solid #3b82f6;
                    padding:0.8rem;border-radius:6px;font-size:0.85rem;">
            <strong>ℹ️ How it works</strong><br>
            Upload a retinal fundus image. Our AI classifies diabetic retinopathy severity in seconds.
        </div>
        """, unsafe_allow_html=True)


#  PAGE: NEW SCAN
# ═══════════════════════════════════════════════════════════════════════════
def assess_image_quality(image):

    img = np.array(
        image.convert("RGB")
    ).astype(np.float32)

    # Convert RGB → grayscale
    gray = (
        0.299 * img[:, :, 0]
        + 0.587 * img[:, :, 1]
        + 0.114 * img[:, :, 2]
    )

    # --------------------------------------------------------
    # 1. Focus / Blur
    # --------------------------------------------------------
    gradient_x = np.diff(gray, axis=1)
    gradient_y = np.diff(gray, axis=0)

    sharpness = float(
        np.var(gradient_x) +
        np.var(gradient_y)
    )

    # --------------------------------------------------------
    # 2. Illumination
    # --------------------------------------------------------
    brightness = float(np.mean(gray))

    # --------------------------------------------------------
    # 3. Field of View
    # --------------------------------------------------------
    dark_pixels = np.sum(gray < 15)
    total_pixels = gray.size

    dark_ratio = dark_pixels / total_pixels

    # --------------------------------------------------------
    # QUALITY THRESHOLDS
    # Calibrated to be less strict for retinal datasets
    # --------------------------------------------------------

    # Previous: sharpness >= 40
    # This was too strict for many APTOS images.
    if sharpness >= 15:
        focus_status = "good"
    elif sharpness >= 7:
        focus_status = "borderline"
    else:
        focus_status = "poor"

    # More tolerant illumination range
    if 25 <= brightness <= 235:
        illumination_status = "good"
    elif 15 <= brightness <= 245:
        illumination_status = "borderline"
    else:
        illumination_status = "poor"

    # Allow normal black circular/background areas
    if dark_ratio <= 0.75:
        field_status = "good"
    elif dark_ratio <= 0.85:
        field_status = "borderline"
    else:
        field_status = "poor"

    # --------------------------------------------------------
    # Collect problems
    # --------------------------------------------------------

    failed_checks = []

    if focus_status == "borderline":
        failed_checks.append(
            "Image has mild blur or reduced sharpness."
        )

    elif focus_status == "poor":
        failed_checks.append(
            "Image is too blurry or out of focus."
        )

    if illumination_status == "borderline":
        failed_checks.append(
            "Image illumination is slightly outside the ideal range."
        )

    elif illumination_status == "poor":
        failed_checks.append(
            "Image is too dark or too bright."
        )

    if field_status == "borderline":
        failed_checks.append(
            "Retinal field of view is partially limited."
        )

    elif field_status == "poor":
        failed_checks.append(
            "Retinal field of view is insufficient."
        )

    # --------------------------------------------------------
    # FINAL DECISION
    # --------------------------------------------------------

    if (
        focus_status == "good"
        and illumination_status == "good"
        and field_status == "good"
    ):
        quality_status = "Good"

    elif (
        focus_status == "poor"
        or illumination_status == "poor"
        or field_status == "poor"
    ):
        quality_status = "Poor"

    else:
        quality_status = "Borderline"

    return {
        "status": quality_status,
        "sharpness": sharpness,
        "brightness": brightness,
        "dark_ratio": dark_ratio,

        "focus_ok": focus_status == "good",
        "illumination_ok": illumination_status == "good",
        "field_ok": field_status == "good",

        "focus_status": focus_status,
        "illumination_status": illumination_status,
        "field_status": field_status,

        "failed_checks": failed_checks,
    }
# ============================================================
# REAL RETINAL BLOOD VESSEL HIGHLIGHTING
# NumPy + PIL only — no OpenCV required
# ============================================================

def generate_vessel_evidence(image):

    import numpy as np
    from PIL import Image, ImageFilter

    # Original RGB image
    rgb = np.array(image.convert("RGB")).astype(np.float32)

    # --------------------------------------------------------
    # 1. Green channel
    # Retinal vessels have strong contrast in the green channel
    # --------------------------------------------------------

    green = rgb[:, :, 1]

    # --------------------------------------------------------
    # 2. Estimate local background using Gaussian blur
    # --------------------------------------------------------

    green_img = Image.fromarray(
        np.clip(green, 0, 255).astype(np.uint8)
    )

    blurred_img = green_img.filter(
        ImageFilter.GaussianBlur(radius=5)
    )

    background = np.array(
        blurred_img
    ).astype(np.float32)

    # --------------------------------------------------------
    # 3. Dark vessel response
    # Vessels are darker than their local surroundings
    # --------------------------------------------------------

    vessel_response = background - green

    # --------------------------------------------------------
    # 4. Retinal field mask
    # Ignore black area outside the retina
    # --------------------------------------------------------

    brightness = np.mean(rgb, axis=2)

    retinal_mask = brightness > 20

    # --------------------------------------------------------
    # 5. Adaptive threshold
    # Only keep stronger vessel-like structures
    # --------------------------------------------------------

    valid_response = vessel_response[retinal_mask]

    if len(valid_response) > 0:
        threshold = np.percentile(
            valid_response,
            88
        )
    else:
        threshold = 10

    vessel_mask = (
        (vessel_response > threshold)
        & retinal_mask
    )

    # --------------------------------------------------------
    # 6. Remove extremely tiny responses
    # --------------------------------------------------------

    local_strength = vessel_response > (
        threshold + 5
    )

    vessel_mask = (
        vessel_mask
        & (
            local_strength
            | (
                vessel_response
                > threshold
            )
        )
    )

    # --------------------------------------------------------
    # 7. Create natural vessel overlay
    # Original retina remains visible
    # --------------------------------------------------------

    result = rgb.copy()

    vessel_pixels = vessel_mask

    # Red-orange vessel highlighting
    result[vessel_pixels, 0] = 255
    result[vessel_pixels, 1] = (
        result[vessel_pixels, 1] * 0.25
    )
    result[vessel_pixels, 2] = (
        result[vessel_pixels, 2] * 0.20
    )

    # --------------------------------------------------------
    # 8. Blend highlighted vessels with original image
    # --------------------------------------------------------

    result = (
        rgb * 0.60
        + result * 0.40
    )

    result = np.clip(
        result,
        0,
        255
    ).astype(np.uint8)

    return Image.fromarray(result)
def generate_lesion_evidence(image):
    """
    Generate candidate retinal lesion evidence.

    This is an AI-assisted visualization only.
    Highlighted areas are candidate regions, not confirmed lesions.
    """

    rgb = np.array(image.convert("RGB")).astype(np.float32)

    red = rgb[:, :, 0]
    green = rgb[:, :, 1]
    blue = rgb[:, :, 2]

    # --------------------------------------------------------
    # 1. Retinal field
    # --------------------------------------------------------

    brightness = np.mean(rgb, axis=2)

    retinal_mask = brightness > 35

    # --------------------------------------------------------
    # 2. Detect reddish retinal regions
    # --------------------------------------------------------

    red_response = (
        red - green
    )

    # Smooth the response
    response_img = Image.fromarray(
        np.clip(red_response, 0, 255).astype(np.uint8)
    )

    response_img = response_img.filter(
        ImageFilter.GaussianBlur(radius=2)
    )

    red_response = np.array(
        response_img
    ).astype(np.float32)

    # --------------------------------------------------------
    # 3. Local darkness
    # Dark red lesions can be darker than surroundings
    # --------------------------------------------------------

    green_img = Image.fromarray(
        np.clip(green, 0, 255).astype(np.uint8)
    )

    background_img = green_img.filter(
        ImageFilter.GaussianBlur(radius=7)
    )

    background = np.array(
        background_img
    ).astype(np.float32)

    dark_response = (
        background - green
    )

    # --------------------------------------------------------
    # 4. Candidate lesion response
    # --------------------------------------------------------

    lesion_response = (
        red_response * 0.65
        + np.clip(dark_response, 0, 80) * 0.35
    )

    # --------------------------------------------------------
    # 5. Adaptive threshold
    # Use a high percentile so only strong candidates remain
    # --------------------------------------------------------

    valid_response = lesion_response[
        retinal_mask
    ]

    if len(valid_response) > 0:

        threshold = np.percentile(
            valid_response,
            99.2
        )

    else:

        threshold = 30

    lesion_mask = (
        (lesion_response > threshold)
        & retinal_mask
    )

    # --------------------------------------------------------
    # 6. Remove very dark background
    # --------------------------------------------------------

    lesion_mask = (
        lesion_mask
        & (brightness > 50)
    )

    # --------------------------------------------------------
    # 7. Remove vessel-like structures
    #
    # Vessels are usually elongated dark structures.
    # We estimate strong vessel response and exclude them.
    # --------------------------------------------------------

    vessel_img = Image.fromarray(
        np.clip(green, 0, 255).astype(np.uint8)
    )

    vessel_background_img = vessel_img.filter(
        ImageFilter.GaussianBlur(radius=5)
    )

    vessel_background = np.array(
        vessel_background_img
    ).astype(np.float32)

    vessel_response = (
        vessel_background - green
    )

    vessel_values = vessel_response[
        retinal_mask
    ]

    if len(vessel_values) > 0:

        vessel_threshold = np.percentile(
            vessel_values,
            90
        )

    else:

        vessel_threshold = 10

    vessel_like = (
        vessel_response > vessel_threshold
    )

    # Don't allow strong vessel pixels to become lesions
    lesion_mask = (
        lesion_mask
        & (~vessel_like)
    )

    # --------------------------------------------------------
    # 8. Remove isolated single-pixel noise
    # --------------------------------------------------------

    mask_img = Image.fromarray(
        (lesion_mask.astype(np.uint8) * 255)
    )

    # Slight morphological cleanup using PIL
    mask_img = mask_img.filter(
        ImageFilter.MaxFilter(size=3)
    )

    mask_img = mask_img.filter(
        ImageFilter.MinFilter(size=3)
    )

    lesion_mask = (
        np.array(mask_img) > 0
    )

    # --------------------------------------------------------
    # 9. Protect optic-disc area
    #
    # The optic disc is naturally very bright and should not
    # be interpreted as a lesion candidate.
    # --------------------------------------------------------

    bright_region = (
        brightness > 180
    )

    lesion_mask = (
        lesion_mask
        & (~bright_region)
    )

    # --------------------------------------------------------
    # 10. Final lesion overlay
    # Keep original retina visible.
    # Candidate areas become dark red.
    # --------------------------------------------------------

    result = rgb.copy()

    lesion_pixels = lesion_mask

    result[lesion_pixels, 0] = (
        result[lesion_pixels, 0] * 0.45
        + 120 * 0.55
    )

    result[lesion_pixels, 1] = (
        result[lesion_pixels, 1] * 0.35
    )

    result[lesion_pixels, 2] = (
        result[lesion_pixels, 2] * 0.35
    )

    result = np.clip(
        result,
        0,
        255
    ).astype(np.uint8)

    return Image.fromarray(result)
# ============================================================
# FAST RETINAL IMAGE ENHANCEMENT
# ============================================================
def enhance_retinal_image(image):
    """
    Retinal image enhancement pipeline:
    1. Resize large images
    2. Illumination normalization
    3. CLAHE local contrast enhancement
    4. Mild denoising
    5. Mild sharpening

    Returns an enhanced PIL RGB image.
    """

    # --------------------------------------------------------
    # 1. Convert PIL image to RGB and resize if necessary
    # --------------------------------------------------------

    original = image.convert("RGB")

    max_size = 1200

    if max(original.size) > max_size:

        scale = max_size / max(original.size)

        new_size = (
            int(original.width * scale),
            int(original.height * scale)
        )

        original = original.resize(
            new_size,
            Image.Resampling.LANCZOS
        )

    # PIL RGB -> NumPy
    rgb = np.array(original).astype(np.uint8)

    # --------------------------------------------------------
    # 2. Illumination Normalization
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2GRAY
    )

    # Estimate uneven illumination
    background = cv2.GaussianBlur(
        gray,
        (0, 0),
        sigmaX=25
    )

    # Prevent division by zero
    background = np.maximum(
        background,
        10
    )

    # Normalize illumination
    normalized = (
        gray.astype(np.float32)
        / background.astype(np.float32)
    ) * 128.0

    normalized = np.clip(
        normalized,
        0,
        255
    ).astype(np.uint8)

    # --------------------------------------------------------
    # 3. CLAHE
    # --------------------------------------------------------

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced_gray = clahe.apply(
        normalized
    )

    # --------------------------------------------------------
    # 4. Preserve Retinal Color
    # --------------------------------------------------------

    lab = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2LAB
    )

    # Replace only luminance channel
    lab[:, :, 0] = enhanced_gray

    enhanced = cv2.cvtColor(
        lab,
        cv2.COLOR_LAB2RGB
    )

    # --------------------------------------------------------
    # 5. Mild Denoising
    # --------------------------------------------------------

    enhanced = cv2.fastNlMeansDenoisingColored(
        enhanced,
        None,
        h=3,
        hColor=3,
        templateWindowSize=7,
        searchWindowSize=21
    )

    # --------------------------------------------------------
    # 6. Mild Sharpening
    # --------------------------------------------------------

    blurred = cv2.GaussianBlur(
        enhanced,
        (0, 0),
        sigmaX=1.0
    )

    enhanced = cv2.addWeighted(
        enhanced,
        1.15,
        blurred,
        -0.15,
        0
    )

    # --------------------------------------------------------
    # 7. Final conversion
    # --------------------------------------------------------

    enhanced = np.clip(
        enhanced,
        0,
        255
    ).astype(np.uint8)

    return Image.fromarray(
        enhanced
    )
# ============================================================
# RETINAL STRUCTURE / LESION EVIDENCE
# ============================================================

def analyze_retinal_structures(image):
    """
    Lightweight retinal structure analysis for the prototype.
    Uses NumPy + PIL only. No OpenCV required.
    """

    img = np.array(image.convert("RGB")).astype(np.float32)

    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]

    # --------------------------------------------------------
    # 1. Blood vessel evidence
    # Vessels are generally darker in the green channel.
    # --------------------------------------------------------

    vessel_score = (
        (r - g) * 0.5 +
        (b - g) * 0.2
    )

    vessel_threshold = np.percentile(vessel_score, 88)

    vessel_mask = vessel_score > vessel_threshold

    # Remove very weak isolated pixels
    vessel_mask = vessel_mask.astype(np.uint8) * 255

    vessel_image = Image.fromarray(vessel_mask).convert("L")

    # --------------------------------------------------------
    # 2. Optic disc localization
    # Look for the brightest retinal region.
    # --------------------------------------------------------

    brightness = (
        0.299 * r +
        0.587 * g +
        0.114 * b
    )

    # Ignore black area outside the circular fundus
    valid = brightness > 30

    if np.any(valid):

        threshold = np.percentile(
            brightness[valid],
            99.5
        )

        bright_mask = (
            (brightness >= threshold) &
            valid
        )

        ys, xs = np.where(bright_mask)

        if len(xs) > 0:

            disc_x = int(np.mean(xs))
            disc_y = int(np.mean(ys))

        else:

            disc_x = int(img.shape[1] * 0.25)
            disc_y = int(img.shape[0] * 0.50)

    else:

        disc_x = int(img.shape[1] * 0.25)
        disc_y = int(img.shape[0] * 0.50)

    # --------------------------------------------------------
    # 3. Candidate lesion evidence
    # Dark reddish retinal regions can indicate
    # possible hemorrhage / microaneurysm candidates.
    # This is NOT a clinical diagnosis.
    # --------------------------------------------------------

    lesion_score = (
        r * 0.6
        - g * 0.3
        - b * 0.3
    )

    lesion_threshold = np.percentile(
        lesion_score[valid],
        97
    ) if np.any(valid) else 0

    lesion_mask = (
        (lesion_score >= lesion_threshold) &
        valid
    )

    lesion_image = Image.fromarray(
        (lesion_mask.astype(np.uint8) * 255)
    ).convert("L")

    return {
        "vessel_image": vessel_image,
        "lesion_image": lesion_image,
        "optic_disc_x": disc_x,
        "optic_disc_y": disc_y,
    }
def page_new_scan():
     st.markdown("""
<div class="scan-header">
    <div>
        <div class="scan-title">🔬 New Retinal Scan</div>
        <div class="scan-subtitle">
            AI-assisted diabetic retinopathy screening
        </div>
    </div>
    <div class="scan-status">
        ● AI MODEL READY
    </div>
   </div>
    """, unsafe_allow_html=True)

     st.markdown("""
     <div class="info-strip">
    <div>🩺 <b>Fundus Screening</b><br><span>Upload a retinal image</span></div>
    <div>🤖 <b>AI Analysis</b><br><span>5-class DR classification</span></div>
    <div>⚡ <b>Fast Results</b><br><span>Confidence-based prediction</span></div>
   </div>
   """, unsafe_allow_html=True)
     col_upload, col_result = st.columns([1, 1], gap="large")


     with col_upload:
        st.markdown("""
<div class="section-heading">
    <span class="section-icon">📤</span>
    <div>
        <div class="section-title">Upload Retinal Image</div>
        <div class="section-subtitle">
            JPG, JPEG or PNG • Clear fundus images recommended
        </div>
    </div>
</div>
""", unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Drag & drop or browse",
            type=["jpg", "jpeg", "png"],
            help="Supported formats: JPG, JPEG, PNG",
        )
        if uploaded_file is not None:
            image_bytes = uploaded_file.getvalue()
            image = Image.open(io.BytesIO(image_bytes))
            image.load()

            quality = assess_image_quality(image)
            st.session_state.image_quality = quality

            st.markdown("""
            <div class="image-status">
                <span>✓</span>
                Retinal image uploaded successfully
            </div>
            """, unsafe_allow_html=True)
        
            st.image(
            image,
            caption="Retinal fundus image",
            use_container_width=True
    )

        notes = st.text_area(
        "Clinical notes (optional)",
        placeholder="Add any relevant notes for this scan..."
    )
        
          

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🧠 Analyse Image", type="primary", use_container_width=True):
                        
               
            # --------------------------------------------------------
            # Show quality result
            # --------------------------------------------------------
            
            if quality["status"] == "Good":

                st.success(
                    "🟢 **Image Quality: Good** — "
                    "The retinal image is suitable for AI screening."
                )

            elif quality["status"] == "Borderline":

                st.warning(
                    "🟡 **Image Quality: Borderline** — "
                    "Attempting automatic image enhancement..."
                )

                # ----------------------------------------------------
                # Automatic enhancement for borderline images
                # ----------------------------------------------------

                
                 # --------------------------------------------------------
                # ORIGINAL RETINAL IMAGE
                # --------------------------------------------------------

                st.image(
                    image,
                    caption="Original Retinal Image",
                    use_container_width=True
                )

                # --------------------------------------------------------
                # ENHANCED RETINAL IMAGE
                # CLAHE + Illumination + Denoising + Sharpening
                # --------------------------------------------------------

                enhanced_image = enhance_retinal_image(image)

               

                st.info(
                    "✨ The image has been automatically enhanced using "
                    "contrast enhancement, CLAHE, illumination normalization, "
                    "and denoising."
                )

                
                                # ----------------------------------------------------
                # ORIGINAL vs ENHANCED IMAGE
                # ----------------------------------------------------

                st.subheader("🔍 Original vs Enhanced Retinal Image")

                col1, col2 = st.columns(2)

                with col1:
                    st.image(
                        original_image,
                        caption="Original Retinal Image",
                        use_container_width=True
                    )

                with col2:
                    st.image(
                        enhanced_image,
                        caption="Enhanced Retinal Image",
                        use_container_width=True
                    )
                # ----------------------------------------------------
                # Re-assess image quality after enhancement
                # ----------------------------------------------------

                enhanced_quality = assess_image_quality(
                    enhanced_image
                )

                st.session_state.image_quality = quality

                # ----------------------------------------------------
                # Check enhanced image
                # ----------------------------------------------------

                if enhanced_quality["status"] == "Good":

                    st.success(
                        "🟢 **Enhanced Image Quality: Good** — "
                        "The image is now suitable for AI screening."
                    )

                    # Use enhanced image for the remaining analysis
                    

                else:

                    st.error(
                        "🔴 **Image Quality: Still Inadequate After Enhancement**"
                    )

                    st.write(
                        "Automatic enhancement could not improve the image "
                        "enough for reliable AI analysis."
                    )

                    for reason in enhanced_quality["failed_checks"]:
                        st.write(f"• {reason}")

                    st.info(
                        "📷 **Recapture Required:** Please capture a clearer "
                        "retinal image with better focus, illumination, and "
                        "retinal field of view."
                    )

                    # Stop AI analysis
                    st.stop()

            else:

                st.error(
                    "🔴 **Image Quality: Poor**"
                )

                st.write(
                    "The retinal image cannot be reliably analysed."
                )

                for reason in quality["failed_checks"]:
                    st.write(f"• {reason}")

                st.info(
                    "📷 **Analysis Stopped:** Please capture or upload "
                    "a better retinal image."
                )

                # Stop AI analysis
                st.stop()
            # ========================================================
            # EXISTING AI ANALYSIS
            # ========================================================

            with st.spinner("AI is analysing your retinal scan..."):

                    # Open the image
                    original_image = Image.open(uploaded_file).convert("RGB")

                    enhanced_image = enhance_retinal_image(
                        original_image
                    )

                    st.session_state.original_scan = original_image
                    st.session_state.enhanced_scan = enhanced_image
                    # ============================================================
                    # REAL RETINAL VESSEL EVIDENCE
                    # ============================================================

                    vessel_evidence = generate_vessel_evidence(
                        original_image
                    )

                    st.session_state.vessel_evidence = vessel_evidence
                                        # ============================================================
                    # RETINAL LESION EVIDENCE
                    # ============================================================

                    lesion_evidence = generate_lesion_evidence(
                        original_image
                    )

                    st.session_state.lesion_evidence = lesion_evidence
                      # Show enhancement for verification
                    # ----------------------------------------------------
                    # ORIGINAL vs ENHANCED IMAGE
                    # ----------------------------------------------------

                    st.subheader("🔍 Original vs Enhanced Retinal Image")

                    col1, col2 = st.columns(2)

                    with col1:
                        st.image(
                            original_image,
                            caption="Original Retinal Image",
                            use_container_width=True
                        )

                    with col2:
                        st.image(
                            enhanced_image,
                            caption="Enhanced Retinal Image",
                            use_container_width=True
                        )
                                              
                    
                    image = enhanced_image

                    # Prepare image for the model
                    input_tensor = IMAGE_TRANSFORM(image).unsqueeze(0)

                                        # ----------------------------------------------------
                    # PREPARE ORIGINAL AND ENHANCED IMAGES
                    # ----------------------------------------------------

                    original_input_tensor = IMAGE_TRANSFORM(
                        image
                    ).unsqueeze(0)

                    enhanced_input_tensor = IMAGE_TRANSFORM(
                        enhanced_image
                    ).unsqueeze(0)

                 

                # ----------------------------------------------------
                # AI MODEL PREDICTION
                # Use ENHANCED image for final prediction
                # ----------------------------------------------------

                    with torch.no_grad():

                        outputs = model(enhanced_input_tensor)

                        probabilities = torch.softmax(
                            outputs,
                            dim=1
                        )

                        confidence_value, predicted_class = torch.max(
                            probabilities,
                            dim=1
                        )

                    # All class probabilities
                    class_probabilities = {
                        CLASS_NAMES[i]: round(
                            probabilities[0][i].item() * 100,
                            1
                        )
                        for i in range(len(CLASS_NAMES))
                    }

                    st.session_state.class_probabilities = class_probabilities

                        # All class probabilities
                    class_probabilities = {
                            CLASS_NAMES[i]: round(
                                probabilities[0][i].item() * 100,
                                1
                            )
                            for i in range(len(CLASS_NAMES))
                        }

                    st.session_state.class_probabilities = class_probabilities

                    # ----------------------------------------------------
                    # EXPLAINABLE AI - GRAD-CAM
                    # ----------------------------------------------------

                    try:

                        heatmap = generate_gradcam(
                            model,
                             enhanced_input_tensor,
                            predicted_class.item()
                        )

                        st.session_state.gradcam_heatmap = heatmap
                        st.session_state.original_scan = original_image

                    except Exception:

                        st.session_state.gradcam_heatmap = None
                        st.session_state.original_scan = original_image

                    # ----------------------------------------------------
                    # FINAL PREDICTION
                    # ----------------------------------------------------

                    prediction = CLASS_NAMES[predicted_class.item()]

                    confidence = round(
                        confidence_value.item() * 100,
                        1
                    )
                                        # ============================================================
                    # RETINAL STRUCTURE ANALYSIS
                    # ============================================================

                    structure_result = analyze_retinal_structures(
                        original_image
                    )

                    st.session_state.retinal_structures = structure_result

                    # ----------------------------------------------------
                    # SAVE RESULT
                    # ----------------------------------------------------

                    save_prediction(
                        user_id=st.session_state.user_id,
                        image_path=uploaded_file.name,
                        prediction=prediction,
                        confidence=confidence,
                    )

                    # ----------------------------------------------------
                    # SAVE RESULT TO SESSION
                    # ----------------------------------------------------

                    st.session_state.analysis_result = {
                        "prediction": prediction,
                        "confidence": confidence,
                    }
                                        # ============================================================
                    # RETINAL STRUCTURE EVIDENCE DISPLAY
                    # ============================================================

                    structure = st.session_state.get(
                        "retinal_structures"
                    )

                    if structure:

                        st.markdown("### 🩸 Retinal Structure Evidence")

                        st.info(
                            "AI-assisted structural analysis highlights retinal "
                            "vessel patterns, optic-disc location and candidate "
                            "lesion regions for ophthalmologist review."
                        )

                        col1, col2 = st.columns(2)

                        st.markdown("#### 🩸 Vessel Evidence")

                        if st.session_state.get("vessel_evidence") is not None:

                            st.image(
                                st.session_state.vessel_evidence,
                                caption="Retinal Vessel Evidence — highlighted blood vessels",
                                width=250
                            )

                        st.markdown("#### 🔴 Candidate Lesion Evidence")

                        if st.session_state.get("lesion_evidence") is not None:

                            st.image(
                                st.session_state.lesion_evidence,
                                caption="Candidate Lesion Evidence — highlighted regions",
                                width=250
                            )
                        st.success(
                            f"🔎 Optic disc localized near "
                            f"({structure['optic_disc_x']}, "
                            f"{structure['optic_disc_y']})"
                        )
                         
                        st.caption(
                            "⚠️ Structural highlights are AI-assisted evidence "
                            "and are not a clinical diagnosis."
                        )

                    # Show success
                    st.success(
                        f"🧠 AI Prediction: **{prediction}** "
                        f"with **{confidence}% confidence**"
                    )
                    

     with col_result:
            st.markdown("### 📊 Analysis Result")
            result = st.session_state.get("analysis_result")

            if not result:
                st.markdown("""
                <div style="border:2px dashed #cbd5e1;border-radius:12px;padding:3rem;text-align:center;color:#94a3b8;">
                    <p style="font-size:2rem;">👁️</p>
                    <p>Results will appear here after analysis</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                pred = result["prediction"]
                conf = result["confidence"]

                # Severity colour map
                severity_info = {
                    "No DR":             ("#f0fdf4", "#16a34a", "No signs of diabetic retinopathy detected. Routine follow-up recommended."),
                    "Mild":              ("#fffbeb", "#d97706", "Mild NPDR detected. Schedule follow-up within 12 months."),
                    "Moderate":          ("#fff7ed", "#ea580c", "Moderate NPDR. Refer to ophthalmologist within 6 months."),
                    "Severe":            ("#fef2f2", "#dc2626", "Severe NPDR. Urgent ophthalmology referral needed."),
                    "Proliferative DR":  ("#fef2f2", "#991b1b", "Proliferative DR. Immediate specialist intervention required."),
                }
                bg, accent, advice = severity_info.get(pred, ("#f8fafc", "#475569", "Please consult a specialist."))

                st.markdown(f"""
                <div style="background:{bg};border:2px solid {accent};border-radius:12px;padding:1.5rem;">
                    <h3 style="color:{accent};margin:0 0 0.3rem;">Diagnosis</h3>
                    <p style="font-size:1.8rem;font-weight:700;color:{accent};margin:0;">{pred}</p>
                    <hr style="border-color:{accent};opacity:0.2;margin:0.8rem 0;">
                    <div style="display:flex;justify-content:space-between;">
                        <div>
                            <span style="font-size:0.8rem;color:#64748b;">CONFIDENCE</span><br>
                            <strong style="font-size:1.3rem;">{conf}%</strong>
                        </div>
                        <div style="text-align:right;">
                            <span style="font-size:0.8rem;color:#64748b;">SEVERITY</span><br>
                            {severity_badge(pred)}
                        </div>
                    </div>
                    <hr style="border-color:{accent};opacity:0.2;margin:0.8rem 0;">
                    <p style="font-size:0.88rem;color:#475569;margin:0;">💡 {advice}</p>
                </div>
                """, unsafe_allow_html=True)

                
                # ── Explainable AI ─────────────────────────────────────────────────────
                st.markdown("### 🧠 Explainable AI")

                gradcam_heatmap = st.session_state.get("gradcam_heatmap")
                original_scan = st.session_state.get("original_scan")

                if gradcam_heatmap is not None and original_scan is not None:

                    st.caption(
                        "The highlighted regions show areas that contributed most "
                        "to the AI prediction."
                    )
                                        # ============================================================
                    # QUICK RESULT SUMMARY
                    # ============================================================

                    # Get prediction safely from session state
                    prediction_text = st.session_state.get(
                        "analysis_result", {}
                    ).get(
                        "prediction",
                        "Unknown"
                    )

                    quick_confidence = st.session_state.get(
                        "analysis_result", {}
                    ).get(
                        "confidence",
                        0
                    )

                                        # ============================================================
                        # EXPLAINABLE AI DISPLAY
                        # ============================================================

                    col_original, col_heatmap = st.columns(2, gap="medium")


                    with col_original:
                            st.markdown(
                             "<div style='height:48px; display:flex; align-items:center; "
                                "white-space:nowrap;'>"
                                "<b>Original Retinal Image</b></div>",
                                unsafe_allow_html=True
    )

                            # Create a fixed display copy
                            original_display = original_scan.convert("RGB").copy()

                            st.image(
                                original_display,
                                width=300
                            )

                    with col_heatmap:
                            st.markdown(
                            "<div style='height:48px; display:flex; align-items:center; "
                            "white-space:nowrap;'>"
                            "<b>AI Attention Heatmap</b></div>",
                            unsafe_allow_html=True
    )

                            # --------------------------------------------------------
                            # Prepare Grad-CAM
                            # --------------------------------------------------------
                            heatmap = np.clip(gradcam_heatmap, 0, 1)

                            heatmap_array = (
                                heatmap * 255
                            ).astype("uint8")

                            # Resize Grad-CAM to the retinal image
                            mask = Image.fromarray(
                                heatmap_array
                            ).resize(
                                original_scan.size,
                                Image.Resampling.BILINEAR
                            )

                            # --------------------------------------------------------
                            # Remove very weak attention
                            # --------------------------------------------------------
                            mask_array = np.array(mask)

                            mask_array = np.where(
                                mask_array < 70,
                                0,
                                mask_array
                            ).astype("uint8")

                            mask = Image.fromarray(mask_array)

                            # --------------------------------------------------------
                            # Create red attention layer
                            # --------------------------------------------------------
                            red_layer = Image.new(
                                "RGB",
                                original_scan.size,
                                (255, 0, 0)
                            )

                            # --------------------------------------------------------
                            # Apply attention
                            # --------------------------------------------------------
                            attention_overlay = Image.composite(
                                red_layer,
                                original_scan.convert("RGB"),
                                mask
                            )

                            # --------------------------------------------------------
                            # Blend with original retina
                            # --------------------------------------------------------
                            overlay = Image.blend(
                                original_scan.convert("RGB"),
                                attention_overlay,
                                0.70
                            )

                            st.image(
                                overlay,
                                width=300
                            )
                    prediction_text = st.session_state.get(
                        "analysis_result", {}
                    ).get(
                        "prediction",
                        "the predicted condition"
                    )

                    # --------------------------------------------------------
                    # Class-specific Explainable AI description
                    # --------------------------------------------------------

                    explanation_map = {

                        "No DR": (
                            "The highlighted retinal regions represent areas that "
                            "contributed to the model's No DR prediction. "
                            "No strong model evidence for diabetic retinopathy "
                            "was identified in these highlighted areas."
                        ),

                        "Mild": (
                            "The highlighted retinal regions represent areas that "
                            "contributed to the model's Mild DR prediction. "
                            "These areas contain visual features that influenced "
                            "the model toward early diabetic retinopathy."
                        ),

                        "Moderate": (
                            "The highlighted retinal regions represent areas that "
                            "contributed to the model's Moderate DR prediction. "
                            "These areas contain visual features that influenced "
                            "the model toward moderate diabetic retinopathy."
                        ),

                        "Severe": (
                            "The highlighted retinal regions represent areas that "
                            "contributed to the model's Severe DR prediction. "
                            "These areas contain visual features that influenced "
                            "the model toward advanced diabetic retinopathy."
                        ),

                        "Proliferative DR": (
                            "The highlighted retinal regions represent areas that "
                            "contributed to the model's Proliferative DR prediction. "
                            "These areas contain visual features that influenced "
                            "the model toward advanced proliferative diabetic retinopathy."
                        )
                    }

                    explanation = explanation_map.get(
                        prediction_text,
                        "The highlighted retinal regions represent areas that "
                        "contributed to the model's prediction."
                    )

                    st.caption(
                        f"Highlighted regions indicate areas that contributed "
                        f"most strongly to the **{prediction_text}** prediction."
                    )

                    st.info(
                        f"🧠 **Explainable AI — {prediction_text}**\n\n"
                        f"{explanation}\n\n"
                        "🔴 Red areas represent stronger model attention. "
                        "They should be interpreted as model evidence, "
                        "not as a confirmed clinical diagnosis."
                    )
                   
                else:
                    st.info("Explainable AI heatmap is not available for this scan.")
                        # ============================================================
            # REFERRAL RECOMMENDATION
            # ============================================================

            st.markdown("### 🩺 Referral Recommendation")

            # Get prediction safely
            analysis_result = st.session_state.get("analysis_result") or {}

            prediction_text = analysis_result.get(
                "prediction",
                "Unknown"
            )

            referable_classes = [
                "Moderate",
                "Severe",
                "Proliferate_DR",
                "Proliferative DR"
            ]

            # ------------------------------------------------------------
            # BEFORE AI ANALYSIS
            # ------------------------------------------------------------

            if prediction_text == "Unknown":

                st.info(
                    "⏳ **Awaiting AI Analysis**\n\n"
                    "Please analyse a retinal scan to determine referral status."
                )

            # ------------------------------------------------------------
            # REFERABLE DR
            # ------------------------------------------------------------

            elif prediction_text in referable_classes:

                st.error(
                    f"🔴 **Referable DR**\n\n"
                    f"AI prediction: **{prediction_text}**\n\n"
                    "Level 2 or higher — ophthalmologist evaluation is recommended."
                )

            # ------------------------------------------------------------
            # NON-REFERABLE DR
            # ------------------------------------------------------------

            else:

                st.success(
                    f"🟢 **Non-Referable DR**\n\n"
                    f"AI prediction: **{prediction_text}**\n\n"
                    "Below the Level 2 referral threshold."
                )


            # ============================================================
            # AI CONFIDENCE BREAKDOWN
            # ============================================================

            st.markdown("### 📊 AI Confidence Breakdown")

            class_probabilities = st.session_state.get(
                "class_probabilities",
                {}
            )

            if class_probabilities:

                predicted_class_name = max(
                    class_probabilities,
                    key=class_probabilities.get
                )

                for class_name, probability in class_probabilities.items():

                    if class_name == predicted_class_name:

                        st.markdown(
                            f"**🧠 Predicted: {class_name} — {probability}%**"
                        )

                    else:

                        st.markdown(
                            f"{class_name} — {probability}%"
                        )

                    st.progress(
                        float(probability) / 100
                    )

            else:

                st.info(
                    "AI confidence information is not available."
                )
            # ============================================================
            # ANALYSE ANOTHER BUTTON
            # ============================================================

            st.markdown("<br>", unsafe_allow_html=True)

            if st.button(
                "🔄 Analyse Another",
                use_container_width=True
            ):

                st.session_state.analysis_result = None
                st.session_state.class_probabilities = {}
                st.session_state.gradcam_heatmap = None
                st.rerun()


# ═══════════════════════════════════════════════════════════════════════════
#  PAGE: HISTORY
# ═══════════════════════════════════════════════════════════════════════════
def page_history():
    st.markdown("### 📜 Scan History")
    st.caption("All your past retinal scans and AI diagnoses")

    history = get_user_history(st.session_state.user_id)

    # ── High-risk banner ──
    high_risk_banner(history)

    if not history:
        st.info("No scan history found. Run your first scan to get started.")

        if st.button("🔬 Start a Scan", type="primary"):
            st.session_state.active_page = "New Scan"
            st.rerun()

        return

    # ── Controls: search + filter + sort + export ──
    col_f1, col_f2, col_f3, col_f4 = st.columns([2, 1.2, 1.2, 1])
    with col_f1:
        search = st.text_input("🔍 Search", placeholder="filename or diagnosis")
    with col_f2:
        filter_pred = st.selectbox(
            "Diagnosis",
            ["All", "No DR", "Mild", "Moderate", "Severe", "Proliferative DR"],
        )
    with col_f3:
        sort_by = st.selectbox(
            "Sort by",
            ["Date (newest)", "Date (oldest)", "Confidence ↑", "Confidence ↓", "Severity"],
        )
    with col_f4:
        st.markdown("<br>", unsafe_allow_html=True)
        st.download_button(
            label="📥 Export CSV",
            data=history_to_csv(history),
            file_name="ocuvision_history.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # ── Filter ──
    filtered = [
        r for r in history
        if (filter_pred == "All" or r.get("prediction") == filter_pred)
        and (
            not search
            or search.lower() in (r.get("image_path") or "").lower()
            or search.lower() in (r.get("prediction") or "").lower()
        )
    ]

    # ── Sort ──
    if sort_by == "Date (oldest)":
        filtered = list(reversed(filtered))
    elif sort_by == "Confidence ↑":
        filtered = sorted(filtered, key=lambda r: r.get("confidence") or 0)
    elif sort_by == "Confidence ↓":
        filtered = sorted(filtered, key=lambda r: r.get("confidence") or 0, reverse=True)
    elif sort_by == "Severity":
        filtered = sorted(
            filtered,
            key=lambda r: SEVERITY_ORDER.index(r["prediction"])
            if r.get("prediction") in SEVERITY_ORDER else 99,
            reverse=True,
        )

    st.markdown(f"**{len(filtered)} record(s)** found")
    st.markdown("---")

    for row in filtered:
        pred = row.get("prediction", "Unknown")
        conf = row.get("confidence") or 0
        raw_date = row.get("created_at")
        raw_date = row.get("created_at")
        if raw_date:
            if hasattr(raw_date, 'strftime'):
                date_display = raw_date.strftime("%B %d, %Y — %H:%M")
            else:
                try:
                    date_display = str(raw_date).replace("-", "/")[0:16]
                except:
                    date_display = str(raw_date)
        else:
            date_display = "—"
        fname = row.get("image_path", "—")

        with st.expander(f"🗓️ {date_display}  ·  {pred}  ·  {fname}", expanded=False):
            c1, c2 = st.columns(2)
            c1.markdown(f"**Diagnosis:** {severity_badge(pred)}", unsafe_allow_html=True)
            c1.markdown(f"**Confidence:** `{conf:.1f}%`")
            c2.markdown(f"**File:** `{fname}`")
            c2.markdown(f"**Date:** {date_display}")


# ═══════════════════════════════════════════════════════════════════════════
#  ROUTER
# ════════════════════════════════════
if not st.session_state.logged_in:
    page_login()
else:
    render_sidebar()
    page = st.session_state.active_page
    if page == "Dashboard":
        page_dashboard()
    elif page == "New Scan":
        page_new_scan()
    elif page == "History":
        page_history()