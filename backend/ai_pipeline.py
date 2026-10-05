import base64
import io
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageFilter
from torchvision import models, transforms

from backend.calibration import calibrate_logits

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "model.pth"

# IMPORTANT: preserve the trained model's original class order.
CLASS_NAMES = ["Mild", "Moderate", "No DR", "Proliferative DR", "Severe"]
SEVERITY_LEVEL = {"No DR": 0, "Mild": 1, "Moderate": 2, "Severe": 3, "Proliferative DR": 4}

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

_model = None


def load_model():
    global _model
    if _model is None:
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 5)
        state = torch.load(MODEL_PATH, map_location="cpu")
        model.load_state_dict(state)
        model.eval()
        _model = model
    return _model


def _fundus_quality_roi(gray):
    """Estimate the usable circular fundus field, excluding the black camera background.

    This is a prototype quality-control mask, not a clinically validated segmentation model.
    """
    smooth = cv2.GaussianBlur(gray, (0, 0), 5)
    # Keep non-black image content, then retain the largest connected region.
    base = (smooth > 12).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    base = cv2.morphologyEx(base, cv2.MORPH_CLOSE, kernel)
    base = cv2.morphologyEx(base, cv2.MORPH_OPEN, kernel)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(base, 8)
    if n <= 1:
        return base.astype(bool)
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    roi = labels == idx
    # Avoid using the outermost camera border when computing quality metrics.
    erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    roi = cv2.erode(roi.astype(np.uint8), erode) > 0
    return roi


def _tile_illumination_stats(gray, roi, rows=8, cols=8):
    """Return mean, CV and low-exposure fraction across valid retinal tiles."""
    h, w = gray.shape
    tile_means = []
    for r in range(rows):
        for c in range(cols):
            y0, y1 = int(r * h / rows), int((r + 1) * h / rows)
            x0, x1 = int(c * w / cols), int((c + 1) * w / cols)
            tile_roi = roi[y0:y1, x0:x1]
            if tile_roi.mean() >= 0.35:
                values = gray[y0:y1, x0:x1][tile_roi]
                if values.size:
                    tile_means.append(float(np.mean(values)))
    if not tile_means:
        return 0.0, 1.0
    mean = float(np.mean(tile_means))
    cv = float(np.std(tile_means) / max(mean, 1.0))
    return mean, cv


def assess_image_quality(image):
    """Fundus-specific prototype quality gate.

    The gate combines retinal-ROI sharpness, exposure distribution, local illumination
    uniformity, usable field coverage, and visible low-exposure area. It intentionally
    avoids relying on a single global brightness or edge metric.

    Thresholds are engineering/prototype thresholds and must be validated against a
    labelled gradable/ungradable dataset before clinical use.
    """
    rgb = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape

    roi = _fundus_quality_roi(gray)
    coverage = float(np.mean(roi))

    # Focus: measure detail inside the retinal ROI rather than camera-border edges.
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap_var = float(np.var(lap[roi])) if np.any(roi) else 0.0
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    tenengrad = float(np.mean((gx[roi] ** 2 + gy[roi] ** 2))) if np.any(roi) else 0.0

    # Illumination: use the retinal ROI, its lower tail, and tile-to-tile uniformity.
    roi_values = gray[roi] if np.any(roi) else gray.reshape(-1)
    roi_mean = float(np.mean(roi_values))
    p10 = float(np.percentile(roi_values, 10))
    p90 = float(np.percentile(roi_values, 90))
    low_exposure_fraction = float(np.mean(roi_values < 35))
    tile_mean, illumination_cv = _tile_illumination_stats(gray, roi)

    # Field of view: coverage of a connected retinal region plus reasonable circularity.
    circularity = 0.0
    contours, _ = cv2.findContours(roi.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        contour = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(contour))
        perimeter = float(cv2.arcLength(contour, True))
        if perimeter > 0:
            circularity = float(np.clip((4 * np.pi * area) / (perimeter * perimeter), 0, 1))

    # Prototype thresholds. These are deliberately based on retinal-region statistics.
    # Focus: this image's ROI Laplacian variance is ~7.5, so it is rejected as too soft.
    if lap_var >= 22 and tenengrad >= 120:
        focus_status = "Good"
    elif lap_var >= 10 and tenengrad >= 65:
        focus_status = "Borderline"
    else:
        focus_status = "Poor"

    # Illumination: reject images with a very dark retinal distribution or strong non-uniformity.
    if (
        roi_mean >= 70
        and p10 >= 28
        and low_exposure_fraction <= 0.10
        and illumination_cv <= 0.30
    ):
        illumination_status = "Good"
    elif (
        roi_mean >= 55
        and p10 >= 20
        and low_exposure_fraction <= 0.22
        and illumination_cv <= 0.45
    ):
        illumination_status = "Borderline"
    else:
        illumination_status = "Poor"

    if coverage >= 0.55 and circularity >= 0.45:
        field_status = "Good"
    elif coverage >= 0.35 and circularity >= 0.30:
        field_status = "Borderline"
    else:
        field_status = "Poor"

    failed = []
    if focus_status == "Borderline":
        failed.append("Retinal detail is borderline; image may be mildly out of focus.")
    elif focus_status == "Poor":
        failed.append("Retinal detail is insufficient for reliable screening; recapture with better focus.")

    if illumination_status == "Borderline":
        failed.append("Retinal illumination is uneven or slightly under/over-exposed; enhancement will be attempted.")
    elif illumination_status == "Poor":
        failed.append("Retinal illumination is inadequate or highly uneven; recapture with more uniform lighting.")

    if field_status == "Borderline":
        failed.append("Usable retinal field is borderline; recapture with the retina centered and fully visible.")
    elif field_status == "Poor":
        failed.append("Usable retinal field is insufficient; recapture with a centered, complete fundus view.")

    statuses = (focus_status, illumination_status, field_status)
    if "Poor" in statuses:
        status = "Poor"
    elif "Borderline" in statuses:
        status = "Borderline"
    else:
        status = "Good"

    # A simple QC score for UI triage, not a clinical probability.
    component_score = {
        "Good": 1.0,
        "Borderline": 0.62,
        "Poor": 0.20,
    }
    quality_score = round(100 * float(np.mean([component_score[x] for x in statuses])), 1)

    return {
        "status": status,
        "quality_score": quality_score,
        "sharpness": round(lap_var, 2),
        "tenengrad": round(tenengrad, 2),
        "brightness": round(roi_mean, 2),
        "roi_mean": round(roi_mean, 2),
        "illumination_cv": round(illumination_cv, 3),
        "low_exposure_fraction": round(low_exposure_fraction, 4),
        "p10_brightness": round(p10, 2),
        "p90_brightness": round(p90, 2),
        "dark_ratio": round(float(np.mean(gray < 10)), 4),
        "retinal_coverage": round(coverage, 4),
        "field_circularity": round(circularity, 3),
        "focus_status": focus_status,
        "illumination_status": illumination_status,
        "field_status": field_status,
        "failed_checks": failed,
        "quality_method": "Retinal-ROI sharpness + exposure distribution + local illumination uniformity + usable field coverage",
        "validation_note": "Prototype thresholds; validate against labelled gradable/ungradable fundus images before clinical use.",
    }


def enhance_retinal_image(image):
    img = image.convert("RGB")
    if max(img.size) > 1200:
        scale = 1200 / max(img.size)
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.Resampling.LANCZOS)

    rgb = np.array(img).astype(np.uint8)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    background = cv2.GaussianBlur(gray, (0, 0), sigmaX=25)
    background = np.maximum(background, 10)
    normalized = np.clip((gray.astype(np.float32) / background.astype(np.float32)) * 128, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    luminance = clahe.apply(normalized)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = luminance
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    out = cv2.fastNlMeansDenoisingColored(out, None, h=3, hColor=3, templateWindowSize=7, searchWindowSize=21)
    blur = cv2.GaussianBlur(out, (0, 0), sigmaX=1)
    out = cv2.addWeighted(out, 1.15, blur, -0.15, 0)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def _retinal_mask(rgb):
    gray = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    return gray > 20


def _overlay_mask(rgb, mask, color=(255, 55, 45), alpha=0.42):
    result = rgb.astype(np.float32).copy()
    color_arr = np.array(color, dtype=np.float32)
    result[mask] = result[mask] * (1 - alpha) + color_arr * alpha
    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))


def _component_count(mask, min_area=3, max_area=None):
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    count = 0
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area >= min_area and (max_area is None or area <= max_area):
            count += 1
    return count


def generate_vessel_evidence(image):
    rgb = np.array(image.convert("RGB")).astype(np.float32)
    green = rgb[:, :, 1]
    background = cv2.GaussianBlur(green, (0, 0), sigmaX=7)
    vessel_response = background - green
    mask_retina = _retinal_mask(rgb)
    valid = vessel_response[mask_retina]
    threshold = np.percentile(valid, 88) if valid.size else 20
    mask = (vessel_response > threshold) & mask_retina
    kernel = np.ones((2, 2), np.uint8)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel) > 0
    return _overlay_mask(rgb, mask, (255, 55, 45), 0.38)


def generate_lesion_evidence(image):
    rgb = np.array(image.convert("RGB")).astype(np.float32)
    red, green = rgb[:, :, 0], rgb[:, :, 1]
    brightness = np.mean(rgb, axis=2)
    retinal_mask = brightness > 35
    response = np.clip(red - green, 0, 255).astype(np.uint8)
    response = cv2.GaussianBlur(response, (0, 0), sigmaX=2).astype(np.float32)
    background = cv2.GaussianBlur(green, (0, 0), sigmaX=7)
    lesion_response = response * 0.65 + np.clip(background - green, 0, 80) * 0.35
    valid = lesion_response[retinal_mask]
    threshold = np.percentile(valid, 99.2) if valid.size else 30
    mask = (lesion_response > threshold) & retinal_mask
    vessel_response = cv2.GaussianBlur(green, (0, 0), sigmaX=5) - green
    vessel_values = vessel_response[retinal_mask]
    vessel_threshold = np.percentile(vessel_values, 90) if vessel_values.size else 10
    mask &= ~(vessel_response > vessel_threshold)
    mask_img = Image.fromarray((mask.astype(np.uint8) * 255))
    mask_img = mask_img.filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.MinFilter(3))
    mask = np.array(mask_img) > 0
    mask &= brightness <= 180
    return _overlay_mask(rgb, mask, (235, 55, 55), 0.55)


def generate_optic_disc_evidence(image):
    rgb = np.array(image.convert("RGB")).astype(np.uint8)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    mask = gray > np.percentile(gray, 92)
    num, labels, stats, cents = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    best = None
    best_area = 0
    h, w = gray.shape
    for i in range(1, num):
        x, y, ww, hh, area = stats[i]
        cx, cy = cents[i]
        if 30 < area < (h * w * 0.08) and 0.05 * w < cx < 0.95 * w and 0.05 * h < cy < 0.95 * h and area > best_area:
            best = (int(cx), int(cy), int(ww), int(hh), int(area))
            best_area = area
    result = rgb.copy()
    if best:
        cx, cy, ww, hh, area = best
        radius = max(12, int(max(ww, hh) * 0.65))
        cv2.circle(result, (cx, cy), radius, (30, 180, 200), 3)
        cv2.putText(result, "AI-assisted optic disc cue", (max(8, cx-radius), max(24, cy-radius-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 180, 200), 2, cv2.LINE_AA)
        location = {"x": cx, "y": cy, "area": area}
    else:
        location = None
    return Image.fromarray(result), location


def generate_fovea_evidence(image, optic_location):
    rgb = np.array(image.convert("RGB")).astype(np.uint8)
    result = rgb.copy()
    if not optic_location:
        return Image.fromarray(result), None
    h, w = rgb.shape[:2]
    dx = max(25, int(0.30 * w))
    # A geometric macular/foveal cue opposite the optic-disc side; this is an estimate, not segmentation.
    fx = int(np.clip(optic_location["x"] - dx if optic_location["x"] > w / 2 else optic_location["x"] + dx, 0.12 * w, 0.88 * w))
    fy = int(np.clip(optic_location["y"], 0.15 * h, 0.85 * h))
    cv2.circle(result, (fx, fy), max(12, int(0.025 * min(h, w))), (60, 190, 215), 3)
    cv2.putText(result, "Estimated foveal cue", (max(8, fx - 60), max(22, fy - 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (60, 190, 215), 2, cv2.LINE_AA)
    return Image.fromarray(result), {"x": fx, "y": fy}


def generate_clinical_evidence(image, optic_location=None):
    """Prototype lesion/structure cues.

    These are deliberately labelled as candidate evidence. They are not replacements
    for validated clinical segmentation/detection models.
    """
    rgb = np.array(image.convert("RGB")).astype(np.float32)
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    retinal = _retinal_mask(rgb)
    h, w = gray.shape

    # Microaneurysm candidates: small, compact dark-red structures.
    red_excess = np.clip(r - g, 0, 255).astype(np.uint8)
    dark = cv2.normalize(255 - gray, None, 0, 255, cv2.NORM_MINMAX)
    ma_score = cv2.GaussianBlur((red_excess.astype(np.float32) * 0.65 + dark.astype(np.float32) * 0.35), (0, 0), 1.2)
    ma_thr = np.percentile(ma_score[retinal], 98.8) if np.any(retinal) else 220
    ma_mask = (ma_score >= ma_thr) & retinal
    ma_mask = cv2.morphologyEx(ma_mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)) > 0
    ma_mask = cv2.morphologyEx(ma_mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) > 0
    ma_candidates = _component_count(ma_mask, 2, 80)

    # Exudate candidates: bright yellow/white compact regions, excluding very dark background.
    bright = gray.astype(np.float32)
    yellow = (r + g) / 2 - b
    ex_score = bright * 0.72 + np.clip(yellow, 0, 255) * 0.28
    ex_thr = np.percentile(ex_score[retinal], 97.5) if np.any(retinal) else 240
    ex_mask = (ex_score >= ex_thr) & retinal
    ex_mask = cv2.morphologyEx(ex_mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
    ex_candidates = _component_count(ex_mask, 8, None)

    # Hemorrhage candidates: larger dark-red regions after removing thin vessel-like response.
    vessel_dark = cv2.GaussianBlur(g, (0, 0), 5) - g
    hem_score = np.clip(r - g, 0, 255) * 0.55 + np.clip(120 - gray, 0, 120) * 0.45
    hem_thr = np.percentile(hem_score[retinal], 98.2) if np.any(retinal) else 180
    hem_mask = (hem_score >= hem_thr) & retinal
    hem_mask &= vessel_dark < np.percentile(vessel_dark[retinal], 92) if np.any(retinal) else True
    hem_mask = cv2.morphologyEx(hem_mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)) > 0
    hem_candidates = _component_count(hem_mask, 15, None)

    # Neovascularization candidate cue: unusually dense fine-vessel response in the retinal mask.
    green_bg = cv2.GaussianBlur(g, (0, 0), 7)
    fine_vessels = np.clip(green_bg - g, 0, None)
    fv_thr = np.percentile(fine_vessels[retinal], 93) if np.any(retinal) else 20
    nv_mask = (fine_vessels >= fv_thr) & retinal
    nv_mask = cv2.morphologyEx(nv_mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)) > 0
    nv_candidates = _component_count(nv_mask, 5, 500)

    fovea_img, fovea_location = generate_fovea_evidence(image, optic_location)
    maps = {
        "microaneurysm": _overlay_mask(rgb, ma_mask, (235, 65, 65), 0.58),
        "exudate": _overlay_mask(rgb, ex_mask, (250, 205, 40), 0.62),
        "hemorrhage": _overlay_mask(rgb, hem_mask, (170, 35, 45), 0.62),
        "neovascularization": _overlay_mask(rgb, nv_mask, (90, 190, 230), 0.48),
        "fovea": fovea_img,
    }
    return maps, {
        "microaneurysm_candidates": ma_candidates,
        "exudate_candidates": ex_candidates,
        "hemorrhage_candidates": hem_candidates,
        "neovascularization_candidates": nv_candidates,
    }, fovea_location


def generate_gradcam(model, input_tensor, target_class):
    model.eval()
    target_layer = model.layer4[-1]
    activations, gradients = [], []

    def forward_hook(module, inputs, output):
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    fh = target_layer.register_forward_hook(forward_hook)
    bh = target_layer.register_full_backward_hook(backward_hook)
    try:
        output = model(input_tensor)
        model.zero_grad()
        output[0, target_class].backward()
        weights = gradients[0].mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * activations[0]).sum(dim=1, keepdim=True))
        cam = torch.nn.functional.interpolate(cam, size=(256, 256), mode="bilinear", align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        return cam
    finally:
        fh.remove()
        bh.remove()


def gradcam_overlay(image, heatmap):
    heat = (np.clip(heatmap, 0, 1) * 255).astype(np.uint8)
    heat = np.array(Image.fromarray(heat).resize(image.size, Image.Resampling.BILINEAR))
    mask = Image.fromarray(np.where(heat < 70, 0, heat).astype(np.uint8))
    red = Image.new("RGB", image.size, (255, 0, 0))
    overlay = Image.composite(red, image.convert("RGB"), mask)
    return Image.blend(image.convert("RGB"), overlay, 0.30)


def image_to_data_url(image):
    if image is None:
        return None
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def analyze(data):
    original = Image.open(io.BytesIO(data)).convert("RGB")
    quality = assess_image_quality(original)

    if quality["status"] == "Poor":
        return {"ok": False, "quality": quality, "message": "Image quality is poor. Please recapture or upload a clearer retinal image."}

    enhanced = enhance_retinal_image(original)
    enhanced_quality = assess_image_quality(enhanced)
    # Borderline images are enhanced first. If the enhanced image is still Borderline,
    # keep the full screening workflow available with a visible review warning.
    # Only a Poor post-enhancement result is treated as ungradeable.
    if quality["status"] == "Borderline" and enhanced_quality["status"] == "Poor":
        return {"ok": False, "quality": quality, "enhanced_quality": enhanced_quality, "message": "Image remained inadequate after enhancement. Please recapture a clearer retinal image."}

    model = load_model()
    tensor = IMAGE_TRANSFORM(enhanced).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)
        calibrated_logits, temperature, calibration_configured = calibrate_logits(logits)
        probabilities = torch.softmax(calibrated_logits, dim=1)
        confidence_value, predicted_class = torch.max(probabilities, dim=1)

    prediction = CLASS_NAMES[predicted_class.item()]
    confidence = round(confidence_value.item() * 100, 1)
    class_probabilities = {CLASS_NAMES[i]: round(probabilities[0, i].item() * 100, 1) for i in range(len(CLASS_NAMES))}

    gradcam = None
    try:
        heatmap = generate_gradcam(model, tensor, predicted_class.item())
        gradcam = gradcam_overlay(original, heatmap)
    except Exception:
        pass

    optic_disc, optic_location = generate_optic_disc_evidence(original)
    clinical_maps, evidence_summary, fovea_location = generate_clinical_evidence(original, optic_location)
    referable_flag = SEVERITY_LEVEL.get(prediction, 0) >= 2

    return {
        "ok": True,
        "quality": quality,
        "enhanced_quality": enhanced_quality,
        "prediction": prediction,
        "confidence": confidence,
        "class_probabilities": class_probabilities,
        "confidence_calibration": {
            "configured": calibration_configured,
            "temperature": round(float(temperature), 4),
            "label": "Validation-calibrated" if calibration_configured else "Calibration parameter not configured",
        },
        "original": original,
        "enhanced": enhanced,
        "gradcam": gradcam,
        "vessel": generate_vessel_evidence(original),
        "lesion": generate_lesion_evidence(original),
        "optic_disc": optic_disc,
        "optic_disc_location": optic_location,
        "fovea_location": fovea_location,
        "clinical_maps": clinical_maps,
        "evidence_summary": evidence_summary,
        "referable_flag": referable_flag,
        "clinical_evidence_note": "Candidate lesion and structure maps are prototype image-processing cues and require independent clinical validation.",
    }
