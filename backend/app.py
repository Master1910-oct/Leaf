"""
AgriAI LS-HGNet Inference Server
================================
Lightweight Flask backend serving predictions from the frozen M7.3 LS-HGNet checkpoint.
Checkpoint SHA-256: 4E4145378F2DE852AB7EABE1E72FE7AE7AFA31F608E0F0E246E06C41A7F098A8
Total parameters: 8,041,354
"""

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import io
import time
import hashlib
import base64
from pathlib import Path
from PIL import Image

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from flask import Flask, request, jsonify, send_from_directory

# -----------------------------------------------------------------------------
# Configuration & Verification
# -----------------------------------------------------------------------------
PROJECT_ROOT    = Path("D:/Leaf")
CHECKPOINT_PATH = PROJECT_ROOT / "results/m7_3_full_training/checkpoints/best_model.pt"
EXPECTED_SHA256 = "4E4145378F2DE852AB7EABE1E72FE7AE7AFA31F608E0F0E246E06C41A7F098A8"

FRONTEND_DIR = PROJECT_ROOT / "frontend"

sys.path.insert(0, str(PROJECT_ROOT))
from src.models.ls_hgnet import LS_HGNet
from src.evaluation.gradcam import GradCAM, overlay_cam_on_image

# Authoritative class catalog
CLASS_CATALOG = [
    {
        "class_idx": 0,
        "class_name": "Tomato___Bacterial_spot",
        "display_name": "Tomato — Bacterial Spot",
        "is_healthy": False,
        "description": "Bacterial foliar infection caused by Xanthomonas species, producing small dark water-soaked lesions with yellow halos.",
        "recommendation": "Prune severely affected lower foliage to increase air circulation. Avoid overhead sprinkler irrigation to minimize water splash transmission. Clean and sanitize pruning tools between plants."
    },
    {
        "class_idx": 1,
        "class_name": "Tomato___Early_blight",
        "display_name": "Tomato — Early Blight",
        "is_healthy": False,
        "description": "Fungal infection caused by Alternaria solani, producing brown-to-black necrotic lesions with characteristic concentric target rings.",
        "recommendation": "Remove lower leaves touching soil. Apply organic mulch around plant bases to prevent soil-borne spore splash. Maintain adequate plant spacing and consider protective copper-based bio-fungicide if spread persists."
    },
    {
        "class_idx": 2,
        "class_name": "Tomato___Late_blight",
        "display_name": "Tomato — Late Blight",
        "is_healthy": False,
        "description": "Aggressive oomycete infection caused by Phytophthora infestans, causing rapidly expanding pale-to-dark water-soaked lesions with fuzzy margins.",
        "recommendation": "Promptly isolate and dispose of severely infected foliage in sealed bags (do not compost). Reduce canopy humidity, irrigate only at soil level, and consult local extension services for regional blight advisories."
    },
    {
        "class_idx": 3,
        "class_name": "Tomato___Leaf_Mold",
        "display_name": "Tomato — Leaf Mold",
        "is_healthy": False,
        "description": "Fungal infection caused by Passalora fulva, exhibiting pale green/yellow spots on upper leaf surfaces and olive-brown velvety mold on undersides.",
        "recommendation": "Lower greenhouse relative humidity below 85% by increasing ventilation. Water early in the day directly at the root zone. Thin dense foliage to promote cross-canopy airflow."
    },
    {
        "class_idx": 4,
        "class_name": "Tomato___Septoria_leaf_spot",
        "display_name": "Tomato — Septoria Leaf Spot",
        "is_healthy": False,
        "description": "Fungal disease caused by Septoria lycopersici, featuring numerous small circular spots with ash-gray centers and dark brown margins.",
        "recommendation": "Remove affected lower leaves as soon as spotted. Avoid working in the field when foliage is wet. Rotate solanaceous crops on a 2-3 year cycle and mulch around the base to suppress spore splash."
    },
    {
        "class_idx": 5,
        "class_name": "Tomato___Spider_mites Two-spotted_spider_mite",
        "display_name": "Tomato — Two-Spotted Spider Mites",
        "is_healthy": False,
        "description": "Pest infestation by Tetranychus urticae, causing tiny yellow chlorotic stippling, speckled leaf bronzing, and fine silken webbing on undersides.",
        "recommendation": "Inspect leaf undersides with a hand lens. Spray affected areas with insecticidal soap or horticultural neem oil. In greenhouses, introduce beneficial predatory mites (Phytoseiulus persimilis) to control populations."
    },
    {
        "class_idx": 6,
        "class_name": "Tomato___Target_Spot",
        "display_name": "Tomato — Target Spot",
        "is_healthy": False,
        "description": "Fungal disease caused by Corynespora cassiicola, forming brown lesions with noticeable concentric rings and chlorotic yellow halos.",
        "recommendation": "Improve row spacing to allow rapid leaf drying. Discard diseased crop residues after harvest. Avoid overhead irrigation and apply preventive bio-fungicides if warm, humid conditions persist."
    },
    {
        "class_idx": 7,
        "class_name": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
        "display_name": "Tomato — Yellow Leaf Curl Virus (TYLCV)",
        "is_healthy": False,
        "description": "Viral pathogen transmitted by whiteflies (Bemisia tabaci), causing pronounced upward leaf cupping, stunted bushy growth, and interveinal chlorosis.",
        "recommendation": "Control the whitefly vector using yellow sticky traps and fine insect exclusion netting. Immediately rogue out and bag infected plants to prevent vectors from spreading virus to healthy rows."
    },
    {
        "class_idx": 8,
        "class_name": "Tomato___Tomato_mosaic_virus",
        "display_name": "Tomato — Mosaic Virus (ToMV)",
        "is_healthy": False,
        "description": "Highly contagious plant virus causing light and dark green mottled mosaic patterns, leaf distortion, and stunted growth.",
        "recommendation": "Disinfect pruning shears and hands with 20% non-fat dry milk solution or soap before handling plants. Remove infected plants immediately. Avoid handling tobacco products near crops."
    },
    {
        "class_idx": 9,
        "class_name": "Tomato___healthy",
        "display_name": "Tomato — Healthy Foliage",
        "is_healthy": True,
        "description": "Normal, healthy tomato leaf tissue showing uniform green pigmentation, intact cellular margins, and absence of visual pathogen lesions.",
        "recommendation": "No pathological treatment needed. Maintain balanced N-P-K nutrient feeding, consistent drip irrigation, and routine scouting to preserve vigorous plant immunity."
    }
]

# -----------------------------------------------------------------------------
# Checkpoint Verification & Model Initialization
# -----------------------------------------------------------------------------
def verify_and_load_model():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {CHECKPOINT_PATH}")

    # Compute SHA-256
    h = hashlib.sha256()
    with open(CHECKPOINT_PATH, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    actual_hash = h.hexdigest().upper()

    if actual_hash != EXPECTED_SHA256:
        raise RuntimeError(
            f"Checkpoint SHA-256 mismatch!\nExpected: {EXPECTED_SHA256}\nActual:   {actual_hash}"
        )

    print(f"[Model Server] Checkpoint SHA-256 verified: {actual_hash}")

    # Construct LS-HGNet
    model = LS_HGNet(
        in_channels=3,
        num_classes=10,
        base_channels=64,
        num_stacks=3,
        encoder_levels=4,
        dilation_rates=(1, 2, 4),
        dropout=0.30
    )

    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    raw_sd = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(raw_sd, strict=True)
    model.eval()

    # Freeze all parameters
    for p in model.parameters():
        p.requires_grad = False

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Model Server] Loaded LS-HGNet successfully. Total parameters: {total_params:,}")
    return model, actual_hash

print("[Model Server] Initializing model...")
model, VERIFIED_SHA = verify_and_load_model()

# Initialize Grad-CAM Explainability Engine
try:
    cam_engine = GradCAM(model, "stacked_hourglass.stacks.2.decoder_blocks.3")
    print("[Model Server] Grad-CAM engine initialized on target decoder layer.")
except Exception as _cam_init_err:
    cam_engine = None
    print(f"[Model Server] Grad-CAM initialization deferred: {_cam_init_err}")

# Deterministic Preprocessing matching M9 evaluation
EVAL_TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# -----------------------------------------------------------------------------
# Flask Application Setup
# -----------------------------------------------------------------------------
app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # Max 16 MB upload

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}

def is_allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "service": "AgriAI LS-HGNet Inference API",
        "model": "LS-HGNet",
        "parameters": 8041354,
        "checkpoint_sha256": VERIFIED_SHA,
        "num_classes": 10,
        "input_resolution": "224x224"
    })

@app.route("/debug/gradcam", methods=["GET"])
def debug_gradcam():
    """Diagnostic endpoint — runs Grad-CAM on a synthetic tensor to verify it works server-side."""
    if cam_engine is None:
        return jsonify({"ok": False, "error": "cam_engine is None"})
    try:
        synthetic = torch.zeros(1, 3, 224, 224)
        for p in model.parameters():
            p.requires_grad = True
        with torch.enable_grad():
            t_in = synthetic.clone().detach().requires_grad_(True)
            cam_map, pred_cls, conf, diag = cam_engine.generate_cam(t_in, target_class=0)
        for p in model.parameters():
            p.requires_grad = False
        return jsonify({
            "ok": True,
            "cam_shape": list(cam_map.shape),
            "cam_min": float(cam_map.min()),
            "cam_max": float(cam_map.max()),
            "gradient_norm": diag.get("gradient_norm"),
            "is_grad_enabled_outside": torch.is_grad_enabled()
        })
    except Exception as e:
        import traceback
        for p in model.parameters():
            p.requires_grad = False
        return jsonify({"ok": False, "error": str(e), "trace": traceback.format_exc()})

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({
            "success": False,
            "error": "No image file provided in request. Please upload a leaf image using key 'image'."
        }), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({
            "success": False,
            "error": "No image selected. Please select an image file to upload."
        }), 400

    if not is_allowed_file(file.filename):
        return jsonify({
            "success": False,
            "error": f"Unsupported file type. Allowed formats: {', '.join(sorted(ALLOWED_EXTENSIONS))}."
        }), 400

    try:
        # Read and open image safely
        img_bytes = file.read()
        if len(img_bytes) == 0:
            return jsonify({
                "success": False,
                "error": "The uploaded file is empty."
            }), 400

        try:
            img = Image.open(io.BytesIO(img_bytes))
            img.verify()  # Verify image integrity
            # Reopen after verify
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        except Exception:
            return jsonify({
                "success": False,
                "error": "Corrupted or invalid image file. Please upload a clear leaf image in JPG or PNG format."
            }), 400

        orig_w, orig_h = img.size

        # Simple suitability heuristic (detect tiny thumbnails)
        quality_warning = None
        if orig_w < 50 or orig_h < 50:
            quality_warning = "Image resolution is very low (< 50px). Model prediction may have reduced accuracy."

        # Preprocessing
        tensor_img = EVAL_TRANSFORM(img).unsqueeze(0)  # Shape: [1, 3, 224, 224]

        # Model Inference with precise timing
        t0 = time.perf_counter()
        with torch.no_grad():
            logits = model(tensor_img)  # Shape: [1, 10]
            probs  = F.softmax(logits, dim=1).squeeze(0)  # Shape: [10]
        inference_latency_ms = (time.perf_counter() - t0) * 1000.0

        # Extract top-3 predictions
        top_k_probs, top_k_indices = torch.topk(probs, k=3)
        top_k_probs   = top_k_probs.tolist()
        top_k_indices = top_k_indices.tolist()

        top_predictions = []
        for rank, (idx, prob) in enumerate(zip(top_k_indices, top_k_probs), start=1):
            meta = CLASS_CATALOG[idx]
            top_predictions.append({
                "rank": rank,
                "class_idx": idx,
                "class_name": meta["class_name"],
                "display_name": meta["display_name"],
                "confidence": round(prob, 4),
                "confidence_pct": f"{prob * 100:.1f}%",
                "is_healthy": meta["is_healthy"]
            })

        primary = top_predictions[0]
        primary_meta = CLASS_CATALOG[primary["class_idx"]]

        # Grad-CAM Visual Attribution Heatmap (M8.1 Integration)
        gradcam_data_uri = None
        if cam_engine is not None:
            try:
                # Re-enable gradients on model parameters for backward pass
                for p in model.parameters():
                    p.requires_grad = True

                # Explicitly exit any lingering no_grad context and enable grad computation
                with torch.enable_grad():
                    tensor_cam = tensor_img.clone().detach().requires_grad_(True)
                    cam_map, _, _, _ = cam_engine.generate_cam(tensor_cam, target_class=primary["class_idx"])

                img_224 = img.resize((224, 224))
                overlay_arr = overlay_cam_on_image(img_224, cam_map, alpha=0.55)
                overlay_pil = Image.fromarray((overlay_arr * 255).astype("uint8"))
                buf = io.BytesIO()
                overlay_pil.save(buf, format="JPEG", quality=85)
                gradcam_data_uri = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")
                print(f"[Grad-CAM] OK — URI length: {len(gradcam_data_uri)}", flush=True)
            except Exception as _cam_err:
                import traceback
                print(f"[Grad-CAM] ERROR: {_cam_err}", flush=True)
                traceback.print_exc()
            finally:
                # Always restore frozen state
                for p in model.parameters():
                    p.requires_grad = False

        response_data = {
            "success": True,
            "filename": file.filename,
            "original_resolution": f"{orig_w}x{orig_h}",
            "quality_warning": quality_warning,
            "inference_latency_ms": round(inference_latency_ms, 2),
            "prediction": {
                "class_idx": primary["class_idx"],
                "class_name": primary["class_name"],
                "display_name": primary["display_name"],
                "confidence": primary["confidence"],
                "confidence_pct": primary["confidence_pct"],
                "is_healthy": primary["is_healthy"],
                "description": primary_meta["description"],
                "recommendation": primary_meta["recommendation"]
            },
            "top_predictions": top_predictions,
            "gradcam_image": gradcam_data_uri,
            "model_metadata": {
                "architecture": "LS-HGNet",
                "parameters": 8041354,
                "input_size": "224x224",
                "checkpoint_sha256": VERIFIED_SHA[:16] + "..."
            }
        }

        return jsonify(response_data)

    except Exception as e:
        app.logger.error(f"Inference error: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": "An internal error occurred during leaf analysis. Please try again with a valid image."
        }), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n[Model Server] Starting AgriAI Inference Server on http://127.0.0.1:{port}")
    print("[Model Server] Frontend available at: http://127.0.0.1:5000\n")
    app.run(host="127.0.0.1", port=port, debug=False)
