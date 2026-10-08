/**
 * AgriAI Vision -- Frontend Controller
 * =====================================
 * Handles: drag-and-drop, file selection, image preview,
 * AJAX POST to /predict, and result card rendering.
 *
 * Backend endpoint: POST /predict  (multipart/form-data, field: "image")
 */

"use strict";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
const API_PREDICT_URL = "/predict";
const API_HEALTH_URL  = "/health";

// Confidence thresholds for colour-coding bars
const CONF_HIGH   = 0.80;
const CONF_MEDIUM = 0.50;

// Rank badge accent colours (1st, 2nd, 3rd)
const RANK_COLORS = ["#22c55e", "#f59e0b", "#64748b"];

// ---------------------------------------------------------------------------
// DOM References
// ---------------------------------------------------------------------------
const dropzone               = document.getElementById("dropzone");
const fileInput              = document.getElementById("fileInput");
const dropzoneIdle           = document.getElementById("dropzoneIdle");
const dropzonePreview        = document.getElementById("dropzonePreview");
const previewImage           = document.getElementById("previewImage");
const previewMeta            = document.getElementById("previewMeta");
const btnReplaceImage        = document.getElementById("btnReplaceImage");
const btnRemoveImage         = document.getElementById("btnRemoveImage");
const btnAnalyze             = document.getElementById("btnAnalyze");
const analyzeSpinner         = document.getElementById("analyzeSpinner");
const analyzeBtnText         = document.getElementById("analyzeBtnText");
const notificationBanner     = document.getElementById("notificationBanner");
const notificationMessage    = document.getElementById("notificationMessage");
const notificationClose      = document.getElementById("notificationClose");
const backendStatusBadge     = document.getElementById("backendStatusBadge");
const resultPlaceholder      = document.getElementById("resultPlaceholder");
const resultContent          = document.getElementById("resultContent");
const diagnosisBanner        = document.getElementById("diagnosisBanner");
const statusPill             = document.getElementById("statusPill");
const latencyPill            = document.getElementById("latencyPill");
const diagnosisName          = document.getElementById("diagnosisName");
const diagnosisDesc          = document.getElementById("diagnosisDesc");
const primaryConfidenceValue = document.getElementById("primaryConfidenceValue");
const primaryConfidenceBar   = document.getElementById("primaryConfidenceBar");
const qualityAlert           = document.getElementById("qualityAlert");
const qualityAlertText       = document.getElementById("qualityAlertText");
const confidenceTierBadge    = document.getElementById("confidenceTierBadge");
const topPredictionsList     = document.getElementById("topPredictionsList");
const gradcamPanel           = document.getElementById("gradcamPanel");
const gradcamImage           = document.getElementById("gradcamImage");
const recommendationText     = document.getElementById("recommendationText");
const sampleButtons          = document.getElementById("sampleButtons");

// ---------------------------------------------------------------------------
// Application State
// ---------------------------------------------------------------------------
let currentFile = null;

// ---------------------------------------------------------------------------
// Backend Health Check
// ---------------------------------------------------------------------------
async function checkBackendHealth() {
    try {
        const res = await fetch(API_HEALTH_URL, { method: "GET", cache: "no-store" });
        setBackendStatus(res.ok);
    } catch (_) {
        setBackendStatus(false);
    }
}

function setBackendStatus(online) {
    if (online) {
        backendStatusBadge.innerHTML = '<span class="badge-dot pulse"></span> Model Online';
        backendStatusBadge.classList.remove("badge-offline");
    } else {
        backendStatusBadge.innerHTML = '<span class="badge-dot offline"></span> Model Offline';
        backendStatusBadge.classList.add("badge-offline");
    }
}

// ---------------------------------------------------------------------------
// Notification Banner
// ---------------------------------------------------------------------------
function showNotification(message, type = "error") {
    notificationMessage.textContent = message;
    notificationBanner.classList.remove("hidden", "banner-info", "banner-success", "banner-error");
    notificationBanner.classList.add("banner-" + type);
    clearTimeout(notificationBanner._hideTimer);
    notificationBanner._hideTimer = setTimeout(hideNotification, 8000);
}

function hideNotification() {
    notificationBanner.classList.add("hidden");
}

notificationClose.addEventListener("click", hideNotification);

// ---------------------------------------------------------------------------
// File Handling
// ---------------------------------------------------------------------------
function handleFile(file) {
    if (!file) return;
    // Accept image/* broadly; backend enforces jpg/png/webp
    if (!file.type.startsWith("image/")) {
        showNotification("Unsupported file. Please upload a JPG, PNG, or WEBP image.");
        return;
    }
    if (file.size > 16 * 1024 * 1024) {
        showNotification("File exceeds 16 MB limit. Please use a smaller image.");
        return;
    }
    currentFile = file;
    hideNotification();
    renderPreview(file);
    btnAnalyze.disabled = false;
    showPlaceholder();           // reset result pane until analysis runs
}

function renderPreview(file) {
    const url = URL.createObjectURL(file);
    previewImage.src = url;
    previewImage.onload = () => {
        const w = previewImage.naturalWidth;
        const h = previewImage.naturalHeight;
        previewMeta.textContent = `${w} \u00d7 ${h} px  \u2022  ${formatBytes(file.size)}`;
        URL.revokeObjectURL(url);
    };
    dropzoneIdle.classList.add("hidden");
    dropzonePreview.classList.remove("hidden");
}

function clearFile() {
    currentFile = null;
    fileInput.value = "";
    previewImage.src = "";
    previewMeta.textContent = "";
    dropzonePreview.classList.add("hidden");
    dropzoneIdle.classList.remove("hidden");
    btnAnalyze.disabled = true;
    showPlaceholder();
    hideNotification();
    // Deactivate sample buttons
    sampleButtons.querySelectorAll(".sample-btn").forEach(b => b.classList.remove("active"));
}

function formatBytes(bytes) {
    if (bytes < 1024)    return bytes + " B";
    if (bytes < 1048576) return (bytes / 1024).toFixed(1) + " KB";
    return (bytes / 1048576).toFixed(1) + " MB";
}

// ---------------------------------------------------------------------------
// Dropzone Events
// ---------------------------------------------------------------------------
dropzone.addEventListener("click", e => {
    if (e.target.closest("button")) return;
    fileInput.click();
});

dropzone.addEventListener("keydown", e => {
    if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        fileInput.click();
    }
});

fileInput.addEventListener("change", () => {
    if (fileInput.files && fileInput.files[0]) handleFile(fileInput.files[0]);
});

dropzone.addEventListener("dragover", e => {
    e.preventDefault();
    dropzone.classList.add("drag-over");
});

dropzone.addEventListener("dragleave", e => {
    if (!dropzone.contains(e.relatedTarget)) dropzone.classList.remove("drag-over");
});

dropzone.addEventListener("drop", e => {
    e.preventDefault();
    dropzone.classList.remove("drag-over");
    if (e.dataTransfer.files && e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
});

btnReplaceImage.addEventListener("click", () => fileInput.click());
btnRemoveImage.addEventListener("click", clearFile);

// ---------------------------------------------------------------------------
// Sample Leaf Generator  (synthetic SVG -> PNG for demo purposes)
// ---------------------------------------------------------------------------
const SAMPLE_PALETTES = {
    early_blight: { leaf: "#5a3e10", spot: "#7c2d12", bg: "#fef3c7", vein: "#ffe580" },
    septoria:     { leaf: "#14532d", spot: "#ea580c", bg: "#dcfce7", vein: "#bbf7d0" },
    spider_mites: { leaf: "#166534", spot: "#dc2626", bg: "#f0fdf4", vein: "#86efac" },
    healthy:      { leaf: "#15803d", spot: "#16a34a", bg: "#dcfce7", vein: "#bbf7d0" }
};

function buildSampleSVG(key) {
    const p = SAMPLE_PALETTES[key] || SAMPLE_PALETTES.healthy;
    const isHealthy = key === "healthy";

    // Leaf veins
    let veins = "";
    for (let v = 0; v < 6; v++) {
        const angle = (v * 30) * Math.PI / 180;
        const x2 = (60 + Math.cos(angle) * 38).toFixed(1);
        const y2 = (60 + Math.sin(angle) * 50).toFixed(1);
        veins += `<line x1="60" y1="60" x2="${x2}" y2="${y2}" stroke="${p.vein}" stroke-width="1.2" opacity="0.55"/>`;
    }

    // Disease spots
    let spots = "";
    if (!isHealthy) {
        const positions = [
            [38, 35], [80, 42], [52, 75], [72, 68],
            [44, 55], [82, 60], [62, 38], [50, 82]
        ];
        positions.forEach(([cx, cy], i) => {
            const r = 3 + (i % 3) * 2.5;
            spots += `<circle cx="${cx}" cy="${cy}" r="${r}" fill="${p.spot}" opacity="0.72"/>`;
        });
    }

    return `<svg xmlns="http://www.w3.org/2000/svg" width="224" height="224" viewBox="0 0 120 120">
  <rect width="120" height="120" fill="${p.bg}"/>
  <ellipse cx="60" cy="60" rx="42" ry="52" fill="${p.leaf}" opacity="0.92"/>
  <line x1="60" y1="8" x2="60" y2="112" stroke="${p.vein}" stroke-width="1.8" opacity="0.6"/>
  ${veins}${spots}
</svg>`;
}

async function svgStringToPNGFile(svgString, filename) {
    const blob = new Blob([svgString], { type: "image/svg+xml" });
    const url  = URL.createObjectURL(blob);
    return new Promise(resolve => {
        const img = new Image();
        img.onload = () => {
            const canvas = document.createElement("canvas");
            canvas.width = canvas.height = 224;
            canvas.getContext("2d").drawImage(img, 0, 0, 224, 224);
            canvas.toBlob(pngBlob => {
                URL.revokeObjectURL(url);
                resolve(new File([pngBlob], filename, { type: "image/png" }));
            }, "image/png");
        };
        img.src = url;
    });
}

sampleButtons.addEventListener("click", async e => {
    const btn = e.target.closest(".sample-btn");
    if (!btn) return;
    const key = btn.dataset.sample;
    const svg = buildSampleSVG(key);
    const pngFile = await svgStringToPNGFile(svg, key + "_sample.png");
    handleFile(pngFile);
    sampleButtons.querySelectorAll(".sample-btn").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
});

// ---------------------------------------------------------------------------
// Inference  (POST /predict)
// ---------------------------------------------------------------------------
btnAnalyze.addEventListener("click", runInference);

async function runInference() {
    if (!currentFile) {
        showNotification("Please select a leaf image before analyzing.");
        return;
    }
    setAnalyzeLoading(true);
    hideNotification();

    const formData = new FormData();
    formData.append("image", currentFile);

    try {
        const res  = await fetch(API_PREDICT_URL, { method: "POST", body: formData });
        const data = await res.json();
        if (!res.ok || !data.success) {
            showNotification(data.error || ("Server error " + res.status));
            return;
        }
        renderResults(data);
    } catch (err) {
        showNotification("Cannot reach inference server. Ensure the backend is running on port 5000.");
        console.error("[AgriAI] Fetch error:", err);
    } finally {
        setAnalyzeLoading(false);
    }
}

function setAnalyzeLoading(loading) {
    btnAnalyze.disabled = loading;
    if (loading) {
        analyzeSpinner.classList.remove("hidden");
        analyzeBtnText.classList.add("hidden");
    } else {
        analyzeSpinner.classList.add("hidden");
        analyzeBtnText.classList.remove("hidden");
    }
}

// ---------------------------------------------------------------------------
// Result Card Rendering
// ---------------------------------------------------------------------------
function showPlaceholder() {
    resultPlaceholder.classList.remove("hidden");
    resultContent.classList.add("hidden");
}

function getConfidenceClass(conf) {
    if (conf >= CONF_HIGH)   return "conf-high";
    if (conf >= CONF_MEDIUM) return "conf-medium";
    return "conf-low";
}

function animateBar(el, targetPct, delay = 0) {
    el.style.width = "0%";
    requestAnimationFrame(() => {
        requestAnimationFrame(() => {
            el.style.transition = `width 0.65s ease ${delay}s`;
            el.style.width = targetPct + "%";
        });
    });
}

function renderResults(data) {
    const pred      = data.prediction;
    const topPreds  = data.top_predictions;
    const latencyMs = data.inference_latency_ms;
    const conf      = pred.confidence;           // 0-1 float
    const confPct   = (conf * 100).toFixed(1);

    // ---- Primary Diagnosis Banner ----
    diagnosisName.textContent = pred.display_name;
    diagnosisDesc.textContent = pred.description;

    if (pred.is_healthy) {
        statusPill.textContent = "\u2705 Healthy Foliage";
        statusPill.className   = "status-pill pill-healthy";
        diagnosisBanner.classList.replace("banner-disease", "banner-healthy");
    } else {
        statusPill.textContent = "\uD83D\uDD2C Likely Diagnosis";
        statusPill.className   = "status-pill pill-disease";
        diagnosisBanner.classList.replace("banner-healthy", "banner-disease");
    }

    latencyPill.textContent = "\u26A1 " + latencyMs.toFixed(0) + " ms";

    // Confidence gauge
    primaryConfidenceValue.textContent = confPct + "%";
    primaryConfidenceBar.className = "gauge-bar " + getConfidenceClass(conf);
    animateBar(primaryConfidenceBar, confPct, 0);

    // ---- Confidence Tier Badge ----
    let tierLabel, tierClass;
    if (conf >= 0.90) {
        tierLabel = "High (≥90%)";   tierClass = "tier-high";
    } else if (conf >= 0.50) {
        tierLabel = "Moderate (50–89%)"; tierClass = "tier-moderate";
    } else {
        tierLabel = "Low (<50%)";     tierClass = "tier-low";
    }
    confidenceTierBadge.textContent = tierLabel;
    confidenceTierBadge.className   = "conf-tier-badge " + tierClass;

    // ---- Quality Warning ----
    if (data.quality_warning) {
        qualityAlertText.textContent = data.quality_warning;
        qualityAlert.classList.remove("hidden");
    } else {
        qualityAlert.classList.add("hidden");
    }

    // ---- Top-3 Differential Diagnoses ----
    topPredictionsList.innerHTML = "";
    topPreds.forEach((p, i) => {
        const pct   = (p.confidence * 100).toFixed(1);
        const color = RANK_COLORS[i] || "#94a3b8";
        const item  = document.createElement("div");
        item.className = "prediction-item" + (i === 0 ? " prediction-primary" : "");
        item.innerHTML =
            `<div class="pred-rank-badge" style="background:${color}22;color:${color};border-color:${color}44">#${p.rank}</div>` +
            `<div class="pred-info">` +
                `<div class="pred-name">${p.display_name}</div>` +
                `<div class="pred-bar-track">` +
                    `<div class="pred-bar-fill ${getConfidenceClass(p.confidence)}" data-target="${pct}"></div>` +
                `</div>` +
            `</div>` +
            `<div class="pred-confidence" style="color:${color}">${pct}%</div>`;
        topPredictionsList.appendChild(item);
        // Stagger bar animations
        const bar = item.querySelector(".pred-bar-fill");
        animateBar(bar, pct, 0.08 + i * 0.12);
    });

    // ---- Recommendation ----
    recommendationText.textContent = pred.recommendation;

    // ---- Grad-CAM Heatmap ----
    if (data.gradcam_image) {
        gradcamImage.src = data.gradcam_image;
        gradcamPanel.classList.remove("hidden");
    } else {
        gradcamImage.src = "";
        gradcamPanel.classList.add("hidden");
    }

    // ---- Reveal result card ----
    resultPlaceholder.classList.add("hidden");
    resultContent.classList.remove("hidden");

    // Scroll into view on narrow viewports
    if (window.innerWidth < 900) {
        document.getElementById("resultCard").scrollIntoView({ behavior: "smooth", block: "start" });
    }
}

// ---------------------------------------------------------------------------
// Clipboard Paste Support
// ---------------------------------------------------------------------------
document.addEventListener("paste", e => {
    const items = e.clipboardData && e.clipboardData.items;
    if (!items) return;
    for (const item of items) {
        if (item.type.startsWith("image/")) {
            const f = item.getAsFile();
            if (f) { handleFile(f); break; }
        }
    }
});

// ---------------------------------------------------------------------------
// Initialisation
// ---------------------------------------------------------------------------
(function init() {
    checkBackendHealth();
    setInterval(checkBackendHealth, 30_000);
})();
