"""
AgriAI Evaluation Module
"""

from .metrics import (
    compute_classification_metrics,
    measure_inference_latency,
    plot_confusion_matrix,
    evaluate_dataset
)
from .gradcam import (
    GradCAM,
    overlay_cam_on_image,
    create_gradcam_3panel_figure,
)

__all__ = [
    "compute_classification_metrics",
    "measure_inference_latency",
    "plot_confusion_matrix",
    "evaluate_dataset",
    "GradCAM",
    "overlay_cam_on_image",
    "create_gradcam_3panel_figure",
]

