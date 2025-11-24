"""Calibration module for detection calibration benchmark."""
from .detection_calibration import (
    compute_detection_ece,
    match_detections_to_gt,
    plot_reliability_diagram,
    apply_temperature_scaling,
    apply_platt_scaling,
    apply_isotonic_regression,
    run_calibration_benchmark,
    DetectionCalibrationMetrics,
)
