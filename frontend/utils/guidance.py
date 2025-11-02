from __future__ import annotations

from typing import Dict, List


GUIDANCE: Dict[str, Dict[str, List[str]]] = {
    "Apple___Apple_scab": {
        "Summary": [
            "Fungal disease causing scab-like lesions on leaves and fruit.",
        ],
        "Management": [
            "Prune for airflow; remove fallen leaves.",
            "Avoid overhead irrigation.",
            "Apply labeled fungicides at green tip through early season.",
        ],
        "Prevention": [
            "Choose resistant cultivars where possible.",
            "Sanitize orchard floor each fall.",
        ],
    },
    "Apple___Black_rot": {
        "Summary": ["Caused by Botryosphaeria; leaf spots and fruit rot."],
        "Management": [
            "Remove cankers and mummified fruit.",
            "Maintain tree vigor and proper pruning.",
        ],
        "Prevention": ["Sanitation and fungicide programs in wet periods."],
    },
    "Apple___Cedar_apple_rust": {
        "Summary": ["Gymnosporangium rust; orange leaf spots with fringed margins."],
        "Management": ["Remove nearby junipers where feasible.", "Targeted fungicides during infection windows."],
        "Prevention": ["Resistant varieties and site selection."],
    },
    "Apple___healthy": {
        "Summary": ["No disease detected; monitor regularly."],
        "Management": ["Maintain balanced nutrition and irrigation."],
        "Prevention": ["Scout weekly, remove debris, ensure airflow."],
    },
}


