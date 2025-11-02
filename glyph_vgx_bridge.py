#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
glyph_vgx_bridge.py — minimal adapter so GlyphMatics/SigilAgi can call Vision_Glyphs.

Usage:
    from glyph_vgx_bridge import detect_top
    ok, det = detect_top("/path/to/image.jpg", score_thr=0.6)
    if ok:
        print(det["bbox"], det["label"], det["score"])
"""

from typing import Dict, Any, Tuple, Optional

def detect_top(image_path: str, score_thr: float = 0.5, mode: str = "rcnn") -> Tuple[bool, Dict[str, Any]]:
    try:
        import torch
        from vision_glyphs.rcnn_suite import RCNNDetector, RCNNConfig, FasterDetector, load_rgb, draw_boxes
    except Exception as e:
        return False, {"error": f"vision_glyphs not available: {e}"}

    try:
        dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        if mode == "faster":
            model = FasterDetector()  # will raise if torchvision not present
            out = model.predict_image(image_path, class_names=None)
        else:
            cfg = RCNNConfig()
            det = RCNNDetector(cfg).to(dev).eval()
            out = det.predict_image(image_path, class_names=None)

        boxes = out.get("boxes", [])
        scores = out.get("scores", [])
        labels = out.get("labels", [])
        names  = out.get("names", [])

        best_i = -1
        best_s = -1.0
        for i, s in enumerate(scores):
            if s >= score_thr and s > best_s:
                best_i, best_s = i, float(s)

        if best_i < 0:
            return False, {"reason": "no detection ≥ threshold", "count": len(scores)}

        bbox = boxes[best_i]
        label = names[best_i] if best_i < len(names) else int(labels[best_i])
        return True, {"bbox": bbox, "label": label, "score": best_s}
    except FileNotFoundError as e:
        return False, {"error": f"file not found: {e}"}
    except Exception as e:
        return False, {"error": f"vgx detect error: {e}"}
