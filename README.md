# Vision_Glyphs
RCNN
## Vision-Glyphs (Termux-safe) — Kept Components

- `glyph-loader`: verifies core libs (NumPy, Torch, Pillow, Matplotlib). On Termux it **skips** installing `opencv-python`, `scikit-image`, and reports `torchvision` unavailability cleanly.
- `vision-glyphs` CLI (RCNN mode): pure-PyTorch implementation (no torchvision). Uses selective-search (if `scikit-image` present) or fallbacks to grid proposals. AMP is **disabled on CPU** to avoid `bfloat16` NumPy issues.

**Examples**
```bash
# Storage access on Android:
termux-setup-storage
IMG="$HOME/storage/shared/DCIM/Camera/your_photo.jpg"
OUT="$HOME/storage/shared/Download/det.jpg"

# Verify deps:
glyph-loader

# Run RCNN detector (CPU on Termux):
vision-glyphs --mode rcnn --image "$IMG" --out "$OUT" --score_thr 0.6
