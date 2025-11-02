#!/usr/bin/env python3
# glyph_loader.py — Executable glyph dep-loader (no placeholders)

import sys, subprocess, importlib, io, os
from typing import Dict, Tuple, Callable

try:
    from packaging.version import Version
except Exception:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "packaging"], stdout=sys.stdout)
    from packaging.version import Version

def _pip(spec: str) -> None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", spec], stdout=sys.stdout)

def ensure(import_name: str, pip_name: str, min_version: str, extra_spec: str = ""):
    try:
        m = importlib.import_module(import_name)
    except Exception:
        _pip(pip_name + (extra_spec or ""))
        m = importlib.import_module(import_name)
    ver = None
    for attr in ("__version__","VERSION","version"):
        v = getattr(m, attr, None)
        if isinstance(v, (str, bytes)):
            ver = v.decode() if isinstance(v, bytes) else v; break
    if ver is None and import_name in ("matplotlib.pyplot","PIL","cv2"):
        if import_name == "matplotlib.pyplot":
            import matplotlib; ver = getattr(matplotlib, "__version__", None)
        elif import_name == "PIL":
            import PIL; ver = getattr(PIL, "__version__", None)
        elif import_name == "cv2":
            import cv2; ver = getattr(cv2, "__version__", None)
    if ver is None:
        ver = str(getattr(m, "__dict__", {}).get("__version__", "0"))
    if Version(ver) < Version(min_version):
        _pip(f"{pip_name}>={min_version}" + (extra_spec or ""))
        m = importlib.reload(importlib.import_module(import_name))
        ver2 = getattr(m, "__version__", None)
        if ver2 is None and import_name == "matplotlib.pyplot":
            import matplotlib; ver2 = getattr(matplotlib, "__version__", None)
        if ver2 is None and import_name == "PIL":
            import PIL; ver2 = getattr(PIL, "__version__", None)
        if ver2 is None and import_name == "cv2":
            import cv2; ver2 = getattr(cv2, "__version__", None)
        if ver2 is None: ver2 = "0"
        if Version(ver2) < Version(min_version):
            raise RuntimeError(f"{pip_name} upgrade failed: have {ver2}, need >= {min_version}")
        ver = ver2
    return m, ver

# sanity checks
def _sanity_numpy():
    import numpy as np
    a = np.arange(9).reshape(3,3); assert int(a.sum()) == 36

def _sanity_torch():
    import torch as t
    x = t.randn(2,3); y = t.nn.Linear(3,4)(x); _ = y.detach().numpy(); _ = bool(t.cuda.is_available())

def _sanity_torchvision():
    import torchvision as tv
    from torchvision import transforms as T
    _ = tv.ops.nms is not None
    from PIL import Image as Image_
    pipe = T.Compose([T.Resize((8,8)), T.ToTensor()])
    img = Image_.new("RGB", (16,16), (10,20,30))
    assert pipe(img).shape == (3,8,8)

def _sanity_skimage():
    import numpy as np
    from skimage import filters, measure
    img = np.zeros((10,10), np.float32); img[3:7,3:7] = 1.0
    e = filters.sobel(img); lbl = measure.label(img > 0.5); assert e.shape == img.shape and lbl.max() == 1

def _sanity_opencv():
    import numpy as np, cv2
    bgr = np.zeros((4,4,3), np.uint8); bgr[...,1]=255
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY); assert gray.shape == (4,4)

def _sanity_matplotlib():
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(); ax = fig.add_subplot(111); ax.plot([0,1],[0,1])
    buf = io.BytesIO(); fig.savefig(buf, format="png"); assert buf.tell() > 0; plt.close(fig)

def _sanity_pillow():
    from PIL import Image
    im = Image.new("RGBA", (8,8), (1,2,3,255)); im2 = im.convert("RGB"); assert im2.size == (8,8)

Glyph = Tuple[str, str, str, Callable[[], None]]
GLYPHS: Dict[str, Glyph] = {
    "⊡numpy":        ("numpy",             "numpy",           "1.24.0", _sanity_numpy),
    "⊗torch":        ("torch",             "torch",           "2.2.0",  _sanity_torch),
    "⊕torchvision":  ("torchvision",       "torchvision",     "0.18.0", _sanity_torchvision),
    "⊜skimage":      ("skimage",           "scikit-image",    "0.21.0", _sanity_skimage),
    "⊟opencv":       ("cv2",               "opencv-python",   "4.8.0",  _sanity_opencv),
    "⊠matplotlib":   ("matplotlib.pyplot", "matplotlib",      "3.7.0",  _sanity_matplotlib),
    "⊡pillow":       ("PIL",               "Pillow",          "10.0.0", _sanity_pillow),
}

def run_chain(verbose: bool = True) -> Dict[str, Tuple[str, str]]:
    results: Dict[str, Tuple[str, str]] = {}
    order = ["⊡numpy","⊗torch","⊕torchvision","⊜skimage","⊟opencv","⊠matplotlib","⊡pillow"]
    for g in order:
        import_name, pip_name, min_ver, sanity = GLYPHS[g]
        mod, ver = ensure(import_name, pip_name, min_ver)
        if verbose: print(f"[OK] {g} -> {pip_name} {ver} (>= {min_ver})", flush=True)
        sanity(); results[g] = (pip_name, ver)
    if verbose:
        try:
            import torch
            print(f"[INFO] torch.cuda.is_available={torch.cuda.is_available()}", flush=True)
            if torch.cuda.is_available(): print(f"[INFO] CUDA device count={torch.cuda.device_count()}", flush=True)
        except Exception:
            pass
    return results

def main():
    os.environ.setdefault("MPLBACKEND", "Agg")
    print("⧈[⊗Ωtorch_init ⊕Ωvision_init ⊜Ωskimage_init ⊟Ωcv_init ⊡Ωnp_init ⊠Ωplt_init ⊡Ωpil_init] ⊚auto_verify ⊞system_ready")
    res = run_chain(verbose=True)
    print("\n[SUMMARY]")
    for k,(name,ver) in res.items(): print(f"{k}: {name}=={ver}")
    print("\n[READY]")

if __name__ == "__main__":
    main()
