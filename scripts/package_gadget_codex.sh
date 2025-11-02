@@
-USE_PYZIP=${USE_PYZIP:-0}
+USE_PYZIP=${USE_PYZIP:-0}
+set -euo pipefail
@@
-if command -v unzip >/dev/null 2>&1; then
+if command -v unzip >/dev/null 2>&1; then
   unzip -l "$ART" | sed -n '1,50p'
 else
   python - "$ART" <<'PYL'
 import sys, zipfile
 with zipfile.ZipFile(sys.argv[1],'r') as z:
     for i,info in enumerate(z.infolist()[:50]):
         print(info.filename, info.file_size)
 PYL
 fi
+echo "[DONE] Artifact ready: $ART"
