$ErrorActionPreference = "Stop"

$env:PYTHONUTF8 = "1"
$env:CHROMA_DIR = "./data/chroma310"

& ".\.venv\Scripts\python.exe" "main.py"
