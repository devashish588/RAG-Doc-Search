import os
import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    is_venv = sys.prefix != sys.base_prefix
    venv_python = Path(__file__).parent / ".venv" / "Scripts" / "python.exe"

    if not is_venv and venv_python.exists():
        port = os.getenv("PORT", "9826")
        print(f"--> Auto-switching to project virtual environment: {venv_python}")
        cmd = [str(venv_python), "-m", "uvicorn", "backend.main:app", "--host", "127.0.0.1", "--port", str(port)]
        sys.exit(subprocess.call(cmd))

    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=int(os.getenv("PORT", "9826")))
