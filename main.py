import os
import subprocess
import sys
from pathlib import Path

def _free_port_if_occupied(port: int) -> None:
    """If target port is occupied, terminate stale process using it to prevent Errno 10048."""
    import socket
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1.0)
            if s.connect_ex(("127.0.0.1", port)) == 0:
                print(f"--> Port {port} is currently occupied. Terminating stale process...")
                if sys.platform == "win32":
                    cmd = f"Get-NetTCPConnection -LocalPort {port} -ErrorAction SilentlyContinue | ForEach-Object {{ Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }}"
                    subprocess.run(["powershell", "-Command", cmd], capture_output=True, timeout=5)
                else:
                    subprocess.run(f"fuser -k {port}/tcp", shell=True, capture_output=True, timeout=5)
    except Exception:
        pass


if __name__ == "__main__":
    is_venv = sys.prefix != sys.base_prefix
    venv_python = Path(__file__).parent / ".venv" / "Scripts" / "python.exe"
    port_str = os.getenv("PORT", "9826")
    port = int(port_str)

    _free_port_if_occupied(port)

    if not is_venv and venv_python.exists():
        print(f"--> Auto-switching to project virtual environment: {venv_python}")
        cmd = [str(venv_python), "-m", "uvicorn", "backend.main:app", "--host", "127.0.0.1", "--port", str(port)]
        sys.exit(subprocess.call(cmd))

    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=port)
