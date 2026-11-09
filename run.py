#!/usr/bin/env python3
"""RagMate Unified Launcher.

Runs the complete application stack with a single command:
1. Detects and uses the backend virtualenv (backend/.venv).
2. Initializes backend/.env from .env.example if missing.
3. Automatically starts PostgreSQL (pgvector) container via Docker Compose if not running.
4. Starts the FastAPI server (Uvicorn).
5. Automatically opens the browser once the service is ready.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

# Enable ANSI escape sequences on Windows console
if sys.platform == "win32":
    os.system("")

ROOT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = ROOT_DIR / "backend"
ENV_FILE = BACKEND_DIR / ".env"
ENV_EXAMPLE = BACKEND_DIR / ".env.example"


def log(msg: str) -> None:
    print(f"\033[36m[RagMate]\033[0m {msg}", flush=True)


def log_warn(msg: str) -> None:
    print(f"\033[33m[RagMate Warning]\033[0m {msg}", flush=True)


def log_err(msg: str) -> None:
    print(f"\033[31m[RagMate Error]\033[0m {msg}", file=sys.stderr, flush=True)


def get_venv_python() -> Path | None:
    """Finds the python binary inside backend/.venv or .venv."""
    candidates = [
        BACKEND_DIR / ".venv" / ("Scripts" if sys.platform == "win32" else "bin") / ("python.exe" if sys.platform == "win32" else "python"),
        ROOT_DIR / ".venv" / ("Scripts" if sys.platform == "win32" else "bin") / ("python.exe" if sys.platform == "win32" else "python"),
    ]
    for p in candidates:
        if p.is_file():
            return p
    return None


def ensure_venv(force_install: bool = False) -> Path:
    """Ensures a valid virtual environment exists with dependencies installed."""
    venv_python = get_venv_python()
    target_venv = BACKEND_DIR / ".venv"

    if force_install or not venv_python:
        log("Setting up virtual environment in backend/.venv ...")
        subprocess.run([sys.executable, "-m", "venv", str(target_venv)], check=True)
        venv_python = get_venv_python()
        if not venv_python:
            log_err("Failed to create virtual environment.")
            sys.exit(1)

        log("Installing backend dependencies (pip install -e backend) ...")
        pip_cmd = [str(venv_python), "-m", "pip", "install", "--upgrade", "pip"]
        subprocess.run(pip_cmd, check=True)
        install_cmd = [str(venv_python), "-m", "pip", "install", "-e", str(BACKEND_DIR)]
        subprocess.run(install_cmd, check=True)
        log("Dependencies installed successfully.")

    return venv_python


def switch_to_venv_if_needed() -> None:
    """If running outside the virtualenv, re-executes inside it."""
    venv_python = get_venv_python()
    if not venv_python:
        return

    try:
        current_exe = Path(sys.executable).resolve()
        target_exe = venv_python.resolve()
        if current_exe == target_exe:
            return
    except Exception:
        pass

    # Re-exec with virtual environment python
    args = [str(venv_python), str(Path(__file__).resolve())] + sys.argv[1:]
    log(f"Switching to virtual environment: {venv_python}")
    result = subprocess.run(args)
    sys.exit(result.returncode)


def ensure_env_file() -> None:
    """Ensures backend/.env exists, bootstrapping from .env.example."""
    if not ENV_FILE.exists():
        if ENV_EXAMPLE.exists():
            log("Initializing backend/.env from backend/.env.example ...")
            shutil.copyfile(ENV_EXAMPLE, ENV_FILE)
            log("Created backend/.env with default settings.")
        else:
            log_warn("backend/.env.example not found; skipping .env initialization.")


def parse_db_host_port() -> tuple[str, int]:
    """Extracts database host and port from backend/.env or returns defaults."""
    host = "127.0.0.1"
    port = 5432
    if ENV_FILE.exists():
        try:
            content = ENV_FILE.read_text(encoding="utf-8", errors="ignore")
            # Look for @host:port/ or @host/
            m = re.search(r"DATABASE_URL=[^\n@]+@([^:/]+)(?::(\d+))?", content)
            if m:
                host = m.group(1).strip()
                if m.group(2):
                    port = int(m.group(2))
        except Exception:
            pass

    return host, port


def is_port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    """Checks whether a TCP port is open and listening."""
    target_host = "127.0.0.1" if host in ("localhost", "0.0.0.0", "") else host
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            return s.connect_ex((target_host, port)) == 0
    except Exception:
        return False


def ensure_postgres(no_docker: bool = False) -> None:
    """Checks if PostgreSQL is reachable; if not, attempts to start via Docker Compose."""
    db_host, db_port = parse_db_host_port()
    is_local = db_host in ("127.0.0.1", "localhost", "0.0.0.0", "::1")

    if is_port_open(db_host, db_port):
        log(f"Database reachable at {db_host}:{db_port}.")
        return

    if not is_local or no_docker:
        log_warn(f"Database at {db_host}:{db_port} is not responding. Please ensure PostgreSQL is running.")
        return

    docker_path = shutil.which("docker")
    if not docker_path:
        log_warn(
            f"Database at {db_host}:{db_port} is not responding, and 'docker' CLI was not found.\n"
            f"Please start PostgreSQL (with pgvector) or install/start Docker Desktop."
        )
        return

    log("PostgreSQL is not running. Starting container via 'docker compose up -d postgres' ...")
    try:
        subprocess.run(["docker", "compose", "-f", "docker-compose.yml", "-f", "docker-compose.dev.yml", "up", "-d", "postgres"], cwd=ROOT_DIR, check=True)
    except Exception as e:
        log_warn(f"Could not automatically start Docker container: {e}")
        log_warn("If Docker Desktop is running, you can manually run: docker compose up -d postgres")
        return

    # Wait for PostgreSQL port to become ready
    log("Waiting for PostgreSQL to be ready ...")
    ready = False
    for _ in range(25):
        time.sleep(1)
        if is_port_open(db_host, db_port, timeout=0.5):
            ready = True
            break

    if ready:
        log("PostgreSQL is ready.")
    else:
        log_warn("PostgreSQL container started, but port 5432 is not answering yet. The app will retry on connect.")


def open_browser_when_ready(url: str, check_url: str, timeout: float = 20.0) -> None:
    """Background task to poll the health endpoint and open browser once ready."""
    def _worker():
        start = time.time()
        while time.time() - start < timeout:
            time.sleep(0.5)
            try:
                with urllib.request.urlopen(check_url, timeout=1.0) as resp:
                    if resp.status == 200:
                        log(f"App is ready! Opening {url} in your browser ...")
                        webbrowser.open(url)
                        return
            except Exception:
                pass

    t = threading.Thread(target=_worker, daemon=True)
    t.start()


def main() -> None:
    parser = argparse.ArgumentParser(description="RagMate Unified Launcher")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    parser.add_argument("--no-reload", action="store_true", help="Disable auto-reload")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open the browser")
    parser.add_argument("--no-docker", action="store_true", help="Do not attempt to auto-start docker postgres")
    parser.add_argument("--install", action="store_true", help="Reinstall backend dependencies into backend/.venv")
    parser.add_argument("--check", action="store_true", help="Only check requirements and database connectivity, then exit")
    args = parser.parse_args()

    # Step 1: Ensure we are running with the virtualenv Python
    if args.install or not get_venv_python():
        ensure_venv(force_install=args.install)
    switch_to_venv_if_needed()

    # Step 2: Ensure .env is present
    ensure_env_file()

    # Step 3: Ensure PostgreSQL is available
    ensure_postgres(no_docker=args.no_docker)

    if args.check:
        log("Environment and database check completed.")
        return

    # Check port availability
    if is_port_open(args.host, args.port, timeout=0.3):
        log_warn(f"Port {args.port} is already in use. You can pass '--port <number>' to choose another.")

    # Step 4: Schedule browser open
    display_host = "localhost" if args.host in ("127.0.0.1", "0.0.0.0") else args.host
    web_url = f"http://{display_host}:{args.port}"
    check_url = f"http://127.0.0.1:{args.port}/health"

    if not args.no_browser:
        open_browser_when_ready(web_url, check_url)

    # Step 5: Launch Uvicorn
    log(f"Starting RagMate server on {web_url} ... (Press Ctrl+C to stop)")
    uvicorn_cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "backend.app:app",
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]
    if not args.no_reload:
        uvicorn_cmd.extend(["--reload", "--reload-dir", "backend", "--reload-dir", "frontend"])

    proc = None
    try:
        proc = subprocess.Popen(uvicorn_cmd, cwd=ROOT_DIR)
        proc.wait()
    except KeyboardInterrupt:
        log("Stopping server ...")
        if proc:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
        log("Server stopped.")


if __name__ == "__main__":
    main()
