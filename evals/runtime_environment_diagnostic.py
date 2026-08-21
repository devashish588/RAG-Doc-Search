"""Runtime environment diagnostic.

Usage:
    .venv/Scripts/python.exe evals/runtime_environment_diagnostic.py

Clearly identifies whether the correct Python interpreter is active.
"""
import importlib
import sys
import site


def _check_package(import_name: str) -> str:
    try:
        mod = importlib.import_module(import_name)
        version = getattr(mod, "__version__", getattr(mod, "VERSION", None))
        if isinstance(version, tuple):
            version = ".".join(str(v) for v in version)
        return f"OK ({version})" if version else "OK"
    except ImportError as exc:
        return f"MISSING — {exc}"


def _is_virtualenv() -> bool:
    return sys.prefix != sys.base_prefix


def main():
    print("=" * 60)
    print("Runtime Environment Diagnostic")
    print("=" * 60)
    print()
    print(f"  executable:    {sys.executable}")
    print(f"  Python:        {sys.version.split()[0]}")
    print(f"  prefix:        {sys.prefix}")
    print(f"  base_prefix:   {sys.base_prefix}")
    print(f"  virtualenv:    {_is_virtualenv()}")
    print(f"  site-packages: {site.getsitepackages()[0] if site.getsitepackages() else 'N/A'}")
    print()

    if not _is_virtualenv():
        print("  WARNING: Not running inside a virtual environment.")
        print(f"  The global 'python' resolves to: {sys.executable}")
        print()
        print("  This WILL cause ModuleNotFoundError for chromadb, fastembed, etc.")
        print()
        print("  FIX: Activate the project venv first:")
        print()
        print("    Windows:  .venv\\Scripts\\activate")
        print("    Linux:    source .venv/bin/activate")
        print()
        print("  Then start with:")
        print("    .venv\\Scripts\\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 9826")
        print()
    else:
        print("  Virtual environment: ACTIVE")
        print()

    print("  Dependency Check:")
    print("  " + "-" * 56)
    deps = [
        ("chromadb", "chromadb"),
        ("fastembed", "fastembed"),
        ("numpy", "numpy"),
        ("rank_bm25", "rank-bm25"),
        ("prometheus_client", "prometheus-client"),
        ("flashrank", "flashrank"),
    ]
    all_ok = True
    for import_name, pkg_name in deps:
        status = _check_package(import_name)
        ok = status.startswith("OK")
        if not ok:
            all_ok = False
        mark = "OK" if ok else "MISSING"
        print(f"    {pkg_name:20s} {mark:8s}  {status}")

    print()
    if all_ok:
        print("  ALL DEPENDENCIES OK")
    else:
        print("  SOME DEPENDENCIES MISSING")
        print("  Run: .venv\\Scripts\\python.exe -m pip install -r requirements.txt")
    print("=" * 60)


if __name__ == "__main__":
    main()
