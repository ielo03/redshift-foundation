from __future__ import annotations

import importlib
import platform
import sys


PACKAGES = [
    "numpy",
    "pandas",
    "matplotlib",
    "sklearn",
    "torch",
    "torchvision",
    "tensorflow",
    "PIL",
]


def package_version(name: str) -> str:
    try:
        module = importlib.import_module(name)
    except Exception as exc:
        return f"missing ({exc.__class__.__name__})"

    return getattr(module, "__version__", "version attribute unavailable")


def main() -> None:
    print("Python:", sys.version.split()[0])
    print("Platform:", platform.platform())
    print()
    print("Installed package summary:")
    for package in PACKAGES:
        print(f"- {package}: {package_version(package)}")


if __name__ == "__main__":
    main()
