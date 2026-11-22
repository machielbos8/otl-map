"""`otl-map-examples`: copy the examples of the manual and the manual itself into a new folder (as
hector-examples does for Hector); pip installs them inside the package, where users do not look."""
import argparse
import shutil
import sys
from pathlib import Path

MANUAL = "otl-map_manual_v0.1.pdf"


def main(argv=None):
    p = argparse.ArgumentParser(prog="otl-map-examples",
                                description="Copy the otl-map examples and the user manual into a new folder.")
    p.add_argument("dest", nargs="?", default="otl-map-examples", help="folder to create (default: otl-map-examples)")
    dest = Path(p.parse_args(argv).dest)
    if dest.exists():
        sys.exit(f"Destination already exists: {dest.resolve()}\nRemove it first or choose a different name.")
    pkg = Path(__file__).parent
    shutil.copytree(pkg / "examples", dest, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    manual = pkg / "docs" / MANUAL
    if manual.is_file():
        shutil.copy(manual, dest / MANUAL)
    print(f"Examples copied to: {dest.resolve()}")
    print(f"  {MANUAL}  the user manual (start here: Section 5, Tutorial)")
    print("  ex1  a first map without the coastline (Schwiderski M2, station BRST)")
    print("  ex2  the complete chain with the coastline (BLQ file for BRST)")
    print("  ex3  the same from Python")
    print("Run one with:  cd <folder>/ex1 && python example1.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
