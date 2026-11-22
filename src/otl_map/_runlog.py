"""Minimal stand-in for the provenance Run used in the private repository (prints only)."""
from __future__ import annotations


class Run:
    def __init__(self, purpose, params=None, inputs=None, script=None):
        self.purpose = purpose

    def __enter__(self):
        print(f":: {self.purpose}")
        return self

    def __exit__(self, *exc):
        return False

    def note(self, **kw):
        for k, v in kw.items():
            print(f"   {k} = {v}")

    def output(self, path):
        return path

    def write_text(self, path, txt):
        with open(path, "w") as f:
            f.write(txt)
        return path
