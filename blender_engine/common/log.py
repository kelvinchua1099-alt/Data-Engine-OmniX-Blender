"""Minimal logging that works both inside Blender (stdout) and in plain python."""
import sys

_PREFIX = "[omnix]"


def log(msg):
    print(f"{_PREFIX} {msg}", flush=True)


def warn(msg):
    print(f"{_PREFIX} WARNING: {msg}", flush=True)


def error(msg):
    print(f"{_PREFIX} ERROR: {msg}", file=sys.stderr, flush=True)
