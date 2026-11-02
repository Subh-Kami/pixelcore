"""Verify exported PixelCore state, token and framebuffer consistency without a model."""
import argparse
import json
from pathlib import Path

from qgpu.pixelcore.runtime import verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path, nargs="+")
    args = parser.parse_args()
    for path in args.trace:
        result = verify(json.loads(path.read_text()))
        print(json.dumps({"trace": str(path), **result}, indent=2))


if __name__ == "__main__":
    main()
