#!/usr/bin/env python3

import importlib.util
import pathlib
import struct
import tempfile


PROJECT_DIR = pathlib.Path(__file__).resolve().parents[1]
MODULE_PATH = (
    PROJECT_DIR / "experiment" / "deep100m" / "prepare_deep100m_data.py"
)


def load_preparation_module():
    spec = importlib.util.spec_from_file_location("deep_data_preparation", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load Deep100M preparation module: {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    preparation = load_preparation_module()
    ids = (11, 12, 13, 21, 22, 23)
    distances = (0.1, 0.2, 0.3, 1.1, 1.2, 1.3)

    with tempfile.TemporaryDirectory() as directory:
        root = pathlib.Path(directory)
        source = root / "source.gt"
        target = root / "target.bin"
        source.write_bytes(
            struct.pack("<II", 2, 3)
            + struct.pack("<6I", *ids)
            + struct.pack("<6f", *distances)
        )

        preparation.write_groundtruth(source, target, 3, 1)

        expected = struct.pack("<II3I", 1, 3, *ids[:3])
        actual = target.read_bytes()
        if actual != expected:
            raise AssertionError(
                "prepared groundtruth must contain its header and ID payload only"
            )


if __name__ == "__main__":
    main()
