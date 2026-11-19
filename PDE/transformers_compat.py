"""Install or verify the HulluEdit compatibility files for Transformers 4.31.0."""

from importlib.metadata import distribution
from pathlib import Path
import shutil
import sys


SOURCE = Path(__file__).resolve().parent / "vendor" / "transformers"


def check_transformers(*, install=False):
    if install and sys.prefix == sys.base_prefix:
        raise RuntimeError("Apply the compatibility files inside a virtual environment.")
    package = distribution("transformers")
    if package.version != "4.31.0":
        raise RuntimeError(f"Expected Transformers 4.31.0, found {package.version}.")

    for source in sorted(SOURCE.rglob("*.py")):
        relative_path = source.relative_to(SOURCE)
        target = Path(package.locate_file(Path("transformers") / relative_path))
        if install:
            if not target.resolve().is_relative_to(Path(sys.prefix).resolve()):
                raise RuntimeError(f"Transformers is installed outside the virtual environment: {target}")
            shutil.copyfile(source, target)
        elif target.read_bytes() != source.read_bytes():
            raise RuntimeError("Transformers compatibility files are missing. Rerun scripts/setup.sh.")


if __name__ == "__main__":
    check_transformers(install=True)
