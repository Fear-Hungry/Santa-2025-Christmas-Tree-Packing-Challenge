from __future__ import annotations

from pathlib import Path

from setuptools import Extension, setup


def main() -> None:
    import numpy as np

    here = Path(__file__).resolve().parent
    ext = Extension(
        name="fastcollide",
        sources=[str(here / "fastcollide.cpp")],
        include_dirs=[np.get_include()],
        language="c++",
        extra_compile_args=["-O3", "-std=c++17"],
    )

    setup(
        name="fastcollide",
        version="0.0.0",
        ext_modules=[ext],
    )


if __name__ == "__main__":
    main()

