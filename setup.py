import os

from setuptools import find_namespace_packages, setup

with open("VERSION") as f:
    version = f.read().strip()

if os.getenv("PRERELEASE"):
    version += os.getenv("PRERELEASE")
    with open("VERSION", "w") as f:
        f.write(f"{version}\n")


# NOTE: ~=0.6.10 means >=0.6.10, ==0.6.*
# NOTE: ~=0.6 means >=0.6, ==0.*

# This is exhaustive.
strict_install_requires = [
    "evaluate~=0.4.0",
    "scikit-learn~=1.3.0",
    "rouge-score~=0.1.2",
    "aiolimiter~=1.1.0",
    "backoff~=2.2.1",
    "openai>=1",
    "datasets~=2.14.6",
    "mlflow~=2.3.0",
    "hydra-core~=1.3.2",
    "tiktoken~=0.3.1",
    "transformers~=4.34.0",
    "sentencepiece~=0.1.99",
    "openpyxl==3.1.2",
    "aiofiles~=23.1.0",
]

# NOTE: These are usually installed with conda
strict_all_requires = ["numpy~=1.22.3", "pandas~=2.0.3", "torch~=2.0.0"]

install_requires = [
    "evaluate",
    "scikit-learn",
    "rouge-score",
    "aiolimiter",
    "backoff",
    "openai>=1",
    "datasets",
    "mlflow",
    "hydra-core",
    "tiktoken",
    "transformers",
    "sentencepiece",
    "openpyxl",
    # NOTE: These are usually installed with conda
    "numpy",
    "pandas",
    "torch",
    "aiofiles",
]


setup(
    name="rambla",
    version=version,
    description="LLM evaluation framework",
    packages=find_namespace_packages("."),
    package_dir={"": "."},
    extras_require={
        "pip": install_requires,
        "strict": strict_install_requires,
        "strict_all": strict_install_requires + strict_all_requires,
    },
    include_package_data=True,
)
