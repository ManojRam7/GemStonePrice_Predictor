from pathlib import Path
from setuptools import find_packages, setup


ROOT = Path(__file__).parent
REQUIREMENTS_PATH = ROOT / "requirements.txt"


def get_requirements(path: Path) -> list[str]:
    requirements = []
    for line in path.read_text(encoding="utf-8").splitlines():
        clean = line.strip()
        if clean and not clean.startswith("#") and clean != "-e .":
            requirements.append(clean)
    return requirements


setup(
    name="gemstone-price-predictor",
    version="1.0.0",
    author="Manoj Ram",
    description="Portfolio-grade end-to-end gemstone price prediction project",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=get_requirements(REQUIREMENTS_PATH),
)
