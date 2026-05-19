from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AION_REPO_DIR = PROJECT_ROOT / "AION"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
MODELS_DIR = OUTPUTS_DIR / "models"
TABLES_DIR = OUTPUTS_DIR / "tables"
DOCS_DIR = PROJECT_ROOT / "docs"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
REFERENCES_DIR = PROJECT_ROOT / "references"


def ensure_project_dirs() -> None:
    for path in [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        FIGURES_DIR,
        MODELS_DIR,
        TABLES_DIR,
        DOCS_DIR,
        NOTEBOOKS_DIR,
        REFERENCES_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)
