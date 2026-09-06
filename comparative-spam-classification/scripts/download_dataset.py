from pathlib import Path
import shutil
import kagglehub

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "data" / "raw"
TARGET.mkdir(parents=True, exist_ok=True)
path = Path(kagglehub.dataset_download("ashfakyeafi/spam-email-classification"))
print(f"Kaggle download location: {path}")
for source in path.rglob("*.csv"):
    destination = TARGET / source.name
    shutil.copy2(source, destination)
    print(f"Copied {source.name} -> {destination}")
