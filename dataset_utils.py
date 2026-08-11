from pathlib import Path
from zipfile import ZipFile

import gdown

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = REPO_ROOT / "data"
MALNET_DOWNLOAD_URL = "https://drive.google.com/uc?id=1PtgcLHhluxbvqK3mEyP-MECmWEFALhft"


def normalize_dataset_name(dataset: str) -> str:
    return dataset.lower()


def resolve_dataset_root(dataset: str, dataset_path: str | None = None) -> Path:
    dataset = normalize_dataset_name(dataset)

    if dataset == "cifar10":
        return _resolve_data_dir(dataset_path, "cifar")

    if dataset == "malimg":
        return _resolve_data_dir(dataset_path, "malimg")

    if dataset == "malnet":
        return _resolve_malnet_dir(dataset_path)

    raise ValueError(f"Unsupported dataset: {dataset}")


def _resolve_data_dir(dataset_path: str | None, folder_name: str) -> Path:
    if dataset_path is None:
        root = DEFAULT_DATA_ROOT / folder_name
    else:
        root = Path(dataset_path).expanduser()

    root.mkdir(parents=True, exist_ok=True)
    return root


def _resolve_malnet_dir(dataset_path: str | None) -> Path:
    if dataset_path is None:
        root = DEFAULT_DATA_ROOT / "malnet_resized_32x256"
    else:
        root = Path(dataset_path).expanduser()

    if not root.exists() or not any(root.iterdir()):
        root.mkdir(parents=True, exist_ok=True)
        archive_path = root.parent / "MalNet.zip"
        if not archive_path.exists():
            gdown.download(MALNET_DOWNLOAD_URL, str(archive_path), quiet=False)
        with ZipFile(archive_path, "r") as zobject:
            zobject.extractall(root.parent)
        archive_path.unlink(missing_ok=True)

    return root
