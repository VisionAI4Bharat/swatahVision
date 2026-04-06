"""
Robust asset downloader with retries, checksum verification, and progress.
"""
import hashlib
import logging
import time
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm

from swatahVision.utils.file import get_cache_dir
from swatahvision.assets.list import AssetInfo

logger = logging.getLogger(__name__)


class AssetError(Exception):
    pass


class DownloadFailedError(AssetError):
    pass


class ChecksumMismatchError(AssetError):
    pass


def _sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(81920):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path, expected_checksum: Optional[str] = None) -> Path:
    temp = dest.with_suffix(dest.suffix + ".tmp")

    for attempt in range(1, 4):
        try:
            logger.info(f"Downloading {dest.name} (attempt {attempt})")

            headers = {"User-Agent": "swatahVision"}
            start = temp.stat().st_size if temp.exists() else 0
            if start:
                headers["Range"] = f"bytes={start}-"

            with requests.get(url, stream=True, timeout=30, headers=headers) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0)) + start
                mode = "ab" if start else "wb"

                with open(temp, mode) as f, tqdm(
                    total=total,
                    initial=start,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=dest.name[:30],
                    ncols=80,
                ) as bar:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            bar.update(len(chunk))

            if expected_checksum and _sha256(temp) != expected_checksum:
                temp.unlink(missing_ok=True)
                raise ChecksumMismatchError(f"Checksum mismatch: {dest.name}")

            temp.rename(dest)
            logger.info(f"Downloaded {dest.name}")
            return dest

        except Exception as e:
            logger.warning(f"Attempt {attempt} failed: {e}")
            if attempt < 3:
                time.sleep(2 ** attempt)
            else:
                temp.unlink(missing_ok=True)
                raise DownloadFailedError(f"Failed: {url}") from e


class AssetDownloader:
    def __init__(self):
        base = get_cache_dir() / "swatahVision"
        self.dirs = {
            "image": base / "images",
            "video": base / "videos",
            "model": base / "models",
        }
        for d in self.dirs.values():
            d.mkdir(parents=True, exist_ok=True)

    def get(self, asset: AssetInfo) -> Path:
        path = self.dirs[asset.asset_type] / asset.filename
        if not path.exists():
            _download(asset.url, path, asset.checksum)
        return path


# Singleton
_downloader: Optional[AssetDownloader] = None


def get_downloader() -> AssetDownloader:
    global _downloader
    if _downloader is None:
        _downloader = AssetDownloader()
    return _downloader


def get_asset(asset) -> Path:
    """Get any asset (image, video, model) by AssetInfo."""
    return get_downloader().get(asset)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    from swatahvision.assets.list import Assets

    dl = AssetDownloader()
    print("Downloading all assets...")
    for category in (Assets.Image, Assets.Video, Assets.Model):
        for name in dir(category):
            if name.startswith("__"):
                continue
            asset = getattr(category, name)
            if isinstance(asset, AssetInfo):
                try:
                    dl.get(asset)
                    print(f"  ✓ {asset.filename}")
                except Exception as e:
                    print(f"  ✗ {asset.filename}: {e}")