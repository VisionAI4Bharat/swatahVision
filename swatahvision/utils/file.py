from pathlib import Path
from platformdirs import user_cache_dir
from swatahvision.config import APP_NAME, APP_AUTHOR

def get_cache_dir() -> Path:
    cache_dir = Path(user_cache_dir(APP_NAME, APP_AUTHOR))
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir

def find_file_in_cache(file_name: str) -> Path:
    cache_dir = get_cache_dir()
    file_path = cache_dir / file_name
    is_file = Path(file_path).is_file()
    return is_file
