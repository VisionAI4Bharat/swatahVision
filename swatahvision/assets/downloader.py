import os
from .list import IMAGE_ASSETS, VIDEO_ASSETS


class ImageAssets:

    def __init__(self):
        self.base_dir = os.path.join(
            os.path.dirname(__file__), "images"
        )

    def get(self, name: str):

        name = name.upper()

        if name not in IMAGE_ASSETS:
            raise ValueError(f"[ERROR] Image asset '{name}' not found")

        file_name = IMAGE_ASSETS[name]
        file_path = os.path.join(self.base_dir, file_name)

        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"[ERROR] File '{file_name}' missing in assets/images"
            )

        return file_path


class VideoAssets:

    def __init__(self):
        self.base_dir = os.path.join(
            os.path.dirname(__file__), "videos"
        )

    def get(self, name: str):

        name = name.upper()

        if name not in VIDEO_ASSETS:
            raise ValueError(f"[ERROR] Video asset '{name}' not found")

        file_name = VIDEO_ASSETS[name]
        file_path = os.path.join(self.base_dir, file_name)

        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"[ERROR] File '{file_name}' missing in assets/videos"
            )

        return file_path


class Assets:

    @staticmethod
    def Image(name: str):
        return ImageAssets().get(name)

    @staticmethod
    def Video(name: str):
        return VideoAssets().get(name)