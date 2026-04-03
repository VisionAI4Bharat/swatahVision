import os
import requests

from swatahvision.assets.list import ImageAssets, VideoAssets

BASE_DIR = "assets"
IMAGE_DIR = os.path.join(BASE_DIR, "images")
VIDEO_DIR = os.path.join(BASE_DIR, "videos")

os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(VIDEO_DIR, exist_ok=True)


def get_image(name):
    asset = getattr(ImageAssets, name.upper(), None)
    if not asset:
        raise ValueError("Image not found")

    return download_asset(asset, "image")


def get_video(name):
    asset = getattr(VideoAssets, name.upper(), None)
    if not asset:
        raise ValueError("Video not found")

    return download_asset(asset, "video")

    
def download_asset(asset, asset_type="image"):
    filename = asset["filename"]
    url = asset["url"]

    if asset_type == "image":
        local_path = os.path.join(IMAGE_DIR, filename)
    else:
        local_path = os.path.join(VIDEO_DIR, filename)

    # ✅ Already exists
    if os.path.exists(local_path):
        return local_path

    print(f"Downloading {filename}...")
    print("URL:",url)
    response = requests.get(url,headers={"User-Agent":"Mozilla/5.0"})
    print("Status Code:", response.status_code)
    if response.status_code != 200:
        print("Response Text:",response.status_code)
        raise Exception("Download failed")

    with open(local_path, "wb") as f:
        f.write(response.content)

    return local_path
if __name__ == "__main__":
    print("Downloading all assets...")

    # Download all images
    for name, asset in ImageAssets.__dict__.items():
        if not name.startswith("__"):
            download_asset(asset, "image")

    # Download all videos
    for name, asset in VideoAssets.__dict__.items():
        if not name.startswith("__"):
            download_asset(asset, "video")