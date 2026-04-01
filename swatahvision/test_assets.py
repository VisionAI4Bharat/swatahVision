import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from swatahvision.assets.downloader import get_image, get_video

img = get_image("car")
print("Image:", img)

vid = get_video("vehicles")
print("Video:", vid)