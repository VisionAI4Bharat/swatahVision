import cv2
import numpy as np
import requests

class Image(np.ndarray):
    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls)
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return

    # ---------- Factory methods ----------
    @classmethod
    def load_from_file(cls, path: str):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Failed to load image: {path}")
        return cls(img)

    @classmethod
    def load_from_url(cls, url: str, timeout: int = 10):
        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
        except requests.RequestException as e:
            raise ValueError(f"Download failed: {e}")

        arr = np.frombuffer(r.content, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Invalid image data")

        return cls(img)

    # ---------- Image ops ----------
    def resize(self, width: int, height: int):
        return Image(cv2.resize(self, (width, height)))

    def to_rgb(self):
        return Image(cv2.cvtColor(self, cv2.COLOR_BGR2RGB))

    def to_gray(self):
        return Image(cv2.cvtColor(self, cv2.COLOR_BGR2GRAY))

    def normalize(self):
        return Image(self.astype(np.float32) / 255.0)
    
    # ---------- Display ----------
    @staticmethod
    def show(
        image: "Image",
        window_name: str = "Image",
        wait: int = 0,
        destroy: bool = True,
    ):
        """
        Show image using cv2.imshow

        wait = 0  -> wait forever
        wait > 0  -> wait N milliseconds
        """
        if not isinstance(image, np.ndarray):
            raise TypeError("Input must be a cv2 / numpy image")

        cv2.imshow(window_name, image)
        cv2.waitKey(wait)

        if destroy:
            cv2.destroyWindow(window_name)