from dataclasses import dataclass
from typing import Optional, Literal


@dataclass(frozen=True, slots=True)
class AssetInfo:
    """Immutable asset metadata."""
    filename: str
    url: str
    asset_type: Literal["image", "video", "model"]
    checksum: Optional[str] = None
    size: Optional[int] = None
    version: str = "1.0"


class ImageAssets:
    """Image assets registry."""
    CAR = AssetInfo(
        filename="car.jpg",
        url="https://huggingface.co/datasets/swatah/swatahVision-assets/resolve/main/images/car.jpg",
        asset_type="image",
        size=217681,
    )
    DOG = AssetInfo(
        filename="dog.jpg",
        url="https://huggingface.co/datasets/swatah/swatahVision-assets/resolve/main/images/dog.jpg",
        asset_type="image",
    )
    PEOPLE_WALKING = AssetInfo(
        filename="people-walking.jpg",
        url="https://huggingface.co/datasets/swatah/swatahVision-assets/resolve/main/images/people-walking.jpg",
        asset_type="image",
    )
    SOCCER = AssetInfo(
        filename="soccer.jpg",
        url="https://huggingface.co/datasets/swatah/swatahVision-assets/resolve/main/images/soccer.jpg",
        asset_type="image",
    )


class VideoAssets:
    """Video assets registry."""
    VEHICLES = AssetInfo(
        filename="vehicles.mp4",
        url="https://huggingface.co/swatah/swatahVision/resolve/main/videos/vehicles.mp4",
        asset_type="video",
    )
    TRAFFIC = AssetInfo(
        filename="traffic.mp4",
        url="https://huggingface.co/swatah/swatahVision/resolve/main/videos/vehicles.mp4",
        asset_type="video",
    )
    VEHICLES_2 = AssetInfo(
        filename="vehicles-2.mp4",
        url="https://huggingface.co/datasets/swatah/swatahVision-assets/resolve/main/videos/vehicles-2.mp4",
        asset_type="video",
    )
    SUBWAY = AssetInfo(
        filename="subway.mp4",
        url="https://huggingface.co/datasets/swatah/swatahVision-assets/resolve/main/videos/subway.mp4",
        asset_type="video",
    )
    SKIING = AssetInfo(
        filename="skiing.mp4",
        url="https://huggingface.co/datasets/swatah/swatahVision-assets/resolve/main/videos/skiing.mp4",
        asset_type="video",
    )
    PEOPLE_WALKING = AssetInfo(
        filename="people-walking.mp4",
        url="https://huggingface.co/datasets/swatah/swatahVision-assets/resolve/main/videos/people-walking.mp4",
        asset_type="video",
    )
    MILK_BOTTLING_PLANT = AssetInfo(
        filename="milk-bottling-plant.mp4",
        url="https://huggingface.co/datasets/swatah/swatahVision-assets/resolve/main/videos/milk-bottling-plant.mp4",
        asset_type="video",
    )
    MARKET_SQUARE = AssetInfo(
        filename="market-square.mp4",
        url="https://huggingface.co/datasets/swatah/swatahVision-assets/resolve/main/videos/market-square.mp4",
        asset_type="video",
    )
    GROCERY_STORE = AssetInfo(
        filename="grocery-store.mp4",
        url="https://huggingface.co/datasets/swatah/swatahVision-assets/resolve/main/videos/grocery-store.mp4",
        asset_type="video",
    )
    BEACH = AssetInfo(
        filename="beach-1.mp4",
        url="https://huggingface.co/datasets/swatah/swatahVision-assets/resolve/main/videos/beach-1.mp4",
        asset_type="video",
    )
    BASKETBALL = AssetInfo(
        filename="basketball-1.mp4",
        url="https://huggingface.co/datasets/swatahVision-assets/resolve/main/videos/basketball-1.mp4",
        asset_type="video",
    )


class ModelAssets:
    """Model assets registry."""
    MOBILENETV2 = AssetInfo(
        filename="mobilenetv2_f32.onnx",
        url="https://huggingface.co/swatah/swatahVision/resolve/main/classifiation/mobilenetv2/mobilenetv2_f32.onnx",
        asset_type="model",
    )
    RESNET18 = AssetInfo(
        filename="resnet18_f32.onnx",
        url="https://huggingface.co/swatah/swatahVision/resolve/main/classifiation/resnet18/resnet18_f32.onnx",
        asset_type="model",
    )
    SSD_LITE_MOBILENETV3 = AssetInfo(
        filename="ssdlite-mobilenetv3_f32.onnx",
        url="https://huggingface.co/swatah/swatahVision/resolve/main/detection/ssdlite-mobilenetv3/ssdlite-mobilenetv3_f32.onnx",
        asset_type="model",
    )
    RETINANET_RESNET50_FPN = AssetInfo(
        filename="retinanet-resnet50-fpn-512.onnx",
        url="https://huggingface.co/swatah/swatahVision/resolve/main/detection/retinanet/retinanet-resnet50-fpn-512.onnx",
        asset_type="model",
    )


class Assets:
    """Asset registry with logical namespaces."""
    Image = ImageAssets
    Video = VideoAssets
    Model = ModelAssets
