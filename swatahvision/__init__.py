from swatahVision.model.base import Model

from swatahVision.core.detection.core import Detections
from swatahVision.core.classification.core import Classification

from swatahVision.draw.ui import UI

from swatahVision.annotators.box import BoxAnnotator
from swatahVision.annotators.label import LabelAnnotator
from swatahVision.annotators.trace import TraceAnnotator
from swatahVision.annotators.color import ColorAnnotator
from swatahVision.annotators.hitmap import HeatMapAnnotator

from swatahVision.draw.image import Image
from swatahVision.draw.color import ColorPalette, Color

from swatahVision.draw.utils import (
    calculate_optimal_line_thickness,
    calculate_optimal_text_scale,
    draw_filled_polygon,
    draw_filled_rectangle,
    draw_image,
    draw_line,
    draw_polygon,
    draw_rectangle,
    draw_text,
)

from swatahVision.core.detection.tools.polygon_zone import PolygonZone, PolygonZoneAnnotator

from swatahVision.geometry.core import Position, Point
from swatahVision.geometry.utils import get_polygon_center
from swatahVision.constraints import Engine, Hardware

from swatahVision.tracker.byte_tracker.core import ByteTrack

from swatahVision.utils.video import (
    FPSMonitor,
    VideoInfo,
    VideoSink,
    get_video_frames_generator,
    process_video,
)

from swatahVision.config import APP_NAME, APP_AUTHOR
from swatahVision.utils.file import get_cache_dir

print(f"[debug] Cache dir: {get_cache_dir()}")

__all__ = [
    "APP_NAME", 
    "APP_AUTHOR", 
    "Model", 
    "Detections", 
    "Classification", 
    "UI", 
    "ColorPalette", 
    "Color",
    "calculate_optimal_line_thickness",
    "calculate_optimal_text_scale",
    "draw_filled_polygon",
    "draw_filled_rectangle",
    "draw_image",
    "draw_line",
    "draw_polygon",
    "draw_rectangle",
    "draw_text",
    "PolygonZone", 
    "PolygonZoneAnnotator"
    "Position",
    "BoxAnnotator",
    "LabelAnnotator",
    "TraceAnnotator",
    "ColorAnnotator",
    "HeatMapAnnotator",
    "Image",
    "Position",
    "Point",
    "get_polygon_center",
    "Engine",
    "Hardware",
    "ByteTrack",
    "FPSMonitor",
    "VideoInfo",
    "VideoSink",
    "get_video_frames_generator",
    "process_video",
    ]