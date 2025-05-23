import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image
from typing import List, Optional, Tuple, Union
from supervision import Detections, Color, ColorPalette, Position, ColorLookup, resolve_color
from supervision.draw.color import CLASS_NAME_DATA_FIELD


class OCRALabelAnnotator:
    def __init__(
        self,
        font_path: str = "OCR-A.ttf",
        font_size: int = 24,
        text_color: Tuple[int, int, int] = (0, 255, 0),  # Vert militaire
        text_padding: int = 10,
        text_position: Position = Position.TOP_LEFT,
        color: Union[Color, ColorPalette] = ColorPalette.DEFAULT,
        color_lookup: ColorLookup = ColorLookup.CLASS,
    ):
        self.font = ImageFont.truetype(font_path, size=font_size)
        self.text_color = text_color
        self.text_padding = text_padding
        self.text_position = text_position
        self.color = color
        self.color_lookup = color_lookup

    @staticmethod
    def resolve_text_background_xyxy(center_coordinates, text_wh, position):
        center_x, center_y = center_coordinates
        text_w, text_h = text_wh

        if position == Position.TOP_LEFT:
            return center_x, center_y - text_h, center_x + text_w, center_y
        elif position == Position.TOP_RIGHT:
            return center_x - text_w, center_y - text_h, center_x, center_y
        elif position == Position.TOP_CENTER:
            return center_x - text_w // 2, center_y - text_h, center_x + text_w // 2, center_y
        elif position in [Position.CENTER, Position.CENTER_OF_MASS]:
            return center_x - text_w // 2, center_y - text_h // 2, center_x + text_w // 2, center_y + text_h // 2
        elif position == Position.BOTTOM_LEFT:
            return center_x, center_y, center_x + text_w, center_y + text_h
        elif position == Position.BOTTOM_RIGHT:
            return center_x - text_w, center_y, center_x, center_y + text_h
        elif position == Position.BOTTOM_CENTER:
            return center_x - text_w // 2, center_y, center_x + text_w // 2, center_y + text_h

    def annotate(
        self,
        scene: np.ndarray,
        detections: Detections,
        labels: Optional[List[str]] = None,
        custom_color_lookup: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        img_pil = Image.fromarray(cv2.cvtColor(scene, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        anchors_coordinates = detections.get_anchors_coordinates(anchor=self.text_position).astype(int)

        if labels is not None and len(labels) != len(detections):
            raise ValueError("Labels and detections count mismatch.")

        for i, center in enumerate(anchors_coordinates):
            color = resolve_color(
                color=self.color,
                detections=detections,
                detection_idx=i,
                color_lookup=self.color_lookup if custom_color_lookup is None else custom_color_lookup
            ).as_rgb()

            text = (
                labels[i] if labels is not None else
                detections[CLASS_NAME_DATA_FIELD][i]
                if detections[CLASS_NAME_DATA_FIELD] is not None else
                str(detections.class_id[i]) if detections.class_id is not None else str(i)
            )

            # Calculate size with padding
            text_size = draw.textbbox((0, 0), text, font=self.font)
            text_w = text_size[2] - text_size[0] + 2 * self.text_padding
            text_h = text_size[3] - text_size[1] + 2 * self.text_padding

            # Background box
            bg_x1, bg_y1, bg_x2, bg_y2 = self.resolve_text_background_xyxy(
                center_coordinates=tuple(center),
                text_wh=(text_w, text_h),
                position=self.text_position,
            )

            draw.rectangle(
                [(bg_x1, bg_y1), (bg_x2, bg_y2)],
                fill=color
            )

            # Text position inside box
            text_x = bg_x1 + self.text_padding
            text_y = bg_y1 + self.text_padding
            draw.text((text_x, text_y), text, font=self.font, fill=self.text_color)

        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
