import av
import os
import ast
import numpy as np
from pyzbar.pyzbar import decode
from utils.logger import setup_basic_logger


log = setup_basic_logger(os.path.basename(__file__))


BLANK_GREEN_COLOR = (0, 189, 0)
BLANK_BLUE_COLOR = (0, 0, 255)
BLANK_BLACK_COLOR = (0, 0, 0)


def is_blank_frame(img):
    rgb_medians = np.median(img, axis=(0, 1))

    # Account for impurity due to video compression
    mask = (
        np.isclose(img[:, :, 0], rgb_medians[0], atol=2) &
        np.isclose(img[:, :, 1], rgb_medians[1], atol=2) &
        np.isclose(img[:, :, 2], rgb_medians[2], atol=2)
    )

    return np.mean(mask) > 0.95


def is_color_frame(img, blank_color=BLANK_BLACK_COLOR):
    if len(img.shape) == 2:
        return False

    mask = (
        np.isclose(img[:, :, 0], blank_color[0], atol=2) &
        np.isclose(img[:, :, 1], blank_color[1], atol=2) &
        np.isclose(img[:, :, 2], blank_color[2], atol=2)
    )

    return np.mean(mask) > 0.95


def try_read_qr_code(img: np.ndarray):
    try:
        decoded_qr_code = decode(img)

        if len(decoded_qr_code) == 0:
            return None
        elif len(decoded_qr_code) > 1:
            log.warning("Found more than 1 QR code in the given image. Using the first QR code.")

        decoded_data = decoded_qr_code[0].data.decode("ascii")

        # Parse to dict
        # Note that eval can be dangerous
        return ast.literal_eval(decoded_data)
    except Exception as e:
        log.error(f"Error reading QR code: {e}")
        return None


def open_output_writer(filename: str, template_stream: av.video.stream.VideoStream) -> tuple[av.container.OutputContainer, av.video.stream.VideoStream]:
    output_container = av.open(filename, mode='w')
    output_stream = output_container.add_stream('libx264', rate=30)

    output_stream.width = template_stream.width
    output_stream.height = template_stream.height
    output_stream.pix_fmt = "yuv420p"
    output_stream.time_base = template_stream.time_base    # keep exact tbn

    output_stream.options = {
        "crf": "0",                          # Good quality, visually lossless
        "preset": "veryslow",                # Good balance between speed and compression
        "profile": "high444",                # High profile for better quality
        "refs": "1",                         # 1 reference frames for better compression
        "colorprim": "bt709",                # BT.709 for HD content
        "transfer": "bt709",
        "colormatrix": "bt709",
        "x264opts": "cabac=1",               # CABAC enabled
        "bf": "0",                           # no B frames
    }

    return output_container, output_stream
