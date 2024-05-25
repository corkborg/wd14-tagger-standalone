# DanBooru IMage Utility functions

import numpy as np
import numpy.typing
from PIL import Image, ImageOps

def fill_transparent(image: Image.Image, color='WHITE'):
    image = image.convert('RGBA')
    new_image = Image.new('RGBA', image.size, color)
    new_image.paste(image, mask=image)
    image = new_image.convert('RGB')
    return image

def resize(pic: Image.Image, size: int, keep_ratio=True) -> Image.Image:
    if not keep_ratio:
        target_size = (size, size)
    else:
        min_edge = min(pic.size)
        target_size = (
            int(pic.size[0] / min_edge * size),
            int(pic.size[1] / min_edge * size),
        )

    target_size = (target_size[0] & ~3, target_size[1] & ~3)

    return pic.resize(target_size, resample=Image.Resampling.LANCZOS)

def smart_imread(img):
    if img.endswith(".gif"):
        with Image.open(img) as img_pil:
            img_pil = img_pil.convert("RGB")
            img_array = np.array(img_pil)
            img = img_array[:, :, ::-1]  # Convert RGB to BGR
    else:
        with Image.open(img) as img_pil:
            img = np.array(img_pil)
    return img

def smart_24bit(img: np.ndarray) -> np.ndarray:
    if img.dtype is np.dtype(np.uint16):
        img = (img / 257).astype(np.uint8)

    if len(img.shape) == 2:
        img = np.stack([img, img, img], axis=-1)
    elif img.shape[2] == 4:
        alpha_channel = img[:, :, 3]
        trans_mask = alpha_channel == 0
        img[trans_mask] = [255, 255, 255, 255]  # Fill transparent areas with white
        img = img[:, :, :3]  # Drop the alpha channel
    return img

def make_square(img: np.ndarray, target_size: int):
    img_pil = Image.fromarray(img)

    old_size = img_pil.size
    desired_size = max(old_size)
    desired_size = max(desired_size, target_size)

    delta_w = desired_size - old_size[0]
    delta_h = desired_size - old_size[1]
    pad_left, pad_right = delta_w // 2, delta_w - (delta_w // 2)
    pad_top, pad_bottom = delta_h // 2, delta_h - (delta_h // 2)

    padding = (pad_left, pad_top, pad_right, pad_bottom)
    img_square = ImageOps.expand(img_pil, padding, fill='white')

    return np.array(img_square)

def smart_resize(img: np.ndarray, size):
    img_pil = Image.fromarray(img)
    if max(img_pil.size) > size:
        # TODO: cv2.INTER_AREA is not available in PIL
        img_resized = img_pil.resize((size, size), Image.LANCZOS)
        return img_resized
    elif max(img_pil.size) < size:
        # TODO: cv2.INTER_AREA is not available in PIL
        img_resized = img_pil.resize((size, size), Image.BICUBIC)
    else:
        img_resized = img_pil

    return np.array(img_resized)