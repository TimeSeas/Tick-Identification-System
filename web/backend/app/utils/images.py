"""Normalize image orientation consistently for detection and annotation."""
import io
import warnings
from PIL import Image, ImageOps, UnidentifiedImageError

MAX_IMAGE_PIXELS = 20_000_000

class ImageValidationError(ValueError):
    pass

def load_rgb_image(data: bytes) -> Image.Image:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as image:
                if image.width * image.height > MAX_IMAGE_PIXELS:
                    raise ImageValidationError('图片像素超过2000万，请缩小后重试')
                image.load()
                return ImageOps.exif_transpose(image).convert('RGB')
    except ImageValidationError:
        raise
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombWarning,
            Image.DecompressionBombError) as error:
        raise ImageValidationError('图片无法解码，请上传完整、有效的图片文件') from error
