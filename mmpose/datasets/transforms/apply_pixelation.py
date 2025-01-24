import cv2
from mmcv import BaseTransform

from mmpose.registry import TRANSFORMS


@TRANSFORMS.register_module()
class ApplyPixelation(BaseTransform):
    """Apply a pixelation filter to an image.

    This transformation reduces the resolution of the input image by resizing
    it to a smaller size and then scaling it back to its original dimensions.
    This results in a pixelated effect.

    Required Keys:
        - img (numpy.ndarray): The input image to be processed.

    Modified Keys:
        - img (numpy.ndarray): The pixelated image.

    Args:
        pixel_size (int): The size of the pixelation blocks. A larger value
            creates a more pixelated effect. Defaults to 10.
    """

    def __init__(self, pixel_size=4):
        self.pixel_size = pixel_size

    def transform(self, results: dict) -> dict:
        img = results['img']
        # Get the dimensions of the input image
        h, w = img.shape[:2]

        # Calculate the size of the reduced image
        reduced_w = max(1, w // self.pixel_size)
        reduced_h = max(1, h // self.pixel_size)

        # Resize to reduced dimensions
        img_small = cv2.resize(img, (reduced_w, reduced_h), interpolation=cv2.INTER_LINEAR)

        # Scale back to the original size
        img_pixelated = cv2.resize(img_small, (w, h), interpolation=cv2.INTER_NEAREST)

        # Update the results dictionary
        results['img'] = img_pixelated
        return results
