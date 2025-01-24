import cv2
from mmcv import BaseTransform

from mmpose.registry import TRANSFORMS

@TRANSFORMS.register_module()
class ApplyGaussianBlur(BaseTransform):
    """Apply a Gaussian blur filter to an image.

    This transformation applies a Gaussian blur filter to the input image
    to smooth it, which can help reduce noise and improve downstream
    processing tasks.

    Required Keys:
        - img (numpy.ndarray): The input image to be processed.

    Modified Keys:
        - img (numpy.ndarray): The blurred image.

    Args:
        kernel_size (tuple[int, int]): The size of the Gaussian blur kernel.
            Defaults to (3, 3). """

    def __init__(self, kernel_size=(3, 3)):
        self.kernel_size = kernel_size

    def transform(self, results: dict) -> dict:
        img = results['img']
        img = cv2.GaussianBlur(img, self.kernel_size, 0)
        results['img'] = img
        return results
