

# ImageTranslationDataValidator
"""

import numpy as np
from typing import Union, Optional
# from scipy.ndimage import label
import torch
import warnings


list_of_metrics = ['psnr', 'ssim']


class ImageTranslationDataValidator:
    def __init__(self, raise_warning: bool = True):
        self.raise_warning = raise_warning
        self._supported_dtypes = (np.float32, np.float64, np.uint8, np.uint16)

    def _ensure_numpy(self, data: Union[np.ndarray, list, 'torch.Tensor']) -> np.ndarray:
        """Smart conversion to float32 while preserving value range

        Parameters:
        -----------
        data : Union[np.ndarray, list, torch.Tensor]
            Input data that can be:
            - numpy array
            - Python list
            - PyTorch tensor

        Returns:
        --------
        np.ndarray
            Converted numpy array in float32 format
        """
        # Convert list to numpy array first
        if isinstance(data, list):
            data = np.array(data)

        # Handle torch tensors
        if str(type(data)) == "<class 'torch.Tensor'>":
            data = data.detach().cpu().numpy()

        # Ensure we have a numpy array at this point
        if not isinstance(data, np.ndarray):
            raise TypeError(f"Unsupported input type: {type(data)}. "
                          f"Supported: list, numpy.ndarray, torch.Tensor")

        # Check supported dtypes
        if data.dtype not in self._supported_dtypes:
            raise TypeError(f"Unsupported dtype: {data.dtype}. Supported: {self._supported_dtypes}")

        # Automatic conversion to float32 while preserving values
        if data.dtype == np.uint8:
            return data.astype(np.float32) / 255.0
        elif data.dtype == np.uint16:
            return data.astype(np.float32) / 65535.0
        return data.astype(np.float32, copy=False)

    def _auto_detect_range(self, img: np.ndarray) -> tuple[float, float]:
        """Automatic detection of image value range"""
        if img.dtype == np.uint8:
            return (0, 255)
        elif img.dtype == np.uint16:
            return (0, 65535)
        return (float(np.min(img)), float(np.max(img)))

    def validate_all(
        self,
        img1: Union[np.ndarray, 'torch.Tensor'],
        img2: Union[np.ndarray, 'torch.Tensor'],
        expected_range: Optional[tuple[float, float]] = None,
        check_channels: bool = True,
        auto_scale: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        """Advanced validation with features:
        - Automatic range detection
        - Smart data type conversion
        - Automatic normalization
        """
        img1 = self._ensure_numpy(img1)
        img2 = self._ensure_numpy(img2)

        # Automatic range detection if not specified
        if expected_range is None:
            range1 = self._auto_detect_range(img1)
            range2 = self._auto_detect_range(img2)
            expected_range = (min(range1[0], range2[0]), max(range1[1], range2[1]))

        # Dimension check
        if img1.shape != img2.shape:
            raise ValueError(f"Shape mismatch: {img1.shape} vs {img2.shape}")

        # Channel check
        if check_channels and img1.shape[-1] != img2.shape[-1]:
            if self.raise_warning:
                warnings.warn(f"Channel mismatch: {img1.shape[-1]} vs {img2.shape[-1]}", UserWarning)

        # Automatic normalization
        if auto_scale:
            img1 = (img1 - expected_range[0]) / (expected_range[1] - expected_range[0])
            img2 = (img2 - expected_range[0]) / (expected_range[1] - expected_range[0])
            expected_range = (0.0, 1.0)

        # Final range check
        for img, name in [(img1, "Image1"), (img2, "Image2")]:
            current_min, current_max = np.min(img), np.max(img)
            if current_min < expected_range[0] - 1e-6 or current_max > expected_range[1] + 1e-6:
                raise ValueError(
                    f"{name} pixel range violation: "
                    f"Expected [{expected_range[0]:.2f}, {expected_range[1]:.2f}], "
                    f"got [{current_min:.2f}, {current_max:.2f}]"
                )

        return img1, img2


# Metadata for image-to-image translation metrics
IMAGETOIMAGE_METRIC_DETAILS = {
    "psnr": {
        "image_true": "Ground truth (correct) image. Expected as a NumPy array, list, or PyTorch tensor.",
        "image_test": "Test image. Expected as a NumPy array, list, or PyTorch tensor.",
        "data_range": "Optional float specifying the data range of the input image (max possible value).",
        "validator": "Optional ImageTranslationDataValidator instance for input validation.",
        "validator_kwargs": "Additional keyword arguments for the validator."
    },
    "ssim": {
        "img1": "First input image (ground truth). Expected as a NumPy array.",
        "img2": "Second input image (test image). Expected as a NumPy array.",
        "window_size": "Size of the sliding window used for SSIM computation. Default is 11.",
        "dynamic_range": "Dynamic range of the input images. Default is 255.0 for uint8 images.",
        "multichannel": "If True, treat the last dimension as channels. Default is False.",
        "validator": "Optional ImageTranslationDataValidator instance for input validation.",
        "auto_scale": "If True, automatically scale input images to [0, 1] range. Default is True."
    }
}

def get_metric_details(metric_name):
    """
    Returns the parameter descriptions for a given metric.
    """
    if metric_name in IMAGETOIMAGE_METRIC_DETAILS:
        return IMAGETOIMAGE_METRIC_DETAILS[metric_name]
    else:
        raise ValueError(f"Metric '{metric_name}' not found in the image-to-image translation module.")