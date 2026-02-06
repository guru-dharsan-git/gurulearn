"""
CtScan - CT Scan image processing and denoising.

Provides tools for medical image enhancement, denoising, and quality evaluation.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy import ndimage


@dataclass
class QualityMetrics:
    """Quality metrics for image processing evaluation."""
    mse: float
    psnr: float
    snr: float
    detail_preservation: float
    
    def __str__(self) -> str:
        return (
            f"MSE: {self.mse:.2f}, PSNR: {self.psnr:.2f} dB, "
            f"SNR: {self.snr:.2f} dB, Detail: {self.detail_preservation:.1f}%"
        )
    
    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary."""
        return {
            "MSE": self.mse,
            "PSNR": self.psnr,
            "SNR": self.snr,
            "Detail_Preservation": self.detail_preservation
        }


@dataclass 
class ProcessingResult:
    """Result of CT scan processing."""
    original: np.ndarray
    processed: np.ndarray
    metrics: QualityMetrics
    output_path: Path | None = None
    comparison_path: Path | None = None


class CTScanProcessor:
    """
    CT Scan image processor for denoising and enhancement.
    
    Args:
        kernel_size: Size of the median filter kernel (default: 5)
        clip_limit: CLAHE clip limit for contrast enhancement (default: 2.0)
        tile_grid_size: CLAHE tile grid size (default: (8, 8))
        
    Example:
        >>> processor = CTScanProcessor()
        >>> result = processor.process_ct_scan(
        ...     "scan.jpg", 
        ...     output_folder="processed",
        ...     compare=True
        ... )
        >>> print(result.metrics)
    """
    
    # Supported image formats
    SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

    def __init__(
        self, 
        kernel_size: int = 5, 
        clip_limit: float = 2.0, 
        tile_grid_size: tuple[int, int] = (8, 8)
    ):
        self.kernel_size = kernel_size
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def sharpen(self, image: np.ndarray) -> np.ndarray:
        """
        Apply sharpening filter to enhance edges.
        
        Args:
            image: Input grayscale image
            
        Returns:
            Sharpened image
        """
        kernel = np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ], dtype=np.float32)
        return cv2.filter2D(image, -1, kernel)

    def median_denoise(self, image: np.ndarray) -> np.ndarray:
        """
        Apply median filter for noise reduction.
        
        Args:
            image: Input image
            
        Returns:
            Denoised image
        """
        return ndimage.median_filter(image, size=self.kernel_size)

    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE for adaptive contrast enhancement.
        
        Args:
            image: Input grayscale image
            
        Returns:
            Contrast-enhanced image
        """
        clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit, 
            tileGridSize=self.tile_grid_size
        )
        return clahe.apply(image)

    def bilateral_denoise(self, image: np.ndarray, d: int = 9, sigma_color: float = 75, sigma_space: float = 75) -> np.ndarray:
        """
        Apply bilateral filter for edge-preserving denoising.
        
        Args:
            image: Input image
            d: Diameter of each pixel neighborhood
            sigma_color: Filter sigma in the color space
            sigma_space: Filter sigma in the coordinate space
            
        Returns:
            Denoised image with preserved edges
        """
        return cv2.bilateralFilter(image, d, sigma_color, sigma_space)

    def enhanced_denoise(self, image_path: str | Path) -> np.ndarray:
        """
        Apply complete denoising pipeline to an image.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Processed image
            
        Raises:
            ValueError: If image cannot be read
        """
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not read the image: {image_path}")

        denoised = self.median_denoise(image)
        denoised = self.enhance_contrast(denoised)
        denoised = self.sharpen(denoised)
        return denoised

    def evaluate_quality(self, original: np.ndarray, processed: np.ndarray) -> QualityMetrics:
        """
        Evaluate quality metrics between original and processed images.
        
        Args:
            original: Original image
            processed: Processed image
            
        Returns:
            QualityMetrics with MSE, PSNR, SNR, and detail preservation
            
        Raises:
            ValueError: If either image is None
        """
        if original is None or processed is None:
            raise ValueError("Original or processed image is None")

        original_float = original.astype(np.float64)
        processed_float = processed.astype(np.float64)

        # Mean Squared Error
        mse = float(np.mean((original_float - processed_float) ** 2) + 1e-10)
        
        # Peak Signal-to-Noise Ratio
        max_pixel = 255.0
        psnr = float(20 * np.log10(max_pixel / np.sqrt(mse)))
        
        # Signal-to-Noise Ratio
        signal_power = np.mean(processed_float ** 2)
        noise_power = np.mean((original_float - processed_float) ** 2)
        snr = float(10 * np.log10(signal_power / noise_power) if noise_power > 0 else 100)
        
        # Detail Preservation
        detail_orig = np.std(original_float)
        detail_processed = np.std(processed_float)
        detail_ratio = float(detail_processed / detail_orig if detail_orig > 0 else 1) * 100

        return QualityMetrics(
            mse=mse,
            psnr=psnr,
            snr=snr,
            detail_preservation=detail_ratio
        )

    def compare_images(
        self, 
        original: np.ndarray, 
        processed: np.ndarray, 
        output_path: str | Path
    ) -> np.ndarray:
        """
        Create side-by-side comparison of original and processed images.
        
        Args:
            original: Original image
            processed: Processed image
            output_path: Path to save the comparison
            
        Returns:
            Comparison image
            
        Raises:
            ValueError: If either image is None
        """
        if original is None or processed is None:
            raise ValueError("Original or processed image is None")
        
        comparison = np.hstack((original, processed))
        cv2.imwrite(str(output_path), comparison)
        return comparison

    def process_ct_scan(
        self, 
        input_path: str | Path, 
        output_folder: str | Path,
        comparison_folder: str | Path | None = None,
        compare: bool = False,
        verbose: bool = True
    ) -> ProcessingResult:
        """
        Process a CT scan image with denoising and enhancement.
        
        Args:
            input_path: Path to the input image
            output_folder: Directory to save processed images
            comparison_folder: Directory for comparison images (optional)
            compare: Whether to create comparison images
            verbose: Print processing information
            
        Returns:
            ProcessingResult with original, processed images and metrics
            
        Raises:
            ValueError: If image cannot be read
            FileNotFoundError: If input file doesn't exist
        """
        input_path = Path(input_path)
        output_folder = Path(output_folder)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        # Create output directories
        output_folder.mkdir(parents=True, exist_ok=True)
        
        if compare and comparison_folder:
            comparison_folder = Path(comparison_folder)
            comparison_folder.mkdir(parents=True, exist_ok=True)

        # Read original image
        original = cv2.imread(str(input_path), cv2.IMREAD_GRAYSCALE)
        if original is None:
            raise ValueError(f"Could not read the original image: {input_path}")
        
        # Process the image
        processed = self.enhanced_denoise(input_path)
        metrics = self.evaluate_quality(original, processed)

        if verbose:
            print(f"\nDenoising metrics for {input_path.name}:")
            print(str(metrics))
        
        # Determine output filename (preserve original format or default to png)
        suffix = input_path.suffix if input_path.suffix.lower() in self.SUPPORTED_FORMATS else ".png"
        output_name = input_path.stem + "_denoised" + suffix
        output_path = output_folder / output_name
        cv2.imwrite(str(output_path), processed)
        
        # Create comparison if requested
        comparison_path = None
        if compare and comparison_folder:
            comparison_folder = Path(comparison_folder)
            comparison_name = input_path.stem + "_comparison" + suffix
            comparison_path = comparison_folder / comparison_name
            self.compare_images(original, processed, comparison_path)

        return ProcessingResult(
            original=original,
            processed=processed,
            metrics=metrics,
            output_path=output_path,
            comparison_path=comparison_path
        )

    def process_batch(
        self,
        input_folder: str | Path,
        output_folder: str | Path,
        comparison_folder: str | Path | None = None,
        compare: bool = False,
        verbose: bool = True
    ) -> list[ProcessingResult]:
        """
        Process all supported images in a folder.
        
        Args:
            input_folder: Directory containing input images
            output_folder: Directory to save processed images
            comparison_folder: Directory for comparison images (optional)
            compare: Whether to create comparison images
            verbose: Print processing information
            
        Returns:
            List of ProcessingResult for each processed image
        """
        input_folder = Path(input_folder)
        results = []
        
        for file_path in input_folder.iterdir():
            if file_path.suffix.lower() in self.SUPPORTED_FORMATS:
                try:
                    result = self.process_ct_scan(
                        file_path,
                        output_folder,
                        comparison_folder,
                        compare,
                        verbose
                    )
                    results.append(result)
                except Exception as e:
                    if verbose:
                        print(f"Error processing {file_path.name}: {e}")
        
        if verbose:
            print(f"\nProcessed {len(results)} images")
        
        return results
