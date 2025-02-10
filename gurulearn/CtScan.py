import cv2
import numpy as np
import os
from scipy import ndimage
class CTScanProcessor:
    def __init__(self, kernel_size=5, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.kernel_size = kernel_size
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def sharpen(self, image):
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        return cv2.filter2D(image, -1, kernel)

    def median_denoise(self, image):
        return ndimage.median_filter(image, size=self.kernel_size)

    def enhance_contrast(self, image):
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        return clahe.apply(image)

    def enhanced_denoise(self, image_path):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError("Could not read the image")

        denoised = self.median_denoise(image)
        denoised = self.enhance_contrast(denoised)
        denoised = self.sharpen(denoised)
        return denoised

    def evaluate_quality(self, original, denoised):
        if original is None or denoised is None:
            raise ValueError("Original or denoised image is None.")

        original = original.astype(float)
        denoised = denoised.astype(float)

        mse = np.mean((original - denoised) ** 2) + 1e-10
        max_pixel = 255.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
        
        signal_power = np.mean(denoised ** 2)
        noise_power = np.mean((original - denoised) ** 2)
        snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else 100
        
        detail_orig = np.std(original)
        detail_denoise = np.std(denoised)
        detail_ratio = detail_denoise / detail_orig if detail_orig > 0 else 1
        
        return {
            'MSE': mse,
            'PSNR': psnr,
            'SNR': snr,
            'Detail_Preservation': detail_ratio * 100  
        }

    def compare_images(self, original, processed, output_path):
        """Save a side-by-side comparison of the original and processed images."""
        if original is None or processed is None:
            raise ValueError("Original or processed image is None.")
        
        comparison = np.hstack((original, processed))
        cv2.imwrite(output_path, comparison)
        return comparison

    def print_best_metrics(self, metrics):
        if metrics is None:
            print("No metrics to display.")
            return
        
        print("\nFinal metrics for best result:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.2f}")

    def process_ct_scan(self, input_path, output_folder, comparison_folder="comparison", compare=False):
        try:
            os.makedirs(output_folder, exist_ok=True)
            if compare and comparison_folder:
                os.makedirs(comparison_folder, exist_ok=True)

            
            original = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
            if original is None:
                raise ValueError("Could not read the original image")
            
            
            denoised = self.enhanced_denoise(input_path)
            metrics = self.evaluate_quality(original, denoised)

            print(f"\nDenoising metrics:")
            for metric, value in metrics.items():
                print(f"{metric}: {value:.2f}")
            
            
            output_path = os.path.join(output_folder, os.path.basename(input_path).replace('.jpg', '_denoised.jpg'))
            cv2.imwrite(output_path, denoised)

            
            if compare and comparison_folder:
                comparison_path = os.path.join(comparison_folder, os.path.basename(input_path).replace('.jpg', '_comparison.jpg'))
                self.compare_images(original, denoised, comparison_path)

            self.print_best_metrics(metrics)

            return denoised, metrics
            
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            return None, None
