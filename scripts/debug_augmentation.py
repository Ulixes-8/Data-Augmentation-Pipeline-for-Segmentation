#!/usr/bin/env python
"""
Script: debug_augmentation_fixed.py

Fixed debug version of augment_dataset.py with extensive logging and visualization
to diagnose issues with mask processing during augmentation.
"""

import argparse
import os
import sys
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set

import cv2
import numpy as np
import albumentations as A
from PIL import Image
from tqdm import tqdm

# Add the project root to path to import our modules
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.utils.helpers import create_directory, seed_everything


def setup_logging(log_file: Optional[str] = None) -> logging.Logger:
    """Set up logging configuration."""
    logger = logging.getLogger("debug_augmentation")
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if requested)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Debug augmentation issues for the Oxford-IIIT Pet Dataset"
    )
    
    parser.add_argument(
        "--processed-dir",
        type=str,
        default=str(project_root / "data" / "processed"),
        help="Path to processed dataset directory"
    )
    
    parser.add_argument(
        "--debug-dir",
        type=str,
        default=str(project_root / "debug_output"),
        help="Path to save debug output"
    )
    
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of samples to process for debugging"
    )
    
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Path to log file (optional)"
    )
    
    return parser.parse_args()


def inspect_mask(
    mask: np.ndarray, 
    name: str, 
    stage: str, 
    logger: logging.Logger
) -> None:
    """
    Inspect a mask and log detailed information about its values.
    
    Args:
        mask: Mask array to inspect
        name: Name identifier for the mask
        stage: Processing stage description (e.g., "original", "resized", "augmented")
        logger: Logger for output
    """
    unique_values = np.unique(mask)
    value_counts = {val: np.sum(mask == val) for val in unique_values}
    
    shape_info = f"shape={mask.shape}, dtype={mask.dtype}"
    values_info = f"unique values={unique_values.tolist()}"
    counts_info = ", ".join([f"{val}:{count} px" for val, count in value_counts.items()])
    
    logger.info(f"Mask [{name}] at {stage}: {shape_info}, {values_info}, {counts_info}")


def save_mask_visualization(
    mask: np.ndarray,
    output_path: Path,
    filename: str,
    stage: str,
    logger: logging.Logger
) -> None:
    """
    Save visualization of a mask for debugging.
    
    Args:
        mask: Mask as numpy array
        output_path: Output directory path
        filename: Base filename
        stage: Processing stage description
        logger: Logger for output
    """
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create a colorized version of the mask for visualization
    colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    
    # Background (class 0) - Black
    # Cat (class 1) - Red
    colored_mask[mask == 1] = [255, 0, 0]
    # Dog (class 2) - Green
    colored_mask[mask == 2] = [0, 255, 0]
    # Border/Don't care (class 255) - White
    colored_mask[mask == 255] = [255, 255, 255]
    
    # Save the colored mask
    cv2.imwrite(str(output_path / f"{filename}_{stage}_mask_colored.png"), colored_mask)
    
    # Save the raw mask (preserving exact values)
    Image.fromarray(mask).save(output_path / f"{filename}_{stage}_mask_raw.png")
    
    logger.info(f"Saved mask visualization for {filename} at stage '{stage}'")


def save_image_visualization(
    image: np.ndarray,
    output_path: Path,
    filename: str,
    stage: str,
    logger: logging.Logger
) -> None:
    """
    Save visualization of an image for debugging.
    
    Args:
        image: Image as numpy array
        output_path: Output directory path
        filename: Base filename
        stage: Processing stage description
        logger: Logger for output
    """
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Convert to BGR for OpenCV if image is RGB
    if len(image.shape) == 3 and image.shape[2] == 3:
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        image_bgr = image
    
    # Save the image
    cv2.imwrite(str(output_path / f"{filename}_{stage}_image.jpg"), image_bgr)
    
    logger.info(f"Saved image visualization for {filename} at stage '{stage}'")


def save_combined_visualization(
    image: np.ndarray,
    mask: np.ndarray,
    output_path: Path,
    filename: str,
    stage: str,
    logger: logging.Logger
) -> None:
    """
    Save a side-by-side visualization of image and mask.
    
    Args:
        image: Image as numpy array (RGB)
        mask: Mask as numpy array
        output_path: Output directory path
        filename: Base filename
        stage: Processing stage description
        logger: Logger for output
    """
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create a colorized version of the mask
    colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    colored_mask[mask == 1] = [255, 0, 0]      # Cat (class 1) - Red
    colored_mask[mask == 2] = [0, 255, 0]      # Dog (class 2) - Green
    colored_mask[mask == 255] = [255, 255, 255]  # Border (class 255) - White
    
    # Resize both to the same dimensions for side-by-side display
    h1, w1 = image.shape[:2]
    h2, w2 = colored_mask.shape[:2]
    
    # Use the larger dimensions to ensure nothing gets cropped
    display_h = max(h1, h2)
    display_w = max(w1, w2)
    
    # Resize image and mask to display dimensions
    if (h1, w1) != (display_h, display_w):
        display_image = cv2.resize(image, (display_w, display_h), interpolation=cv2.INTER_AREA)
    else:
        display_image = image.copy()
    
    if (h2, w2) != (display_h, display_w):
        display_mask = cv2.resize(colored_mask, (display_w, display_h), interpolation=cv2.INTER_NEAREST)
    else:
        display_mask = colored_mask.copy()
    
    # Create side-by-side visualization
    combined = np.zeros((display_h, display_w * 2, 3), dtype=np.uint8)
    combined[:, :display_w] = display_image
    combined[:, display_w:] = display_mask
    
    # Add text labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(combined, "Image", (10, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(combined, "Mask", (display_w + 10, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(combined, f"Stage: {stage}", (10, display_h - 10), font, 0.8, (255, 255, 255), 2)
    
    # Add color legend for the mask
    cv2.putText(combined, "Red: Cat", (display_w + 10, 70), font, 0.7, (0, 0, 255), 2)
    cv2.putText(combined, "Green: Dog", (display_w + 10, 100), font, 0.7, (0, 255, 0), 2)
    cv2.putText(combined, "White: Border", (display_w + 10, 130), font, 0.7, (255, 255, 255), 2)
    
    # Save the visualization
    cv2.imwrite(str(output_path / f"{filename}_{stage}_combined.jpg"), combined)
    
    logger.info(f"Saved combined visualization for {filename} at stage '{stage}'")


def resize_mask_to_match_image_debug(
    mask: np.ndarray, 
    target_shape: Tuple[int, int],
    name: str,
    output_dir: Path,
    logger: logging.Logger
) -> Dict[str, np.ndarray]:
    """
    Resize a mask to match the shape of the corresponding image
    with detailed logging and visualization for debugging.
    
    Args:
        mask: Original mask as numpy array
        target_shape: Target shape (height, width)
        name: Name identifier for debugging
        output_dir: Output directory for visualizations
        logger: Logger for output
        
    Returns:
        Dictionary of resized masks using different methods
    """
    logger.info(f"Resizing mask '{name}' from {mask.shape} to {target_shape}")
    
    # Inspect original mask
    inspect_mask(mask, name, "pre-resize", logger)
    
    # Save visualization of original mask
    save_mask_visualization(mask, output_dir, name, "original", logger)
    
    # Try different interpolation methods for comparison
    interpolation_methods = {
        'INTER_NEAREST': cv2.INTER_NEAREST,
        'INTER_LINEAR': cv2.INTER_LINEAR,
        'INTER_AREA': cv2.INTER_AREA
    }
    
    resized_masks = {}
    for method_name, method in interpolation_methods.items():
        try:
            # Resize mask using this interpolation method
            resized = cv2.resize(
                mask, (target_shape[1], target_shape[0]),
                interpolation=method
            )
            
            # Inspect resized mask
            inspect_mask(resized, name, f"post-resize-{method_name}", logger)
            
            # Save visualization
            save_mask_visualization(resized, output_dir, name, f"resized_{method_name}", logger)
            
            resized_masks[method_name] = resized
        except Exception as e:
            logger.error(f"Error resizing with {method_name}: {e}")
    
    # Check if PIL gives different results
    try:
        # Convert to PIL Image
        pil_mask = Image.fromarray(mask)
        
        # Resize with PIL using nearest neighbor
        pil_resized = pil_mask.resize((target_shape[1], target_shape[0]), Image.NEAREST)
        
        # Convert back to numpy
        pil_resized_np = np.array(pil_resized)
        
        # Inspect resized mask
        inspect_mask(pil_resized_np, name, "post-resize-PIL-NEAREST", logger)
        
        # Save visualization
        save_mask_visualization(pil_resized_np, output_dir, name, "resized_PIL_NEAREST", logger)
        
        resized_masks['PIL_NEAREST'] = pil_resized_np
    except Exception as e:
        logger.error(f"Error resizing with PIL: {e}")
    
    return resized_masks


def test_augmentation(
    image: np.ndarray,
    mask: np.ndarray,
    image_name: str,
    output_dir: Path,
    logger: logging.Logger
) -> None:
    """
    Test basic augmentations on an image-mask pair and visualize results.
    
    Args:
        image: Image as numpy array
        mask: Mask as numpy array
        image_name: Base name for output files
        output_dir: Output directory for visualizations
        logger: Logger for output
    """
    logger.info(f"Testing augmentations on {image_name}")
    
    # Create a simple augmentation pipeline
    augmentations = [
        ("HorizontalFlip", A.HorizontalFlip(p=1.0)),
        ("ShiftScaleRotate", A.ShiftScaleRotate(
            shift_limit=0.1, scale_limit=0.1, rotate_limit=15, 
            p=1.0, border_mode=cv2.BORDER_CONSTANT
        )),
        ("RandomBrightnessContrast", A.RandomBrightnessContrast(
            brightness_limit=0.2, contrast_limit=0.2, p=1.0
        )),
        ("HueSaturationValue", A.HueSaturationValue(
            hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=1.0
        ))
    ]
    
    # Test each augmentation
    for aug_name, transform in augmentations:
        try:
            logger.info(f"Applying {aug_name} to {image_name}")
            
            # Apply augmentation
            augmented = transform(image=image, mask=mask)
            augmented_image = augmented['image']
            augmented_mask = augmented['mask']
            
            # Inspect augmented mask
            inspect_mask(augmented_mask, image_name, f"after-{aug_name}", logger)
            
            # Save visualizations
            save_image_visualization(augmented_image, output_dir, image_name, f"aug_{aug_name}_img", logger)
            save_mask_visualization(augmented_mask, output_dir, image_name, f"aug_{aug_name}_mask", logger)
            save_combined_visualization(
                augmented_image, augmented_mask, 
                output_dir, image_name, f"aug_{aug_name}", logger
            )
            
        except Exception as e:
            logger.error(f"Error applying {aug_name} to {image_name}: {e}")


def debug_dataset_issues(args: argparse.Namespace, logger: logging.Logger) -> None:
    """
    Debug mask processing and augmentation issues.
    
    Args:
        args: Command-line arguments
        logger: Logger for output
    """
    # Set paths
    processed_dir = Path(args.processed_dir)
    debug_dir = Path(args.debug_dir)
    
    # Create debug output directory
    debug_dir.mkdir(parents=True, exist_ok=True)
    
    # Get paths for train images and masks
    train_dir = processed_dir / "Train"
    resized_img_dir = train_dir / "resized"
    orig_img_dir = train_dir / "color"
    mask_dir = train_dir / "label"
    
    # Get all training images
    train_images = list(resized_img_dir.glob("*.jpg"))
    logger.info(f"Found {len(train_images)} training images")
    
    # Create a dictionary mapping image stem to mask path
    mask_dict = {mask_path.stem: mask_path for mask_path in mask_dir.glob("*.png")}
    logger.info(f"Found {len(mask_dict)} training masks")
    
    # Sampling for debugging
    num_samples = min(args.num_samples, len(train_images))
    logger.info(f"Sampling {num_samples} images for debugging")
    
    # Get sample indices ensuring we include both cats and dogs
    cat_images = []
    dog_images = []
    
    for img_path in train_images:
        mask_path = mask_dict.get(img_path.stem)
        if mask_path:
            # Check if mask exists and load it
            try:
                mask = np.array(Image.open(mask_path))
                if 1 in mask:  # Cat
                    cat_images.append((img_path, mask_path))
                elif 2 in mask:  # Dog
                    dog_images.append((img_path, mask_path))
            except Exception:
                pass
    
    # Ensure we have both cats and dogs in our sample
    num_cats = min(num_samples // 2, len(cat_images))
    num_dogs = min(num_samples - num_cats, len(dog_images))
    
    logger.info(f"Sampling {num_cats} cat images and {num_dogs} dog images")
    
    # Randomly select samples
    np.random.seed(42)
    selected_cats = np.random.choice(cat_images, num_cats, replace=False).tolist()
    selected_dogs = np.random.choice(dog_images, num_dogs, replace=False).tolist()
    
    sample_pairs = selected_cats + selected_dogs
    
    # Process each sample
    for idx, (img_path, mask_path) in enumerate(sample_pairs):
        logger.info(f"Processing sample {idx+1}/{len(sample_pairs)}: {img_path.name}")
        
        # Create sample-specific output directory
        sample_dir = debug_dir / f"sample_{idx+1}_{img_path.stem}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Load image and mask
            logger.info(f"Loading image-mask pair: {img_path.name}, {mask_path.name}")
            
            # Read image with OpenCV
            image = cv2.imread(str(img_path))
            if image is None:
                raise ValueError(f"Failed to read image: {img_path}")
            
            # Convert to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Read mask with PIL
            mask = np.array(Image.open(mask_path))
            
            # Log info about loaded files
            logger.info(f"Image: shape={image.shape}, dtype={image.dtype}")
            inspect_mask(mask, img_path.stem, "loaded", logger)
            
            # Save visualizations of original files
            save_image_visualization(image, sample_dir, img_path.stem, "original", logger)
            save_mask_visualization(mask, sample_dir, img_path.stem, "original", logger)
            
            # Resize mask to match image dimensions
            if image.shape[:2] != mask.shape[:2]:
                logger.info(f"Mask shape {mask.shape} doesn't match image shape {image.shape[:2]}, resizing...")
                resized_masks = resize_mask_to_match_image_debug(
                    mask, image.shape[:2], img_path.stem, sample_dir, logger
                )
                
                # Test augmentations with each resizing method
                for method_name, resized_mask in resized_masks.items():
                    method_dir = sample_dir / method_name
                    method_dir.mkdir(parents=True, exist_ok=True)
                    
                    try:
                        # Try to create a side-by-side visualization
                        save_combined_visualization(
                            image, resized_mask, method_dir, 
                            img_path.stem, f"resized_{method_name}", logger
                        )
                        
                        # Test augmentations with this resized mask
                        test_augmentation(image, resized_mask, img_path.stem, method_dir, logger)
                    except Exception as e:
                        logger.error(f"Error with {method_name} mask for {img_path.name}: {e}")
            else:
                logger.info("Mask and image have the same dimensions, no resizing needed")
                # Create a side-by-side visualization
                save_combined_visualization(image, mask, sample_dir, img_path.stem, "original", logger)
                
                # Test augmentations on the original mask
                test_augmentation(image, mask, img_path.stem, sample_dir, logger)
            
        except Exception as e:
            logger.error(f"Error processing {img_path.name}: {e}")
    
    logger.info("Debugging complete. Check the debug output directory for visualizations.")


def main() -> None:
    """Main function for debugging dataset issues."""
    # Parse arguments
    args = parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_file)
    
    # Debug dataset issues
    debug_dataset_issues(args, logger)


if __name__ == "__main__":
    main()