#!/usr/bin/env python
"""
Script: preprocess_dataset.py

This script preprocesses the Oxford-IIIT Pet Dataset for semantic segmentation:
1. Detects and removes corrupt images
2. Splits the TrainVal dataset into separate Train and Validation sets
3. Standardizes image sizes with aspect ratio preservation
4. Creates properly structured directories in the processed folder

Example Usage:
    python scripts/preprocess_dataset.py --val-ratio 0.2 --size 512
"""

import argparse
import logging
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

# Add the project root to path to import our modules
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.utils.helpers import create_directory, seed_everything


def setup_logging(log_file: Optional[str] = None) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        log_file: Optional path to log file
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger("preprocess")
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
        description="Preprocess the Oxford-IIIT Pet Dataset for semantic segmentation"
    )
    
    parser.add_argument(
        "--raw-dir",
        type=str,
        default=str(project_root / "data" / "raw" / "Dataset_filtered"),
        help="Path to the raw dataset directory"
    )
    
    parser.add_argument(
        "--processed-dir",
        type=str,
        default=str(project_root / "data" / "processed"),
        help="Path to store preprocessed data"
    )
    
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Proportion of TrainVal data to use for validation"
    )
    
    parser.add_argument(
        "--size",
        type=int,
        default=512,
        help="Target size for both dimensions (with padding to preserve aspect ratio)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Path to log file (optional)"
    )
    
    return parser.parse_args()


def is_image_corrupt(image_path: Path) -> bool:
    """
    Check if an image is corrupt.
    
    Args:
        image_path: Path to the image
        
    Returns:
        True if image is corrupt, False otherwise
    """
    try:
        # Try reading with OpenCV
        img = cv2.imread(str(image_path))
        if img is None:
            return True
        
        # Try reading with PIL as well (catches different types of corruption)
        with Image.open(image_path) as img_pil:
            img_pil.verify()
        
        return False
    except Exception:
        return True


def find_corrupt_images(image_paths: List[Path], logger: logging.Logger) -> Set[str]:
    """
    Find corrupt images in a list of image paths.
    
    Args:
        image_paths: List of image paths to check
        logger: Logger for output
        
    Returns:
        Set of base names (without extension) of corrupt images
    """
    corrupt_images = set()
    
    for img_path in tqdm(image_paths, desc="Checking for corrupt images"):
        if is_image_corrupt(img_path):
            corrupt_images.add(img_path.stem)
            logger.warning(f"Corrupt image detected: {img_path}")
    
    return corrupt_images


def get_class_from_mask(mask_path: Path) -> int:
    """
    Determine if a mask contains a cat (1) or dog (2).
    
    Args:
        mask_path: Path to the mask file
        
    Returns:
        1 for cat, 2 for dog, 0 if neither could be determined
    """
    try:
        mask = np.array(Image.open(mask_path))
        if 1 in mask:
            return 1  # Cat
        elif 2 in mask:
            return 2  # Dog
        else:
            return 0  # Unknown
    except Exception:
        return 0  # Error case


def stratified_train_val_split(
    image_paths: List[Path],
    mask_paths: List[Path],
    val_ratio: float,
    corrupt_images: Set[str],
    seed: int
) -> Tuple[List[Tuple[Path, Path]], List[Tuple[Path, Path]]]:
    """
    Create a stratified train/val split based on class (cat/dog).
    
    Args:
        image_paths: List of image paths
        mask_paths: List of mask paths
        val_ratio: Ratio of validation set size to total
        corrupt_images: Set of corrupt image names to exclude
        seed: Random seed
        
    Returns:
        Tuple of (train_pairs, val_pairs) where each pair is (image_path, mask_path)
    """
    # Set random seed
    random.seed(seed)
    
    # Create a dictionary mapping image stem to mask path
    mask_dict = {mask_path.stem: mask_path for mask_path in mask_paths}
    
    # Group by class
    cat_images, dog_images = [], []
    
    for img_path in image_paths:
        # Skip corrupt images
        if img_path.stem in corrupt_images:
            continue
        
        # Find corresponding mask
        mask_path = mask_dict.get(img_path.stem)
        if not mask_path:
            continue
        
        # Determine class
        class_id = get_class_from_mask(mask_path)
        
        if class_id == 1:  # Cat
            cat_images.append((img_path, mask_path))
        elif class_id == 2:  # Dog
            dog_images.append((img_path, mask_path))
    
    # Shuffle both lists
    random.shuffle(cat_images)
    random.shuffle(dog_images)
    
    # Calculate split points
    cat_val_count = int(len(cat_images) * val_ratio)
    dog_val_count = int(len(dog_images) * val_ratio)
    
    # Split the data
    cat_val = cat_images[:cat_val_count]
    cat_train = cat_images[cat_val_count:]
    
    dog_val = dog_images[:dog_val_count]
    dog_train = dog_images[dog_val_count:]
    
    # Combine
    train_pairs = cat_train + dog_train
    val_pairs = cat_val + dog_val
    
    # Shuffle again to mix cats and dogs
    random.shuffle(train_pairs)
    random.shuffle(val_pairs)
    
    return train_pairs, val_pairs


def resize_with_padding(
    image: np.ndarray, 
    target_size: int
) -> np.ndarray:
    """
    Resize image preserving aspect ratio and pad to square.
    
    Args:
        image: Input image
        target_size: Target size for both dimensions after padding
        
    Returns:
        Resized and padded image
    """
    height, width = image.shape[:2]
    
    # Calculate target dimensions preserving aspect ratio
    if height > width:
        # Portrait orientation
        scale = target_size / height
        new_height = target_size
        new_width = int(width * scale)
    else:
        # Landscape or square orientation
        scale = target_size / width
        new_width = target_size
        new_height = int(height * scale)
    
    # Resize image
    resized = cv2.resize(image, (new_width, new_height))
    
    # Create a square canvas with black background
    if len(image.shape) == 3:  # Color image
        channels = image.shape[2]
        padded = np.zeros((target_size, target_size, channels), dtype=np.uint8)
    else:  # Grayscale image or mask
        padded = np.zeros((target_size, target_size), dtype=np.uint8)
    
    # Calculate padding offsets to center the image
    pad_y = (target_size - new_height) // 2
    pad_x = (target_size - new_width) // 2
    
    # Place the resized image on the padded canvas
    if len(image.shape) == 3:  # Color image
        padded[pad_y:pad_y+new_height, pad_x:pad_x+new_width, :] = resized
    else:  # Grayscale image or mask
        padded[pad_y:pad_y+new_height, pad_x:pad_x+new_width] = resized
    
    return padded


def process_image_mask_pair(
    img_path: Path,
    mask_path: Path,
    output_img_path: Path,
    output_mask_path: Path,
    target_size: int,
    logger: logging.Logger
) -> bool:
    """
    Process an image-mask pair: resize with padding and save.
    
    Args:
        img_path: Path to input image
        mask_path: Path to input mask
        output_img_path: Path to save processed image
        output_mask_path: Path to save processed mask
        target_size: Target size for both dimensions
        logger: Logger for output
        
    Returns:
        True if processing succeeded, False otherwise
    """
    try:
        # Create output directories
        output_img_path.parent.mkdir(parents=True, exist_ok=True)
        output_mask_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Read image with OpenCV (BGR format)
        image = cv2.imread(str(img_path))
        if image is None:
            logger.error(f"Failed to read image: {img_path}")
            return False
        
        # Read mask with PIL to preserve label values
        mask = np.array(Image.open(mask_path))
        
        # Resize and pad image
        processed_img = resize_with_padding(image, target_size)
        
        # Resize and pad mask with nearest neighbor interpolation to preserve label values
        # We need to process the mask separately to ensure label values are preserved
        processed_mask = resize_with_padding(mask, target_size)
        
        # Save image
        cv2.imwrite(str(output_img_path), processed_img)
        
        # Save mask (using PIL to preserve label values)
        Image.fromarray(processed_mask).save(output_mask_path)
        
        return True
        
    except Exception as e:
        logger.error(f"Error processing {img_path}: {e}")
        return False


def copy_test_set(
    raw_dir: Path,
    processed_dir: Path,
    target_size: int,
    logger: logging.Logger
) -> None:
    """
    Process and copy the test set.
    
    Args:
        raw_dir: Path to raw dataset directory
        processed_dir: Path to processed dataset directory
        target_size: Target size for images
        logger: Logger for output
    """
    # Define paths
    test_dir = raw_dir / "Test"
    test_img_dir = test_dir / "color"
    test_mask_dir = test_dir / "label"
    
    # Define output paths
    output_test_dir = processed_dir / "Test"
    output_test_img_dir = output_test_dir / "color"
    output_test_mask_dir = output_test_dir / "label"
    
    # Create output directories
    create_directory(output_test_img_dir)
    create_directory(output_test_mask_dir)
    
    # Get all test images
    test_images = list(test_img_dir.glob("*.jpg"))
    test_masks = list(test_mask_dir.glob("*.png"))
    
    # Create a dictionary mapping image stem to mask path
    mask_dict = {mask_path.stem: mask_path for mask_path in test_masks}
    
    # Process each image-mask pair
    logger.info(f"Processing {len(test_images)} test images...")
    
    for img_path in tqdm(test_images, desc="Processing test set"):
        # Find corresponding mask
        mask_path = mask_dict.get(img_path.stem)
        if not mask_path:
            logger.warning(f"No mask found for {img_path.name}, skipping")
            continue
        
        # Define output paths
        output_img_path = output_test_img_dir / img_path.name
        output_mask_path = output_test_mask_dir / mask_path.name
        
        # Process and save
        success = process_image_mask_pair(
            img_path, mask_path, 
            output_img_path, output_mask_path,
            target_size, logger
        )
        
        if not success:
            logger.warning(f"Failed to process test image: {img_path.name}")


def main() -> None:
    """Main function to preprocess the dataset."""
    # Parse arguments
    args = parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_file)
    
    # Set paths
    raw_dir = Path(args.raw_dir)
    processed_dir = Path(args.processed_dir)
    trainval_dir = raw_dir / "TrainVal"
    
    # Set seed for reproducibility
    seed_everything(args.seed)
    
    # Log basic info
    logger.info(f"Raw dataset directory: {raw_dir}")
    logger.info(f"Processed dataset directory: {processed_dir}")
    logger.info(f"Validation ratio: {args.val_ratio}")
    logger.info(f"Target size: {args.size}x{args.size}")
    
    # Check if dataset exists
    if not trainval_dir.exists():
        logger.error(f"TrainVal directory does not exist: {trainval_dir}")
        sys.exit(1)
    
    # Define paths
    trainval_img_dir = trainval_dir / "color"
    trainval_mask_dir = trainval_dir / "label"
    
    # Get all TrainVal images and masks
    trainval_images = list(trainval_img_dir.glob("*.jpg"))
    trainval_masks = list(trainval_mask_dir.glob("*.png"))
    
    logger.info(f"Found {len(trainval_images)} images and {len(trainval_masks)} masks in TrainVal")
    
    # Find corrupt images
    corrupt_images = find_corrupt_images(trainval_images, logger)
    logger.info(f"Found {len(corrupt_images)} corrupt images that will be excluded")
    
    # Create train/val split
    train_pairs, val_pairs = stratified_train_val_split(
        trainval_images, trainval_masks, args.val_ratio, corrupt_images, args.seed
    )
    
    logger.info(f"Split dataset into {len(train_pairs)} training and {len(val_pairs)} validation samples")
    
    # Create output directories
    train_img_dir = processed_dir / "Train" / "color"
    train_mask_dir = processed_dir / "Train" / "label"
    val_img_dir = processed_dir / "Val" / "color"
    val_mask_dir = processed_dir / "Val" / "label"
    
    create_directory(train_img_dir)
    create_directory(train_mask_dir)
    create_directory(val_img_dir)
    create_directory(val_mask_dir)
    
    # Process training set
    logger.info("Processing training set...")
    for img_path, mask_path in tqdm(train_pairs, desc="Processing training set"):
        output_img_path = train_img_dir / img_path.name
        output_mask_path = train_mask_dir / mask_path.name
        
        success = process_image_mask_pair(
            img_path, mask_path, 
            output_img_path, output_mask_path,
            args.size, logger
        )
        
        if not success:
            logger.warning(f"Failed to process training image: {img_path.name}")
    
    # Process validation set
    logger.info("Processing validation set...")
    for img_path, mask_path in tqdm(val_pairs, desc="Processing validation set"):
        output_img_path = val_img_dir / img_path.name
        output_mask_path = val_mask_dir / mask_path.name
        
        success = process_image_mask_pair(
            img_path, mask_path, 
            output_img_path, output_mask_path,
            args.size, logger
        )
        
        if not success:
            logger.warning(f"Failed to process validation image: {img_path.name}")
    
    # Process test set
    copy_test_set(raw_dir, processed_dir, args.size, logger)
    
    # Calculate class distribution
    train_cats = sum(1 for _, mask_path in train_pairs if get_class_from_mask(mask_path) == 1)
    train_dogs = sum(1 for _, mask_path in train_pairs if get_class_from_mask(mask_path) == 2)
    val_cats = sum(1 for _, mask_path in val_pairs if get_class_from_mask(mask_path) == 1)
    val_dogs = sum(1 for _, mask_path in val_pairs if get_class_from_mask(mask_path) == 2)
    
    logger.info("Final dataset statistics:")
    logger.info(f"Training set: {len(train_pairs)} images ({train_cats} cats, {train_dogs} dogs)")
    logger.info(f"Validation set: {len(val_pairs)} images ({val_cats} cats, {val_dogs} dogs)")
    logger.info("Preprocessing complete!")


if __name__ == "__main__":
    main()