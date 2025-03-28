# Oxford-IIIT Pet Dataset: Augmentation for Semantic Segmentation

This repository contains scripts for preprocessing and augmenting the Oxford-IIIT Pet Dataset for semantic segmentation tasks. The implementation focuses on class-balanced data augmentation to address the dog-to-cat imbalance in the original dataset.

## Repository Structure

```
.
├── config/
│   └── augmentation_config.yaml    # Configuration for augmentation parameters
├── data/
│   └── processed/                  # Processed dataset (details below)
├── src/
│   ├── augment_dataset.py          # Class-specific augmentation implementation
│   ├── dataset_analyzer.py         # Dataset analysis utilities
│   ├── debug_mask_values.py        # Tool for analyzing mask encoding
│   ├── download_and_extract.py     # Download dataset from Google Drive
│   ├── preprocess_dataset.py       # Split dataset and resize images
│   ├── preprocess_test_val_labels.py # Process val/test masks without resizing
│   └── preprocess_training_labels.py # Resize training masks to match images
└── utils/
    └── helpers.py                  # Utility functions
```

## Data Directory Structure

```
data/
└── processed/
    ├── Train/
    │   ├── augmented/
    │   │   ├── images/         # Augmented .jpg images (resized)
    │   │   └── masks/          # Corresponding augmented .png masks (resized)
    │   ├── color/              # Original training .jpg images
    │   ├── label/              # Original training .png masks
    │   ├── resized/            # Resized .jpg images (512x512)
    │   └── resized_label/      # Resized .png masks (512x512)
    │
    ├── Val/
    │   ├── color/              # Original validation .jpg images
    │   ├── label/              # Original validation .png masks
    │   ├── resized/            # Resized .jpg validation images
    │   └── processed_labels/   # Processed .png masks
    │
    └── Test/
        ├── color/              # Original test .jpg images
        ├── label/              # Original test .png masks
        ├── resized/            # Resized .jpg test images
        └── processed_labels/   # Processed .png test masks
```

## Augmentation Philosophy

The augmentation strategy is designed to:

1. **Address Class Imbalance**: The original dataset has a 2:1 dog-to-cat ratio. Our approach generates more augmentations for cats to balance the classes.
2. **Preserve Semantic Integrity**: All augmentations maintain the semantic meaning of mask labels (0: background, 1: cat, 2: dog, 255: border/don't care).
3. **Maintain Domain Information**: Augmentations preserve essential characteristics of pets while creating useful variations.

## Preprocessing Pipeline

The preprocessing consists of multiple stages:

1. **Dataset Download and Extraction** (`download_and_extract.py`)
   - Downloads the Oxford-IIIT Pet Dataset from Google Drive
   - Extracts into the `data/raw/` directory

2. **Dataset Preprocessing** (`preprocess_dataset.py`)
   - Detects and removes corrupt images
   - Splits the TrainVal data into separate Train and Validation sets
   - Standardizes image sizes (512×512 by default) with aspect ratio preservation
   - Preserves original mask files for accurate evaluation

3. **Mask Processing** (`preprocess_training_labels.py` and `preprocess_test_val_labels.py`)
   - Processes training masks to match resized images
   - Handles test and validation masks with proper class values

4. **Data Augmentation** (`augment_dataset.py`)
   - Applies class-specific augmentation strategies
   - More aggressive augmentation for cats (minority class)
   - Conservative augmentation for dogs (majority class)
   - Generates multiple augmented versions of each image (more for cats, fewer for dogs)

## Augmentation Details

### Cat Augmentation (Minority Class)
- Generates 5 augmented versions of each cat image by default
- Applies more aggressive and varied augmentations
- Includes elastic transforms, perspective changes, and diverse color transformations

### Dog Augmentation (Majority Class)
- Generates 2 augmented versions of each dog image by default
- Applies more conservative augmentations
- Uses less extreme parameter values for transformations

## Class Balancing Results

The augmentation strategy transforms the dataset from:
- Original: ~948 cats, ~1,991 dogs (2:1 ratio)
- After augmentation: ~5,688 cats, ~5,973 dogs (balanced)

## Usage

### Prerequisites
- Python 3.8+
- Dependencies listed in `requirements.txt`

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd <repository-dir>

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Data Preparation
```bash
# Download and extract the dataset
python src/download_and_extract.py

# Preprocess the dataset
python src/preprocess_dataset.py --val-ratio 0.2 --size 512

# Process training masks
python src/preprocess_training_labels.py --size 512

# Process validation and test masks
python src/preprocess_test_val_labels.py
```

### Data Augmentation
```bash
# Run augmentation with default settings (5 augmentations per cat, 2 per dog)
python src/augment_dataset.py --config config/augmentation_config.yaml

# Or with custom settings
python src/augment_dataset.py --cat-augmentations 3 --dog-augmentations 1
```

### Analysis and Debugging
```bash
# Analyze the dataset statistics
python src/dataset_analyzer.py data/processed

# Debug mask values
python src/debug_mask_values.py
```

## Mask Value Encoding

The masks use the following value encoding:
- **0**: Background
- **1**: Cat
- **2**: Dog
- **255**: Border/Don't care (used for pixels that should be ignored during training)

## Configuration

The `config/augmentation_config.yaml` file contains detailed parameters for all augmentation operations, including:
- Spatial transforms (flip, scale, rotate, shift)
- Elastic transforms (for cats only)
- Pixel-level transforms (brightness, contrast, hue, saturation)
- Noise and blur effects
- Lighting variations

## Notes on Implementation

- Augmentation uses the Albumentations library for fast, reliable image transforms
- Masks are processed with `cv2.INTER_NEAREST` interpolation to preserve exact class values
- The augmentation process creates new directories for augmented data while preserving originals
- All augmentations maintain image-mask correspondence

## License

This project is distributed under the MIT License. See the `LICENSE` file for more information.

## Acknowledgments

The Oxford-IIIT Pet Dataset was created by Parkhi et al:
- O. M. Parkhi, A. Vedaldi, A. Zisserman, C. V. Jawahar, "Cats and Dogs," IEEE Conference on Computer Vision and Pattern Recognition, 2012.