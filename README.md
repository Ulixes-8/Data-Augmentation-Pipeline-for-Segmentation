
# Revised Augmentation Strategy

## 1. Augmentation Philosophy

With our correctly processed dataset and proper understanding of the evaluation requirements, we should focus on:

1. **Preserving Original Domain Information**: All augmentations should maintain the essential characteristics of pets while creating useful variations.

2. **Addressing Class Imbalance**: With a 2:1 dog-to-cat ratio, we need cat-focused augmentation to balance the training data.

3. **Semantic Integrity**: Augmentations must preserve the semantic meaning of mask labels (0: background, 1: cat, 2: dog, 255: border/don't care).

## 2. Augmentation Structure

### Directory Organization
```
data/processed/
├── Train/
│   ├── color/      (original images)
│   ├── label/      (original masks)
│   ├── resized/    (model-ready images)
│   └── augmented/  (NEW - will contain augmented data)
│       ├── images/ (augmented model-ready images)
│       └── masks/  (corresponding augmented masks)
├── Val/
│   ├── color/      (original images)
│   ├── label/      (original masks)
│   └── resized/    (model-ready images)
└── Test/
    ├── color/      (original images)
    ├── label/      (original masks)
    └── resized/    (model-ready images)
```

### Augmentation Storage Logic
- Original images/masks remain untouched in their respective directories
- We'll create new directories for augmented data
- Naming convention: `{original_filename}_aug{number}.{extension}`
- This approach maintains clear traceability between original and augmented data

## 3. Class-Specific Augmentation Strategy

### Cats (Minority Class)
- Generate 3 augmented versions of each cat image
- Apply more aggressive and varied augmentations
- Goal: Increase representation and variability of cat data

### Dogs (Majority Class)
- Generate 1 augmented version of each dog image
- Apply more conservative augmentations
- Goal: Add some variability without overrepresenting dogs

## 4. Detailed Augmentation Pipelines

### Cat Augmentation Pipeline
```python
cat_transform = A.Compose([
    # Spatial Transforms - More aggressive for cats
    A.HorizontalFlip(p=0.5),                                # Horizontal flip only (realistic for animals)
    A.ShiftScaleRotate(scale_limit=0.15,                   # Allow larger scale changes
                       rotate_limit=15,                     # Small rotations to maintain realism
                       shift_limit=0.1,                     # Allow slight shifting
                       p=0.8,                              # Higher probability for cats
                       border_mode=cv2.BORDER_CONSTANT,    # Black padding
                       value=0),
    
    # Moderate Elastic Transforms - only for cats
    A.OneOf([
        A.ElasticTransform(alpha=40, sigma=4, alpha_affine=15, p=0.6),  # Mild elastic deformation
        A.GridDistortion(num_steps=5, distort_limit=0.2, p=0.5),        # Slight grid distortion
        A.OpticalDistortion(distort_limit=0.2, shift_limit=0.15, p=0.5) # Mild optical distortion
    ], p=0.3),
    
    # Pixel-level Transforms - More variety for cats
    A.OneOf([
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.8),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=30, val_shift_limit=20, p=0.5),
        A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5)
    ], p=0.8),
    
    # Additional transformations for more variety
    A.OneOf([
        A.GaussNoise(var_limit=(10, 30), p=0.5),
        A.GaussianBlur(blur_limit=3, p=0.4),
        A.MotionBlur(blur_limit=3, p=0.4),
    ], p=0.4),
    
    # Lighting variations
    A.OneOf([
        A.RandomShadow(shadow_roi=(0, 0, 1, 1), p=0.3),
        A.RandomSunFlare(flare_roi=(0, 0, 1, 1), angle_lower=0, angle_upper=1, p=0.2),
    ], p=0.3),
], bbox_params=None)  # No bounding box params needed for segmentation
```

### Dog Augmentation Pipeline
```python
dog_transform = A.Compose([
    # Spatial Transforms - More conservative for dogs
    A.HorizontalFlip(p=0.5),                                # Horizontal flip only
    A.ShiftScaleRotate(scale_limit=0.1,                    # Smaller scale changes
                       rotate_limit=10,                     # More limited rotation
                       shift_limit=0.05,                    # Minimal shifting
                       p=0.5,                              # Lower probability
                       border_mode=cv2.BORDER_CONSTANT,
                       value=0),
    
    # No elastic transforms for dogs (to maintain more consistent appearance)
    
    # Pixel-level Transforms - Less variety for dogs
    A.OneOf([
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.7),
        A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=20, val_shift_limit=15, p=0.4),
    ], p=0.6),
    
    # Minimal noise/blur for dogs
    A.OneOf([
        A.GaussNoise(var_limit=(5, 20), p=0.3),
        A.GaussianBlur(blur_limit=3, p=0.3),
    ], p=0.3),
], bbox_params=None)
```

## 5. Implementation Considerations

### Mask Handling
- **Critical Point**: All spatial augmentations must be applied identically to both images and masks
- Use `mask_interpolation=cv2.INTER_NEAREST` for mask transformation to preserve exact class values
- Ensure the 255 (don't care/border) values are preserved exactly in masks

### Validation
- No augmentation for validation data
- Validation images are only resized/padded for model input
- Evaluation happens against original masks after inverse-transforming predictions

### Quality Assurance
- Visual inspection of augmented samples before full processing
- Check class distribution post-augmentation
- Verify masks maintain proper class values
- Ensure image-mask correspondence is maintained

## 6. Projected Class Balance

With our current dataset:
- Training: 948 cats, 1,991 dogs (2:1 ratio)
- Generate 3 augmentations per cat: 948 × 3 = 2,844 additional cat images
- Generate 1 augmentation per dog: 1,991 additional dog images

Final training set:
- Original: 948 cats + 1,991 dogs = 2,939 images
- Augmented: 2,844 cat augmentations + 1,991 dog augmentations = 4,835 images
- Total: 7,774 images with approximately 3,792 cats and 3,982 dogs (balanced)

## 7. Justification for Design Choices

1. **Aspect-Preserving Resize with Padding**: Maintains all visual information without distortion, critical for accurate segmentation.

2. **Horizontal Flips Only**: Vertical flips would create unrealistic animal orientations, potentially confusing the model.

3. **Limited Rotation Angles**: Large rotations could create unrealistic pet postures. Small rotations (±15° max) add variety while maintaining realism.

4. **Class-Specific Augmentation**: Addresses imbalance while ensuring each class gets appropriate transformations.

5. **Avoiding MixUp/CutMix**: These techniques blend class information, which could harm segmentation performance where clear boundaries are essential.

6. **Mild Elastic Deformations (Cats Only)**: Elastic transforms can help with generalization but must be mild to preserve recognizable features.

7. **Creating New Directories**: Keeps original data intact while clearly organizing augmented data.

This strategy balances the need for data augmentation with the preservation of semantic information critical for successful segmentation. The approach aligns with the professor's guidance to evaluate in the original domain while providing sufficient model-ready training data.