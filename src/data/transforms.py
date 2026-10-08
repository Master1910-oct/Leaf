"""
AgriAI — M1.5: Preprocessing and Augmentation Pipeline
Implements the exact image pipeline specified in locked M0 research spec (§11.3, §11.4).

Policy:
- Training: Random horizontal flip (p=0.5), Random rotation (+/-30 deg, p=0.5),
            Color jitter (0.3, 0.3, 0.3, 0.1), Random resized crop (0.7, 1.0),
            ImageNet normalization.
            (Random vertical flip is deliberately excluded to preserve biological validity).
- Validation/Test: Deterministic resize, ImageNet normalization. NO random augmentation.
"""

from torchvision import transforms

# Standard ImageNet normalization parameters locked in M0
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_train_transforms(image_size: int = 128, include_vertical_flip: bool = False):
    """
    Returns the training transformation pipeline.
    
    Args:
        image_size: Target square image dimension (128 for dev, 224 for final).
        include_vertical_flip: Optional ablation flag (ABL-05). False by default.
    """
    transform_list = [
        transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=30),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)
    ]
    
    if include_vertical_flip:
        transform_list.append(transforms.RandomVerticalFlip(p=0.5))
        
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    
    return transforms.Compose(transform_list)

def get_eval_transforms(image_size: int = 128):
    """
    Returns deterministic transformation pipeline for validation and testing.
    Strictly NO random augmentations applied.
    
    Args:
        image_size: Target square image dimension (128 for dev, 224 for final).
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
