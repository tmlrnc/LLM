from albumentations import Compose, HorizontalFlip, Rotate, ColorJitter
from albumentations.pytorch import ToTensorV2

augmentations = Compose([
    HorizontalFlip(p=0.5),
    Rotate(limit=30, p=0.5),
    ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.5),
    ToTensorV2(),
])

def augment_image(image):
    augmented = augmentations(image=image)
    return augmented["image"]