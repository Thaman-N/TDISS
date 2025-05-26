import torch
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
import random
import numpy as np

class RandAugment:
    """
    RandAugment implementation as mentioned in the CUE-Net paper.
    Based on 'RandAugment: Practical automated data augmentation with a reduced search space'
    """
    def __init__(self, n=2, m=9):
        self.n = n  # Number of augmentations to apply
        self.m = m  # Magnitude of augmentations
        
        # Define available augmentations
        self.augmentations = [
            self.auto_contrast,
            self.equalize,
            self.rotate,
            self.solarize,
            self.color,
            self.contrast,
            self.brightness,
            self.sharpness,
            self.shear_x,
            self.shear_y,
            self.translate_x,
            self.translate_y,
            self.posterize,
        ]
    
    def __call__(self, img):
        """Apply n random augmentations to the image"""
        augmentations = random.choices(self.augmentations, k=self.n)
        for aug in augmentations:
            magnitude = random.uniform(0, self.m)
            img = aug(img, magnitude)
        return img
    
    def auto_contrast(self, img, magnitude):
        return F.autocontrast(img)
    
    def equalize(self, img, magnitude):
        return F.equalize(img)
    
    def rotate(self, img, magnitude):
        degrees = magnitude * 3.0  # Scale to reasonable rotation range
        return F.rotate(img, degrees)
    
    def solarize(self, img, magnitude):
        threshold = 256 - magnitude * 25.6  # Scale to 0-256
        return F.solarize(img, threshold)
    
    def color(self, img, magnitude):
        factor = 1.0 + magnitude * 0.9 / 10.0  # Scale to reasonable factor
        return F.adjust_saturation(img, factor)
    
    def contrast(self, img, magnitude):
        factor = 1.0 + magnitude * 0.9 / 10.0  # Scale to reasonable factor
        return F.adjust_contrast(img, factor)
    
    def brightness(self, img, magnitude):
        factor = 1.0 + magnitude * 0.9 / 10.0  # Scale to reasonable factor
        return F.adjust_brightness(img, factor)
    
    def sharpness(self, img, magnitude):
        factor = 1.0 + magnitude * 0.9 / 10.0  # Scale to reasonable factor
        return F.adjust_sharpness(img, factor)
    
    def shear_x(self, img, magnitude):
        factor = magnitude * 0.3 / 10.0  # Scale to reasonable shear range
        return F.affine(img, angle=0, translate=[0, 0], scale=1.0, shear=[factor, 0])
    
    def shear_y(self, img, magnitude):
        factor = magnitude * 0.3 / 10.0  # Scale to reasonable shear range
        return F.affine(img, angle=0, translate=[0, 0], scale=1.0, shear=[0, factor])
    
    def translate_x(self, img, magnitude):
        factor = magnitude * 0.3 / 10.0  # Scale to reasonable translation
        pixels = factor * img.size[0]
        return F.affine(img, angle=0, translate=[pixels, 0], scale=1.0, shear=[0, 0])
    
    def translate_y(self, img, magnitude):
        factor = magnitude * 0.3 / 10.0  # Scale to reasonable translation
        pixels = factor * img.size[1]
        return F.affine(img, angle=0, translate=[0, pixels], scale=1.0, shear=[0, 0])
    
    def posterize(self, img, magnitude):
        bits = 8 - int(magnitude * 0.8)  # Scale to reasonable bit range (8 to 1)
        bits = max(1, bits)
        return F.posterize(img, bits)

def get_training_transforms(spatial_size=336):
    """Create training transforms with RandAugment as specified in the paper"""
    return transforms.Compose([
        # No ToPILImage here - we've already converted to PIL in the dataset class
        transforms.Resize((spatial_size, spatial_size)),
        RandAugment(n=2, m=9),  # Apply 2 random augmentations with magnitude 9
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])

def get_validation_transforms(spatial_size=336):
    """Create validation transforms"""
    return transforms.Compose([
        # No ToPILImage here - we've already converted to PIL in the dataset class
        transforms.Resize((spatial_size, spatial_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])