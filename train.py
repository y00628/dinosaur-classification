from torch.utils.data import DataLoader
import torch
import torchvision.transforms as standard_transforms
import torchvision.models as models
from baseline import *
import numpy as np

mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
input_transform = standard_transforms.Compose(
    [standard_transforms.ToTensor(), standard_transforms.Normalize(*mean_std)]
)

# target_transform = MaskToTensor()