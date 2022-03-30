from torchvision import transforms as T
from config import *

test_augmentation = T.Compose([
    T.Resize([IMAGE_SIZE, IMAGE_SIZE]),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_augmentation = T.Compose([
    T.RandomRotation((0, 90)),
    T.ColorJitter(.3, .3, .3, .3),
    T.RandomVerticalFlip(p=.33),
    T.RandomHorizontalFlip(p=.33),
    T.RandomResizedCrop(IMAGE_SIZE),
    T.Resize([IMAGE_SIZE, IMAGE_SIZE]),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
