from lib import *

class ImageTransform:
    def __init__(self, resize):
        self.data_transform = {
            'train': A.Compose([
                A.Resize(height=resize[0], width=resize[1]),
                A.Rotate(limit=35, p=1.0),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.1),
                A.Normalize(
                    mean=[0.0, 0.0, 0.0],
                    std=[1.0, 1.0, 1.0],
                    max_pixel_value=255.0,
                ),
                ToTensorV2()
            ]),
            'val': A.Compose([
                A.Resize(height=resize[0], width=resize[1]),
                A.Normalize(
                    mean=[0.0, 0.0, 0.0],
                    std=[1.0, 1.0, 1.0],
                    max_pixel_value=255.0,
                ),
                ToTensorV2(),
            ])
        }
