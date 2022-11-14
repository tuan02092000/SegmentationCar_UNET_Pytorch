'''
DATASET:
    train:
        images
        masks
    val:
        images
        masks
'''

from lib import *
import config
from transform import ImageTransform

def make_data(dataset_path):
    data_dict = {'image': [], 'mask': []}
    folder_path = os.path.join(dataset_path, 'images')
    for image in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image)
        mask_path = os.path.join(folder_path.replace('images', 'masks'), image.replace('.jpg', '_mask.gif'))
        data_dict['image'].append(image_path)
        data_dict['mask'].append(mask_path)
    return data_dict

class CarvanaDataset(Dataset):
    def __init__(self, dataset_path, transform=None):
        self.dataset = make_data(dataset_path)
        self.image = self.dataset['image']
        self.mask = self.dataset['mask']
        self.transform = transform
    def __len__(self):
        return len(self.image)
    def __getitem__(self, idx):
        image_path = self.image[idx]
        mask_path = self.mask[idx]
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        image = np.array(image)
        mask = np.array(mask, dtype=np.float32)
        mask[mask == 255.0] = 1.0
        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]
        return image, mask

if __name__ == '__main__':
    train_dataset = CarvanaDataset(config.TRAIN_DATASET_PATH, transform=ImageTransform(config.RESIZE).data_transform['train'])
    val_dataset = CarvanaDataset(config.VAL_DATASET_PATH, transform=ImageTransform(config.RESIZE).data_transform['val'])
    print("[INFO] Train 0: ", train_dataset[0])
