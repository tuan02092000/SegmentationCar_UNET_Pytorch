from lib import *
import config
from dataset import CarvanaDataset, make_data
from transform import ImageTransform

def save_checkpoint(state, filename='my_checkpoint.path.tar'):
    print("[INFO] Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("[INFO] Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders(train_ds_path, val_ds_path, batch_size, train_transform, val_transform, num_workers=4, pin_memory=True):
    train_ds = CarvanaDataset(train_ds_path, train_transform)
    val_ds = CarvanaDataset(val_ds_path, val_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=False)

    return train_loader, val_loader

def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / (
                (preds + y).sum() + 1e-8
            )

    print(
        f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
    )
    print(f"Dice score: {dice_score/len(loader)}")
    model.train()

def save_predictions_as_imgs(
    loader, model, folder="saved_images/", device="cuda"
):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.png"
        )
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}.png")

    model.train()

if __name__ == '__main__':
    dataset = make_data(config.VAL_DATASET_PATH)
    train_loader, val_loader = get_loaders(config.TRAIN_DATASET_PATH,
                                           config.VAL_DATASET_PATH,
                                           batch_size=config.BATCH_SIZE,
                                           train_transform=ImageTransform(config.RESIZE).data_transform['train'],
                                           val_transform=ImageTransform(config.RESIZE).data_transform['val'],
                                           num_workers=config.NUM_WORKERS,
                                           pin_memory=config.PIN_MEMORY)
    # print(dataset['image'][1])
    # print(dataset['mask'][1])
    print("[INFO] Train loader: ", train_loader)
    print("[INFO] Val loader: ", val_loader)
