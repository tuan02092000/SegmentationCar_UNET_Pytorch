import torch.cuda

import config
from lib import *
from model import UNET
from dataset import Dataset, make_data
from my_utils import *

device = config.DEVICE if torch.cuda.is_available() else "cpu"

def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device)
        targets = targets.float().unsqueeze(1).to(device)
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loop.set_postfix(loss=loss.item())

def main():
    train_transform = ImageTransform(config.RESIZE).data_transform["train"]
    val_transform = ImageTransform(config.RESIZE).data_transform["val"]

    model = UNET(in_channels=3, out_channels=1).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    train_loader, val_loader = get_loaders(config.TRAIN_DATASET_PATH,
                                           config.VAL_DATASET_PATH,
                                           batch_size=config.BATCH_SIZE,
                                           train_transform=train_transform,
                                           val_transform=val_transform,
                                           num_workers=config.NUM_WORKERS,
                                           pin_memory=config.PIN_MEMORY)
    if config.LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.rar"), model)
    check_accuracy(val_loader, model, device)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(config.EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        check_point = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }
        save_checkpoint(check_point)

        check_accuracy(val_loader, model, device=device)
        save_predictions_as_imgs(val_loader, model, folder="saved_images/", device=device)

if __name__ == '__main__':
    main()


