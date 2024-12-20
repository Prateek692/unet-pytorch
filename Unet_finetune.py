import os
import numpy as np
from PIL import Image
import cv2
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import time

def pad_to_divisible(image, divisor=32):
    height, width = image.shape[1:3]
    new_height = ((height + divisor - 1) // divisor) * divisor
    new_width = ((width + divisor - 1) // divisor) * divisor
    pad_height = new_height - height
    pad_width = new_width - width

    padded_image = torch.nn.functional.pad(image, (0, pad_width, 0, pad_height), mode='constant', value=0)

    return padded_image

class CustomSegmentationDataset(Dataset):

    def __init__(self, image_dir, mask_dir, transform_2):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        # self.transform_1 = transform_1
        self.transform_2 = transform_2
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index])

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        # print(np.shape(image))
        # print(np.shape(mask))

        # if self.transform:
        # print(image_path)
        # print(mask_path)
        # print(np.unique(mask))
        # image = self.transform_1(image)
        # mask = self.transform_1(mask)
        # print(np.unique(mask))
        image = cv2.resize(image, (512,512), interpolation=cv2.INTER_NEAREST)
        mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST_EXACT)
        
        
        image = self.transform_2(image)
        mask = self.transform_2(mask)

        # image = pad_to_divisible(image)
        # mask = pad_to_divisible(mask)
        # print(mask.type())
        # print(torch.unique(mask))
        mask = (mask * 1000).round().long()
        mask //= 4
        # mask = np.array(mask)
        # print(np.unique(mask))
        # mask = torch.from_numpy(mask).long()
        # print(np.shape(image))
        # print(np.shape(mask))
        # image = image.squeeze(0)
        mask = mask.squeeze(0)
        # print(f"{mask_path},: {torch.unique(mask)}")
        # print(torch.unique(mask))
        # print(np.shape(mask))
        return image, mask

from torchvision import transforms

# transform_1 = transforms.Compose([
#     transforms.ToPILImage(),
#     # transforms.Resize((512, 512)),
# ])

transform_2 = transforms.Compose([
    transforms.ToTensor()
])

import segmentation_models_pytorch as smp

image_dir = "training_data/train_1/images"
mask_dir = "training_data/train_1/labels"
num_classes = 4
batch_size = 10
model = smp.Unet("resnet50", encoder_weights = "imagenet", encoder_depth = 5, classes = num_classes, activation = "softmax", in_channels = 1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model.to(device=device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

from torch.utils.data import DataLoader

train_dataset = CustomSegmentationDataset(image_dir=image_dir, mask_dir=mask_dir, transform_2=transform_2)
# print(type(train_dataset))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

num_epochs = 10

for epoch in range(num_epochs):
    start = time.time()
    model.train()
    running_loss = 0.0

    total_batches = len(train_loader)

    for batch_index, (images, masks) in enumerate(train_loader):
        images = images.to(device)
        masks = masks.to(device)
        optimizer.zero_grad()

        print(torch.unique(masks))
        # if (1 in torch.unique(masks)) and (2 in torch.unique(masks)):
        #     masks[masks==2] = 0

        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if (batch_index + 1) % 2 == 0 or (batch_index + 1) == 0:
            avg_loss = running_loss / (batch_index + 1)

            print(f"Epoch [{epoch + 1} / {num_epochs}], Batch [{batch_index + 1} / {total_batches}], Loss: {avg_loss:.4f}")

    avg_loss = running_loss / total_batches

    end = time.time()
    # time_taken = end-start
    print(f"Time taken for epoch {epoch + 1}: {((end-start)/60):.3f} minutes, Loss: {avg_loss:.4f}")

torch.save(model.state_dict(), "unet_finetune_test.pth")
print("Model saved.")