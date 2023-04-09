import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import os
from torchvision import transforms as T
# train_transforms = transforms.Compose([
# #     transforms.RandomResizedCrop(256),
# #     transforms.RandomHorizontalFlip(),
# #     transforms.ToTensor(),
# #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# # ])
# # test_transforms = transforms.Compose([
# #     transforms.Resize((256,256)),
# #     transforms.ToTensor(),
# #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# # ])
normalise_means = [0.4914, 0.4822, 0.4465]
normalise_stds = [0.2023, 0.1994, 0.2010]

train_transform = T.Compose([
    T.Resize((224,224)),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize(normalise_means, normalise_stds),])

test_transform = T.Compose([
    T.Resize((224,224)),
    T.ToTensor(),
    T.Normalize(normalise_means, normalise_stds)])

class ImageClassificationDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = os.listdir(root_dir)
        self.class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}
        self.images = []
        self.labels = []

        for c in self.classes:
            class_path = os.path.join(self.root_dir, c)
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                self.images.append(img_path)
                self.labels.append(self.class_to_idx[c])

    def __getitem__(self, index):
        img_path = self.images[index]
        label = self.labels[index]
        img = Image.open(img_path)
        if self.transform == 'train':
            img = train_transform(img)
        elif self.transform == 'val' or self.transform == 'test':
            img = test_transform(img)
        else:
            img = img
        return img, label

    def __len__(self):
        return len(self.labels)

# # Set the path to your data folder
# data_path = "E:/caltech256images/Kaggle Competition Train and test/Caltech 256_Train"
#
# # # Define the transforms to be applied to the images
# # transform = transforms.Compose([
# #     transforms.Resize((256, 256)),
# #     transforms.ToTensor(),
# # ])
# # #
# # # # Create the dataset
# # # dataset = ImageClassificationDataset(data_path, transform=transform)
# # #
# # # # Create a dataloader
# # # dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
# # #
# # # for images, labels in dataloader:
# # #     print(images.shape,labels)