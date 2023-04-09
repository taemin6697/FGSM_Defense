import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import os
from dataset import ImageClassificationDataset
from model import VGG16,DnCNN
from utils import newmodel_freeze
import torch.nn as nn
import torch.nn.functional as F

if __name__ == "__main__":

    data_path = "E:/caltech256images/Kaggle Competition Train and test/Caltech 256_Train"
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    dataset = ImageClassificationDataset(data_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True,num_workers=4)
    class_list = dataset.classes
    criterion = nn.MSELoss()
    criterion2 = nn.CrossEntropyLoss()