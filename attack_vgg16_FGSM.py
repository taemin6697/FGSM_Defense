import torch
import torchvision
from matplotlib import pyplot as plt
from torch import optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import os
from dataset import ImageClassificationDataset
from model import VGG16, DnCNN, DenoisingUNet
from utils import model_freeze, only_class_FGSM,FGSM
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from model import new_model


if __name__ == "__main__":
    train_data_path = "./archive/train"
    val_data_path = "./archive/valid"
    test_data_path = "./archive/kk/test"
    checkpoints_dir = "./archive/checkpoint"

    train_dataset = ImageClassificationDataset(train_data_path, transform='train')
    val_dataset = ImageClassificationDataset(val_data_path, transform='val')
    test_dataset = ImageClassificationDataset(test_data_path, transform='test')

    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=False,num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)

    class_list = train_dataset.classes
    print(class_list)

    model = torchvision.models.vgg16(pretrained=True)
    model.classifier[6] = nn.Linear(in_features=4096, out_features=515)
    model.load_state_dict(torch.load('E:/archive/checkpoint/BIRDS_515_SPECIES_epoch_5_valacc_93.70873786407768.pt'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #model.load_state_dict(torch.load('./archive/checkpoint/BIRDS_515_Dncnn_vgg16_epoch_8_trainacc_77.70175523427301.pt'))
    model = model.to(device)

    ep = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
    accuracy = []
    for i in tqdm(range(len(ep))):
        p = only_class_FGSM(test_dataloader,model=model,epsilon=ep[i])
        accuracy.append(p)
    print(accuracy)

    plt.plot(ep,accuracy , marker='.', c='red', label="only_VGG16")
    plt.legend(loc='upper right')
    plt.grid()
    plt.xlabel('ep')
    plt.ylabel('accuracy')
    plt.show()