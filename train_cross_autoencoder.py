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
from model import VGG16,DnCNN,new_model,REDNet30,DenoisingAutoencoder
from utils import model_freeze,denormalize,show_test_batch,FGSM
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import warnings

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

if __name__ == "__main__":
    train_data_path = "./archive/train"
    val_data_path = "./archive/valid"
    test_data_path = "./archive/test"
    checkpoints_dir = "./archive/checkpoint"

    train_dataset = ImageClassificationDataset(train_data_path, transform='train')
    val_dataset = ImageClassificationDataset(val_data_path, transform='val')
    test_dataset = ImageClassificationDataset(test_data_path, transform='test')

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True,num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    class_list = train_dataset.classes
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vgg16 = torchvision.models.vgg16(pretrained=True)
    vgg16.classifier[6] = nn.Linear(in_features=4096, out_features=515)
    vgg16.load_state_dict(torch.load('E:/archive/checkpoint/BIRDS_515_SPECIES_epoch_5_valacc_93.70873786407768.pt'))


    #autoencoder = DnCNN()
    autoencoder = DenoisingAutoencoder()

    model = new_model(autoencoder,vgg16)

    model.load_state_dict(torch.load('./archive/checkpoint/BIRDS_515_Dncnn_vgg16_epoch_11_trainacc_54.385668004448526.pt'))

    model = model_freeze(model)
    model = model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=0.0005)
    scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=0.001)

    CrossEntropy = nn.CrossEntropyLoss()
    MSE = nn.MSELoss()

    COUNTER = 10

    noise_factor = 0.1
    n_epochs = 100
    batch_size= 32

    train_acc_list = []
    val_acc_list = []
    best_train_acc = 0.0
    best_val_acc = 0.0
    patience = 0
    early_stop_epochs = 100
    vis_count = 0
    for epoch in range(1, n_epochs + 1):
        # monitor training loss
        model.train()
        train_loss = 0.0
        train_loss2 = 0.0
        train_loss3 = 0.0
        correct = 0
        total = 0
        i_count = 0
        acc_total = 0
        ###################
        # train the model #
        ###################
        for data in tqdm(train_dataloader):
            # _ stands in for labels, here
            # no need to flatten images
            images, _ = data
            ## add random noise to the input images
            noisy_imgs = images + noise_factor * torch.randn(*images.shape)
            # Clip the images to be between 0 and 1
            noisy_imgs = np.clip(noisy_imgs, 0., 1.)
        #
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            ## forward pass: compute predicted outputs by passing *noisy* images to the model
            outputs, class_ = model(noisy_imgs.cuda())
            # calculate the loss
            # the "target" is still the original, not-noisy images

            loss = MSE(outputs, images.cuda())
            loss2 = CrossEntropy(class_, _.cuda())*0.1

            loss3 = loss + loss2
             # backward pass: compute gradient of the loss with respect to model parameters
            loss3.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update running training loss
            train_loss += loss.item() * images.size(0)
            train_loss2 += loss2.item() * batch_size
            train_loss3 += loss3.item() * batch_size
            pred = class_.argmax(dim=1)
            pred.to(device)

            total += _.size(0)
            correct += (pred == _.to(device)).sum().item()

            vis_count += 1
            if vis_count%500 == 0:
                show_test_batch(outputs[:16], images[:16], _[:16], pred[:16], class_list)

        # print avg training statistics
        train_loss = train_loss / len(train_dataloader)
        train_loss2 = train_loss2 / len(train_dataloader)
        train_loss3 = train_loss3 / len(train_dataloader)
        train_acc = 100.0 * correct / total
        train_acc_list.append(train_acc)

        model.eval()
        val_loss = 0.0
        val_loss1 = 0.0
        val_loss2 = 0.0
        val_loss3 = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for data in tqdm(val_dataloader):
                images, _ = data


                outputs, class_ = model(images.to(device))
                val_mse_loss = MSE(outputs, images.cuda())
                val_ce_loss = CrossEntropy(class_, _.cuda())
                val_total_loss = val_loss1 + val_loss2

                val_loss += val_mse_loss.item() * images.size(0)
                val_loss1 += val_ce_loss.item() * 16
                pred = class_.argmax(dim=1)
                pred.to(device)

                total += _.size(0)
                correct += (pred == _.to(device)).sum().item()

        show_test_batch(outputs, images, _, pred, class_list)

        scheduler.step()
        val_loss = val_loss / len(val_dataloader)
        val_loss1 = val_loss1 / len(val_dataloader)
        total_val_loss = val_loss + val_loss1
        val_acc = 100.0 * correct / total
        val_acc_list.append(val_acc)


        print('Epoch: {} \tTraining Loss MSE: {:.6f}'.format(
            epoch,
            val_loss
        ))
        print('Epoch: {} \tTraining Loss CE: {:.6f}'.format(
            epoch,
            val_loss1
        ))
        print('Epoch: {} \tTotal Loss: {:.6f}'.format(
            epoch,
            total_val_loss
        ))
        print('Epoch: {} \tAccuracy: {:.6f}'.format(
            epoch,
            train_acc
        ))



        if train_acc > best_train_acc:
            best_train_acc = train_acc
            patience = 0
            checkpoint_path = os.path.join(checkpoints_dir, f"BIRDS_515_Dncnn_vgg16_epoch_{epoch + 1}_trainacc_{train_acc}.pt")
            torch.save(model.state_dict(), checkpoint_path)
        else:
            patience += 1
            if patience >= early_stop_epochs:
                print(f"No improvement in validation accuracy for {early_stop_epochs} epochs. Stopping training early.")
                break
