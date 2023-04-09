import torch
import torchvision
from torch import optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import os
from dataset import ImageClassificationDataset
from model import VGG16,DnCNN
from utils import model_freeze
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

if __name__ == "__main__":
    train_data_path = "E:/archive/train"
    val_data_path = "E:/archive/valid"
    test_data_path = "E:/archive/test"
    checkpoints_dir = "E:/archive/checkpoint"

    train_dataset = ImageClassificationDataset(train_data_path, transform='train')
    val_dataset = ImageClassificationDataset(val_data_path, transform='val')
    test_dataset = ImageClassificationDataset(test_data_path, transform='test')

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True,num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    class_list = train_dataset.classes
    print(class_list)
    criterion = nn.CrossEntropyLoss()

    model = torchvision.models.vgg16(pretrained=True)
    model.classifier[6] = nn.Linear(in_features=4096, out_features=515)

    # Define the optimizer and learning rate scheduler
    optimizer = optim.AdamW(model.parameters(), lr=0.0001)
    scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=0.001)
    # Define the loss function
    criterion = nn.CrossEntropyLoss()
    # Define the device to use for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Transfer the model to the device
    model = model.to(device)
    # Define the number of epochs to train for
    num_epochs = 20
    train_acc_list = []
    val_acc_list = []
    test_acc_list = []

    best_val_acc = 0.0
    patience = 0
    early_stop_epochs = 3

    # Train the model
    for epoch in range(num_epochs):
        # Set the model to training mode
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        # Loop over the training dataset in batches
        for i, (images, labels) in enumerate(tqdm(train_dataloader)):
            # Transfer the data to the device
            images = images.to(device)
            labels = labels.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)

            # Compute the loss
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        train_loss = train_loss / len(train_dataloader.dataset)
        train_acc = 100.0 * correct / total
        train_acc_list.append(train_acc)

        # Set the model to evaluation mode
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        # Turn off gradients for validation
        with torch.no_grad():
            # Loop over the validation dataset in batches
            for images, labels in tqdm(val_dataloader):
                # Transfer the data to the device
                images = images.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = model(images)

                # Compute the loss
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # Update the learning rate scheduler
        scheduler.step()
        val_loss = val_loss / len(val_dataloader.dataset)
        val_acc = 100.0 * correct / total
        val_acc_list.append(val_acc)
        print('train_acc = ',train_acc,'val_acc = ',val_acc,'train_loss = ',train_loss,'val_loss = ',val_loss)
        # Save the model checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience = 0
            checkpoint_path = os.path.join(checkpoints_dir, f"BIRDS_515_SPECIES_epoch_{epoch + 1}_valacc_{val_acc}.pt")
            torch.save(model.state_dict(), checkpoint_path)
        else:
            patience += 1
            if patience >= early_stop_epochs:
                print(f"No improvement in validation accuracy for {early_stop_epochs} epochs. Stopping training early.")
                break
