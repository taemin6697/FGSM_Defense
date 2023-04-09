import os
import shutil
import numpy as np
import splitfolders
import torch
from matplotlib import pyplot as plt
from torch.autograd import Variable
import torch.nn as nn
import warnings

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
normalise_means = [0.4914, 0.4822, 0.4465]
normalise_stds = [0.2023, 0.1994, 0.2010]

def denormalize(img):
    means = torch.tensor(normalise_means).view(3, 1, 1)
    stds = torch.tensor(normalise_stds).view(3, 1, 1)
    return img * stds + means

def show_test_only_batch(images,ori,class_map):
    images = denormalize(images.detach().cpu())
    ori = denormalize(ori.detach().cpu())
    images = images.detach().cpu()
    ori = ori.detach().cpu()
    images = images.numpy()

    fig, ax = plt.subplots(4, 4, figsize=(15, 15))
    fig.tight_layout()
    for i in range(2):
        for j in range(4):
            ax[i+2, j].imshow(np.transpose(images[i*4+j], (1, 2, 0)))
            ax[i+2, j].axis("off")

            ax[i, j].imshow(np.transpose(ori[i * 4 + j], (1, 2, 0)))
            ax[i, j].axis("off")

    plt.show()


def show_test_batch(images,ori,preds, targets, class_map):
    images = denormalize(images.detach().cpu())
    ori = denormalize(ori.detach().cpu())
    images = images.detach().cpu()
    ori = ori.detach().cpu()
    images = images.numpy()
    preds = preds.detach().cpu().numpy()
    targets = targets.cpu().numpy()

    fig, ax = plt.subplots(4, 4, figsize=(15, 15))
    fig.tight_layout()
    for i in range(2):
        for j in range(4):
            ax[i+2, j].imshow(np.transpose(images[i*4+j], (1, 2, 0)))
            ax[i+2, j].set_title(f"Actual: {class_map[targets[i*4+j]]}\nPred: {class_map[preds[i*4+j]]}")
            ax[i+2, j].axis("off")

            ax[i, j].imshow(np.transpose(ori[i * 4 + j], (1, 2, 0)))
            ax[i, j].axis("off")

    plt.show()
def model_freeze(model):
    for name, param in model.named_parameters():
        print(name, param.requires_grad)
    for name, child in model.named_children():
        if name == 'model2':
            for param in child.parameters():
                param.requires_grad = False
    return model

def train_val_split():

    # Set the path to the downloaded dataset directory
    dataset_path = "E:/caltech256images/Kaggle Competition Train and test/Caltech 256_Train"

    # Set the ratio of train to validation split
    train_ratio = 0.8

    # Create directories for train and validation sets
    train_dir = os.path.join(dataset_path, "train")
    os.makedirs(train_dir, exist_ok=True)
    val_dir = os.path.join(dataset_path, "val")
    os.makedirs(val_dir, exist_ok=True)

    # Loop through each class folder in the dataset directory
    for class_folder in os.listdir(dataset_path):
        if os.path.isdir(os.path.join(dataset_path, class_folder)):
            # Create subdirectories for train and validation sets within each class folder
            train_class_dir = os.path.join(train_dir, class_folder)
            os.makedirs(train_class_dir, exist_ok=True)
            val_class_dir = os.path.join(val_dir, class_folder)
            os.makedirs(val_class_dir, exist_ok=True)

            # Loop through each image file in the class folder and divide them randomly into train and validation sets
            class_images = os.listdir(os.path.join(dataset_path, class_folder))
            num_train = int(train_ratio * len(class_images))
            train_images = set(np.random.choice(class_images, size=num_train, replace=False))
            val_images = set(class_images) - train_images

            # Copy train images to the train subdirectory for the current class
            for train_image in train_images:
                train_image_path = os.path.join(dataset_path, class_folder, train_image)
                train_image_dest = os.path.join(train_class_dir, train_image)
                shutil.copy(train_image_path, train_image_dest)

            # Copy validation images to the validation subdirectory for the current class
            for val_image in val_images:
                val_image_path = os.path.join(dataset_path, class_folder, val_image)
                val_image_dest = os.path.join(val_class_dir, val_image)
                shutil.copy(val_image_path, val_image_dest)


def only_class_FGSM(test_loader,model,epsilon=0.1, min_val=-1, max_val=1):
    correct = 0  # Fast gradient sign method
    adv_correct = 0
    misclassified = 0
    total = 0
    adv_noise = 0
    adverserial_images = []
    y_preds = []
    y_preds_adv = []
    test_images = []
    test_label = []
    criterion = nn.CrossEntropyLoss()
    for i, (images, labels) in enumerate(test_loader):
        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()
        images = Variable(images, requires_grad=True)
        labels = Variable(labels)

        outputs = model(images)

        loss = criterion(outputs, labels)
        model.zero_grad()
        if images.grad is not None:
            images.grad.data.fill_(0)
        loss.backward()

        grad = torch.sign(images.grad.data)  # Take the sign of the gradient.
        images_adv = torch.clamp(images.data + epsilon * grad, min_val, max_val)  # x_adv = x + epsilon*grad

        adv_output = model(Variable(images_adv))  # output by the model after adding adverserial noise

        _, predicted = torch.max(outputs.data, 1)  # Prediction on the clean image
        _, adv_predicted = torch.max(adv_output.data, 1)  # Prediction on the image after adding adverserial noise

        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        adv_correct += (adv_predicted == labels).sum().item()
        misclassified += (predicted != adv_predicted).sum().item()

        adverserial_images.extend((images_adv).cpu().data.numpy())
        y_preds.extend(predicted.cpu().data.numpy())
        y_preds_adv.extend(adv_predicted.cpu().data.numpy())
        test_images.extend(images.cpu().data.numpy())
        test_label.extend(labels.cpu().data.numpy())

    np.save('adverserial_images.npy', adverserial_images)  # Save the adverserial labels, images
    np.save('y_preds.npy', y_preds)
    np.save('y_preds_adv.npy', y_preds_adv)
    np.save('test_images.npy', test_images)
    np.save('test_label.npy', test_label)
    print('Accuracy of the model w/0 adverserial attack on test images is : {} %'.format(100 * correct / total))
    print('Accuracy of the model with adverserial attack on test images is : {} %'.format(100 * adv_correct / total))
    print('Number of misclassified examples(as compared to clean predictions): {}/{}'.format(misclassified, total))

    return (100 * adv_correct / total)


def FGSM(test_loader,model,epsilon=0.1, min_val=-1, max_val=1):
    correct = 0  # Fast gradient sign method
    adv_correct = 0
    misclassified = 0
    total = 0
    adv_noise = 0
    adverserial_images = []
    y_preds = []
    y_preds_adv = []
    test_images = []
    test_label = []
    criterion = nn.CrossEntropyLoss()
    for i, (images, labels) in enumerate(test_loader):
        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()
        images = Variable(images, requires_grad=True)
        labels = Variable(labels)

        outputs, temp = model(images)

        loss = criterion(outputs, labels)
        model.zero_grad()
        if images.grad is not None:
            images.grad.data.fill_(0)
        loss.backward()

        grad = torch.sign(images.grad.data)  # Take the sign of the gradient.
        images_adv = torch.clamp(images.data + epsilon * grad, min_val, max_val)  # x_adv = x + epsilon*grad

        adv_output, temp = model(Variable(images_adv))  # output by the model after adding adverserial noise

        _, predicted = torch.max(outputs.data, 1)  # Prediction on the clean image
        _, adv_predicted = torch.max(adv_output.data, 1)  # Prediction on the image after adding adverserial noise

        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        adv_correct += (adv_predicted == labels).sum().item()
        misclassified += (predicted != adv_predicted).sum().item()

        adverserial_images.extend((images_adv).cpu().data.numpy())
        y_preds.extend(predicted.cpu().data.numpy())
        y_preds_adv.extend(adv_predicted.cpu().data.numpy())
        test_images.extend(images.cpu().data.numpy())
        test_label.extend(labels.cpu().data.numpy())

    np.save('adverserial_images.npy', adverserial_images)  # Save the adverserial labels, images
    np.save('y_preds.npy', y_preds)
    np.save('y_preds_adv.npy', y_preds_adv)
    np.save('test_images.npy', test_images)
    np.save('test_label.npy', test_label)
    print('Accuracy of the model w/0 adverserial attack on test images is : {} %'.format(100 * correct / total))
    print('Accuracy of the model with adverserial attack on test images is : {} %'.format(100 * adv_correct / total))
    print('Number of misclassified examples(as compared to clean predictions): {}/{}'.format(misclassified, total))

    return (100 * adv_correct / total)
#splitfolders.ratio("E:/caltech256images/Kaggle Competition Train and test/Caltech 256_Train", output="E:/caltech256images/Kaggle Competition Train and test/output",seed=1337, ratio=(.8, .1, .1), group_prefix=None, move=False)
