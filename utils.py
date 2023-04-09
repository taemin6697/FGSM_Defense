import os
import shutil
import numpy as np
import splitfolders


def newmodel_freeze(model):
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
#splitfolders.ratio("E:/caltech256images/Kaggle Competition Train and test/Caltech 256_Train", output="E:/caltech256images/Kaggle Competition Train and test/output",seed=1337, ratio=(.8, .1, .1), group_prefix=None, move=False)
