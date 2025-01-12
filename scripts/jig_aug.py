
import glob
import sys
import os
from tqdm.auto import tqdm
import wandb
import joblib
import time
from PIL import Image
import concurrent
import random
import shutil
import gc

# data science
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import sklearn

# torch
import torch
import torchvision
import torchvision.transforms.v2 as transforms
import torchsummary
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics


# lightning
import lightning as L

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



#default dataset code from pytorch documentation
class ImagesDataset(torch.utils.data.Dataset):
    def __init__(self, annotations_file: str = "./", img_dir: str = "./", transform = None, target_transform = None):
        self.img_labels = pd.read_csv(annotations_file, header=None)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = torchvision.io.read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    def get_path(self,idx):
        return os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])


class MainDataModule(L.LightningDataModule):
    def __init__(self, data_urls: dict, transform = None, batch_size_train: int = 512):
        super().__init__()
        self.data_urls = data_urls
        self.transform = transform
        self.batch_size_train = batch_size_train



    def prepare_data(self):
        #download annotations
        os.system(f"curl -O {self.data_urls['annotations']}")
        #download training images
        os.system(f"curl -O {self.data_urls['training_images']}")
        #download validation images
        os.system(f"curl -O {self.data_urls['validation_images']}")

        # Extract tar files
        os.system("tar -xf annot.tar")
        os.system("tar -xf train.tar")
        os.system("tar -xf val.tar")

        # Clean up the working directory
        os.remove("annot.tar")
        os.remove("train.tar")
        os.remove("val.tar")
        os.remove("test_info.csv")

        # Rename directories and files
        os.rename("val_set", "test_set")
        os.rename("val_info.csv", "test_info.csv")
        os.rename("train_info.csv", "train_val_info.csv")

        #Extract the validation set
        os.makedirs("val_set", exist_ok=True)

        train_ratio = 0.8

        with open("train_val_info.csv", 'r') as f:
            annotations = f.readlines()

        random.shuffle(annotations)

        split_index = int(len(annotations) * train_ratio)

        train_annotations = annotations[:split_index]
        val_annotations = annotations[split_index:]

        with open("train_info.csv", 'w') as f:
            f.writelines(train_annotations)

        with open("val_info.csv", 'w') as f:
            f.writelines(val_annotations)


        val_image_names = [line.split(' ')[0] for line in val_annotations]

        for val_image in val_image_names:
            image_path = os.path.join("train_set", val_image)

            if os.path.exists(image_path):
                shutil.move(image_path, os.path.join("val_set", val_image))

        os.remove("train_val_info.csv")

    def setup(self, stage: str = "train"):

        # make assignments (val/train/test split)
        self.train = ImagesDataset(annotations_file = "train_info.csv", img_dir = "train_set", transform = self.transform)
        self.val = ImagesDataset(annotations_file = "train_info.csv", img_dir = "train_set", transform = self.transform)
        self.test = ImagesDataset(annotations_file = "test_info.csv",img_dir = "test_set", transform = self.transform)

        #define the label to class hashmap
        self.label_to_class = {}

        with open("class_list.txt", 'r') as file:
            for line in file:
                label, class_name = line.strip().split()
                self.label_to_class[label] = class_name


        self.train_labels = pd.read_csv("train_info.csv", header = None).iloc[:, 1].values
        self.val_labels = pd.read_csv("val_info.csv", header = None).iloc[:, 1].values
        self.test_labels = pd.read_csv("test_info.csv", header = None).iloc[:, 1].values

    #Dataloaders
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train, batch_size = self.batch_size_train, shuffle=False)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val, batch_size = 1, shuffle = False)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test, batch_size = 1, shuffle = False)


def get_spaced_permutations( n_elements, n_permutations, overlap_tolerance, verbose = False):

    permutations = [torch.arange(n_elements)]

    while len(permutations) < n_permutations + 1:

        current_permutation = torch.randperm(n_elements)

        if not any((current_permutation == perm).sum().item() > overlap_tolerance for perm in permutations):
            permutations.append(current_permutation)

    permutations = permutations[1:]


    if verbose:
        overlap_matrix = torch.zeros((n_permutations,n_permutations))

        for i, per in enumerate(permutations):
            for j, per_ in enumerate(permutations):
                overlap_matrix[i,j] = (per == per_).sum()

        overlap_matrix = overlap_matrix.numpy()

        print(f"overlap matrix: \n {overlap_matrix}")

    return permutations

unfold = torch.nn.Unfold(kernel_size = (64,64), stride=64, padding=0)
fold = torch.nn.Fold(output_size = (256,256), kernel_size = (64,64), stride=64, padding=0)

data_urls = {
    "annotations": "https://food-x.s3.amazonaws.com/annot.tar",
    "training_images": "https://food-x.s3.amazonaws.com/train.tar",
    "validation_images": "https://food-x.s3.amazonaws.com/val.tar"
}

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # ImageNet Normalisation
])


my_data = MainDataModule(data_urls = data_urls, transform = transform, batch_size_train = 64)

my_data.prepare_data()

my_data.setup()

os.makedirs("train_jigsaw", exist_ok=True)
os.makedirs("val_jigsaw", exist_ok=True)
os.makedirs("test_jigsaw", exist_ok=True)

dataloaders = [
    my_data.train_dataloader(),
    my_data.val_dataloader(),
    my_data.test_dataloader()
]

save_folders = [
    'train_jigsaw',
    'val_jigsaw',
    'test_jigsaw'
]



def data_augmentation(batch, permutations):

    x, y = batch

    #start with the identity permutation
    #the images remain the same, the labels become the identity permutation
    x_augmented = x.clone()
    y_augmented = torch.arange(permutations[0].shape[0]).repeat(x.shape[0],1)

    for permutation in permutations:
        x_ = unfold(x)
        x_ = x_[:, :, permutation]
        x_ = fold(x_)
        y_ = permutation.repeat(x.shape[0], 1)
        x_augmented = torch.cat((x_augmented, x_), dim=0)
        y_augmented = torch.cat((y_augmented, y_), dim=0)

    return x_augmented, y_augmented

permutations = get_spaced_permutations(16,16,1)
print(f"permutations: {permutations}")

to_pil = transforms.ToPILImage()

for dataloader, save_folder in tqdm(zip(dataloaders,save_folders)):

    annotations = []

    for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):

        x_augmented, y_augmented = data_augmentation(batch, permutations)

        print(f"x_augmented.shape: {x_augmented.shape}")
        print(f"y_augmented.shape: {y_augmented.shape}")

        for i in tqdm(range(x_augmented.shape[0])):

            image_tensor = x_augmented[i]
            label = y_augmented[i]

            image_tensor = image_tensor.clamp(0, 1)

            pil_image = to_pil(image_tensor)

            image_path = os.path.join(save_folder, f"img_{batch_idx}_{i:04}.png")
            pil_image.save(image_path)

            annotations.append((f"{save_folder}_{batch_idx}_{i:04}.png", label.tolist()))

    annotations_df = pd.DataFrame(annotations)
    annotations_df.to_csv(save_folder + "_info.csv", index=False, header=False)