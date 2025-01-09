#utilities
import glob
import os
from tqdm.auto import tqdm
import wandb
import joblib
import time
from PIL import Image
import concurrent

# data science
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import sklearn

#computer vision
import cv2

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

#default dataset code from pytorch documentation
class ImagesDataset(torch.utils.data.Dataset):
    def __init__(self, annotations_file: str = "./", img_dir: str = "./", transform = None, target_transform = None):
        self.img_labels = pd.read_csv(annotations_file)
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
    
class FeaturesDataset(torch.utils.data.Dataset):
    def __init__(self, annotations_file: str = "./", feat_dir: str = "./"):
        self.labels = pd.read_csv(annotations_file)
        self.feat_dir = feat_dir

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        feat_path = os.path.join(self.feat_dir, self.labels.iloc[idx, 0].replace('.jpg', '.npy'))
        feat = np.load(feat_path, allow_pickle=True)
        label = self.labels.iloc[idx, 1]

        return feat, label

class ScrambledImagesDataset(torch.utils.data.Dataset):
    def __init__(self, annotations_file: str = "./", img_dir: str = "./", transform = None, target_transform = None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = torchvision.io.read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        label_tensor = torch.tensor(label, dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, label_tensor

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
        os.system(f"wget {self.data_urls['annotations']}")
        #download training images
        os.system(f"wget {self.data_urls['training_images']}")
        #download validation images
        os.system(f"wget {self.data_urls['validation_images']}")

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

    def setup(self, stage: str = "train"):

        # make assignments (val/train/test split)
        self.train_val = ImagesDataset(annotations_file = "train_info.csv", img_dir = "train_set", transform = self.transform)

        self.train, self.val = torch.utils.data.random_split(
            self.train_val,
            [0.8, 0.2],
            generator = torch.Generator().manual_seed(42)
        )

        self.test = ImagesDataset(annotations_file = "test_info.csv",img_dir = "test_set", transform = self.transform)

        #define the label to class hashmap
        self.label_to_class = {}

        with open("class_list.txt", 'r') as file:
            for line in file:
                label, class_name = line.strip().split()
                self.label_to_class[label] = class_name

        self.train_val_labels = pd.read_csv("train_info.csv", header = None).iloc[:, 1].values
        self.test_labels = pd.read_csv("test_info.csv", header = None).iloc[:, 1].values

        # Use the indices to split the labels
        self.train_labels = self.train_val_labels[self.train.indices]
        self.val_labels = self.train_val_labels[self.val.indices]



    #Dataloaders
    def train_dataloader(self, num_workers):
        return torch.utils.data.DataLoader(self.train, batch_size = self.batch_size_train, shuffle=True, num_workers= num_workers)

    def val_dataloader(self, num_workers):
        return torch.utils.data.DataLoader(self.val, batch_size = 1, shuffle = False, num_workers= num_workers)

    def test_dataloader(self, num_workers):
        return torch.utils.data.DataLoader(self.test, batch_size = 1, shuffle = False, num_workers= num_workers)




    def visualise_some_images(self, n: int = 20):

        #imagenet normalisation
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])

        images, labels = next(iter(self.train_dataloader()))
        image, labels =  images[:n], labels[:n]
        n_rows = max(int((n-1)/10)+1,2)
        fig, axes = plt.subplots(n_rows, 10, figsize=(50, n_rows * 5) )


        for i in range(n):
            image = images[i]
            label = labels[i]
            image = image.mul(std.unsqueeze(1).unsqueeze(2))
            image = image.add(mean.unsqueeze(1).unsqueeze(2))
            image = image.clamp(0, 1)
            image = image.permute(1, 2, 0).numpy()

            axes[int(i/10),i%10].imshow(image)
            axes[int(i/10),i%10].set_title(self.label_to_class[str(label.item())])
            axes[int(i/10),i%10].axis('off')

        plt.show()


    def extract_sift_features(self, sift_mode: str = "normal", dense_sift_step: int = 8):

        # Create the directories
        os.makedirs("test_features", exist_ok=True)
        os.makedirs("train_features", exist_ok=True)
        os.makedirs("test_features/descriptors", exist_ok=True)
        os.makedirs("test_features/keypoints", exist_ok=True)
        os.makedirs("train_features/descriptors", exist_ok=True)
        os.makedirs("train_features/keypoints", exist_ok=True)

        sift = cv2.SIFT_create()

        for dataset, folder in [(self.train_val, "train_features"), (self.test, "test_features")]:

            for i in tqdm(range(len(dataset))):

                image_path = dataset.get_path(i)

                image = cv2.imread(image_path)

                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                ## keypoints computation
                if sift_mode == "normal":
                    #keypoints for normal sift
                    kp = sift.detect(gray_image, None)

                elif sift_mode == "dense":
                    #keypoints for dense sift (sorry)
                    kp = [cv2.KeyPoint(x, y, dense_sift_step) for y in range(0, gray_image.shape[0], dense_sift_step) for x in range(0, gray_image.shape[1], dense_sift_step)]

                else:
                    raise Exception("sift_mode should be either normal or dense")

                ## descriptors computation
                kp, des = sift.compute(gray_image, kp)

                ## save everything

                # descriptors
                file_name = os.path.splitext(os.path.basename(image_path))[0]
                np.save(os.path.join(folder, "descriptors", file_name + ".npy") , des)

                # keypoints
                kp = np.array([[
                    kp_.pt[0],  # x-coordinate
                    kp_.pt[1],  # y-coordinate
                    kp_.size,   # size
                    kp_.angle,  # angle
                    kp_.response,  # response
                    kp_.octave,  # octave
                    kp_.class_id   # class_id
                ] for kp_ in kp])

                np.save(os.path.join(folder, "keypoints", file_name + ".npy") , kp)


    def setup_sift_datasets(self):

        # des
        self.train_val_sift_des = FeaturesDataset(annotations_file = "train_info.csv", feat_dir = "train_features/descriptors")

        self.train_sift_des, self.val_sift_des = torch.utils.data.random_split(
            self.train_val_sift_des,
            [0.8, 0.2],
            generator = torch.Generator().manual_seed(42)
        )

        self.test_sift_des = FeaturesDataset(annotations_file = "test_info.csv",feat_dir = "test_features/descriptors")


        # kp
        self.train_val_sift_kp = FeaturesDataset(annotations_file = "train_info.csv", feat_dir = "train_features/keypoint")

        self.train_sift_kp, self.val_sift_kp = torch.utils.data.random_split(
            self.train_val_sift_kp,
            [0.8, 0.2],
            generator = torch.Generator().manual_seed(42)
        )

        self.test_sift_kp = FeaturesDataset(annotations_file = "test_info.csv",feat_dir = "test_features/keypoints")



    def extract_bow_features(self, vocabulary_size: int = 1000, batch_size = 10000):

        '''
        this method will extract the bow descriptions for the train, val and
        test sets, it requires sift features, so both extract sift_features and
        setup_sift_datasets must have been called before it
        '''


        kmeans = sklearn.cluster.MiniBatchKMeans(n_clusters = vocabulary_size, batch_size = batch_size)


        #get all descriptors

        for dataset in [self.train_sift_des, self.val_sift_des]:

            n_batches = int(len(dataset) / batch_size) + 1

            for batch in tqdm(range(n_batches)):

                des = []

                for i in tqdm(range(batch_size)):

                    idx = batch*batch_size + i

                    if idx >= len(dataset):
                        continue

                    descriptor = dataset[idx][0]

                    if len(descriptor.shape) == 2:
                        des.append(descriptor)

                des = np.concatenate(des, axis=0)

                kmeans.partial_fit(des)



        #build the bow matrices
        for dataset, bow_name in [(self.train_sift_des, "train_bow"), (self.val_sift_des, "val_bow"), (self.test_sift_des, "test_bow")]:

            current_bow = np.zeros((len(dataset), vocabulary_size))

            for i in tqdm(range(len(dataset))):

                descriptor = dataset[i][0]

                if len(descriptor.shape) == 2:
                    predictions = kmeans.predict(descriptor)
                    current_bow[i,:] = np.bincount(predictions, minlength = vocabulary_size)

                else:
                    print(f"anomaly at index {i}")
                    current_bow[i,:] = np.zeros((1, vocabulary_size))


            setattr(self, bow_name, current_bow)


        np.save("train_features/train_bow.npy", self.train_bow)
        np.save("train_features/val_bow.npy", self.val_bow)
        np.save("test_features/test_bow.npy", self.test_bow)


    def setup_bow_features(self):
        self.train_bow = np.load("train_features/train_bow.npy")
        self.val_bow = np.load("train_features/val_bow.npy")
        self.test_bow = np.load("test_features/test_bow.npy")



unfold = torch.nn.Unfold(kernel_size = (64,64), stride=64, padding=0)
fold = torch.nn.Fold(output_size = (256,256), kernel_size = (64,64), stride=64, padding=0)

def augment_single_permutation(permutation, x, unfold = unfold, fold = fold):
    x_ = unfold(x)
    x_ = x_[:, :, permutation]
    x_ = fold(x_)
    y_ = permutation.repeat(x.shape[0], 1)
    return x_, y_

class SslDataModule(L.LightningDataModule):
    def __init__(self, parent_module, batch_size_train: int =8):
        super().__init__()
        self.parent_module = parent_module
        self.batch_size_train = batch_size_train

        self.unfold = torch.nn.Unfold(kernel_size = (64,64), stride=64, padding=0)
        self.fold = torch.nn.Fold(output_size = (256,256), kernel_size = (64,64), stride=64, padding=0)

    def prepare_data(self):

        os.makedirs("train_jigsaw", exist_ok=True)
        os.makedirs("val_jigsaw", exist_ok=True)
        os.makedirs("test_jigsaw", exist_ok=True)
        
        dataloaders = [
            self.parent_module.train_dataloader(num_workers = os.cpu_count()),
            self.parent_module.val_dataloader(num_workers = os.cpu_count()),
            self.parent_module.test_dataloader(num_workers = os.cpu_count())
        ]

        save_folders = [
            'train_jigsaw',
            'val_jigsaw',
            'test_jigsaw'
        ]

        self.permutations = self.get_spaced_permutations(16,16,1)



        def save_image_and_annotation(image_tensor, label, save_folder, batch_idx, idx):
            image_tensor = image_tensor.clamp(0, 1)
            to_pil = transforms.ToPILImage()
            pil_image = to_pil(image_tensor)

            image_path = os.path.join(save_folder, f"img_{batch_idx}_{idx:04}.png")
            pil_image.save(image_path)

            # Return the annotation for the saved image
            return (f"img_{batch_idx}_{idx:04}.png", label.tolist())

        with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:

            for dataloader, save_folder in tqdm(zip(dataloaders,save_folders)):

                annotations = []
                futures = []

                for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):

                    x_augmented, y_augmented = self.data_augmentation(batch, self.permutations)

                    for i in range(x_augmented.shape[0]):
                        image_tensor = x_augmented[i]
                        label = y_augmented[i]

                        future = executor.submit(save_image_and_annotation,
                            image_tensor,
                            label,
                            save_folder,
                            batch_idx,
                            i
                        )

                        futures.append(future)


                for future in concurrent.futures.as_completed(futures):
                    annotations.append(future.result())
                annotations_df = pd.DataFrame(annotations)
                annotations_df.to_csv(save_folder + "_info.csv", index=False, header=False)




    def setup(self, stage: str = "train"):

        # make assignments (val/train/test split)
        self.train = ScrambledImagesDataset(
            annotations_file = "train_jigsaw_info.csv",
            img_dir = "train_jigsaw"
        )

        self.val = ScrambledImagesDataset(
            annotations_file = "val_jigsaw_info.csv",
            img_dir = "val_jigsaw"
        )

        self.test = ScrambledImagesDataset(
            annotations_file = "test_jigsaw_info.csv",
            img_dir = "test_jigsaw"
        )



        self.train_labels = pd.read_csv("test_jig_saw_info.csv", header = None).iloc[:, 1].values
        self.val_labels = pd.read_csv("test_jigsaw_info.csv", header = None).iloc[:, 1].values
        self.test_labels = pd.read_csv("test_jigsaw_info.csv", header = None).iloc[:, 1].values



    #Dataloaders
    def train_dataloader(self, num_workers):
        return torch.utils.data.DataLoader(self.train,batch_size = self.batch_size_train, shuffle=True, num_workers=num_workers)

    def val_dataloader(self, num_workers):
        return torch.utils.data.DataLoader(self.val, batch_size = 1, shuffle = False, num_workers=num_workers)

    def test_dataloader(self, num_workers):
        return torch.utils.data.DataLoader(self.test, batch_size = 1, shuffle = False, num_workers=num_workers)



    def visualise_some_images(self, n: int = 20):

        #imagenet normalisation
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])

        images, labels = next(iter(self.train_dataloader()))
        image, labels =  images[:n], labels[:n]
        n_rows = max(int((n-1)/10)+1,2)
        fig, axes = plt.subplots(n_rows, 10, figsize=(50, n_rows * 5) )


        for i in range(n):
            image = images[i]
            label = labels[i]
            image = image.mul(std.unsqueeze(1).unsqueeze(2))
            image = image.add(mean.unsqueeze(1).unsqueeze(2))
            image = image.clamp(0, 1)
            image = image.permute(1, 2, 0).numpy()

            axes[int(i/10),i%10].imshow(image)
            axes[int(i/10),i%10].set_title(self.label_to_class[str(label.item())])
            axes[int(i/10),i%10].axis('off')

        plt.show()


    def get_spaced_permutations(self, n_elements, n_permutations, overlap_tolerance, verbose = False):

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


    def data_augmentation(self, batch, permutations):

        x, y = batch

        #start with the identity permutation
        #the images remain the same, the labels become the identity permutation
        x_augmented = x.clone()
        y_augmented = torch.arange(self.permutations[0].shape[0]).repeat(x.shape[0],1)

        results = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            for perm in permutations:
                future = executor.submit(augment_single_permutation, perm, x)
                results.append(future)

        for future in results:
            x_, y_ = future.result()
            x_augmented = torch.cat((x_augmented, x_), dim=0)
            y_augmented = torch.cat((y_augmented, y_), dim=0)

        return x_augmented, y_augmented


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



my_data = MainDataModule(data_urls = data_urls, transform = transform, batch_size_train = 128)

my_data.prepare_data()

my_data.setup()

my_ssl_data = SslDataModule(my_data)

my_ssl_data.prepare_data()

my_ssl_data.setup()

