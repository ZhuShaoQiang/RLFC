# -*- coding: utf-8 -*-

"""
这个里面存放经验
经验是一个图片序列，每个经验是一个dataloader
"""
import os
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from natsort import natsorted

class ImageDataset(Dataset):
    def __init__(self, experiment_dir, transform=None):
        self.experiment_dir = experiment_dir
        _image_files = os.listdir(experiment_dir)
        self.image_files = natsorted(_image_files)
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.experiment_dir, self.image_files[idx])
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image

class ExpDataset(Dataset):
    def __init__(self, root_dir, to_size: tuple=(96, 96), gray_scale=True,
                 extra_trans: list=[]):
        self.root_dir = root_dir
        self.experiments = os.listdir(root_dir)

        _transform = [
            transforms.Resize(to_size)
        ]
        if gray_scale:
            _transform.append(transforms.Grayscale())
        
        _transform = _transform+extra_trans

        _transform.append(transforms.ToTensor())

        self.transform = transforms.Compose(_transform)

    def __len__(self):
        return len(self.experiments)

    def __getitem__(self, idx):
        experiment_dir = os.path.join(self.root_dir, self.experiments[idx])
        experiment_dataset = ImageDataset(experiment_dir, transform=self.transform)
        return experiment_dataset

def generate_exps(
        split_ratio=0.3, 
        to_size=(96, 96), **kwargs):
    """
    产生经验
    这个地方一次产生一个经验片段，这个经验片段和普通的经验片段不一样，这个经验片段是一系列的图片
    这个返回值是一个dataloader的generator
    because we seperate the dataset by hand, so this split_ratio is abandoned
    """
    root_dir = './exp/carracing_exp/'
    train_dir = root_dir+"train/"
    test_dir = root_dir+"test/"
    train_dataset = ExpDataset(train_dir, to_size=to_size)  # get dataset_train
    test_dataset = ExpDataset(test_dir, to_size=to_size)  # get dataset_train
    return train_dataset, test_dataset

if __name__ == "__main__":
    """
    测试加载经验的函数的
    """
    # Example usage
    root_dir = './exp/carracing_exp/'
    train_dir = root_dir+"train/"
    test_dir = root_dir+"test/"
    dataset = ExpDataset(train_dir)  # get a dataset
    print(f"get a dataset of train_dir")
    exp_dataset = dataset[0]  # Get the first experiment's dataset
    print(f"exp_datset: {exp_dataset}")
    print(type(exp_dataset))
    exit()

    # Create a DataLoader for the first experiment's dataset
    exp_loader = DataLoader(
        exp_dataset,
        batch_size=1,  # Set your desired batch size
        shuffle=False,  # Set to True if you want to shuffle the images within each experiment
        num_workers=0  # Set the number of workers for data loading
    )
    for exp in exp_loader:
        print(exp)