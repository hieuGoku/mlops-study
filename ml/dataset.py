'''Dataset module for Food3 dataset.'''

import os
from pathlib import Path
from typing import Tuple, Dict, List
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms


class Food3Dataset(Dataset):
    '''Food3 dataset.'''

    def __init__(self, targ_dir: str, train=None) -> None:
        '''Initializes the dataset.'''

        self.paths = list(Path(targ_dir).glob("*/*.jpg"))
        self.train = train
        self.classes, self.class_to_idx = self.find_classes(targ_dir)

        if self.train:
            print(f'[INFO] Found {len(self.paths)} training images, belonging to {len(self.classes)} classes.')
            print(f'[INFO] Class to index mapping: {self.class_to_idx}')
        else:
            print(f'[INFO] Found {len(self.paths)} validation images, belonging to {len(self.classes)} classes.')
            print(f'[INFO] Class to index mapping: {self.class_to_idx}')

        self.train_transforms = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor()
        ])

        self.test_transforms = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])

    def load_image(self, index: int) -> Image.Image:
        '''Opens an image via a path and returns it.'''
        image_path = self.paths[index]
        return Image.open(image_path)

    def __len__(self) -> int:
        '''Returns the length of the dataset.'''
        return len(self.paths)

    def __getitem__(self, index: int) -> Tuple[Tensor, int]:
        '''Returns one sample of data, data and label (X, y).'''
        img = self.load_image(index)
        # expects path in data_folder/class_name/image.jpeg
        class_name = self.paths[index].parent.name
        class_idx = self.class_to_idx[class_name]

        if self.train:
            return self.train_transforms(img), class_idx
        return self.test_transforms(img), class_idx

    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        '''Finds the class folder names in a target directory.

        Assumes target directory is in standard image classification format.

        Args:
            directory (str): target directory to load classnames from.

        Returns:
            Tuple[List[str], Dict[str, int]]: (list_of_class_names, dict(class_name: idx...))

        Example:
            find_classes("food_images/train")
            >>> (["class_1", "class_2"], {"class_1": 0, ...})
        '''
        # 1. Get the class names by scanning the target directory
        classes = sorted(entry.name for entry in os.scandir(
            directory) if entry.is_dir())

        # 2. Raise an error if class names not found
        if not classes:
            raise FileNotFoundError(
                f"Couldn't find any classes in {directory}.")

        # 3. Crearte a dictionary of index labels
        # (computers prefer numerical rather than string labels)
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx
