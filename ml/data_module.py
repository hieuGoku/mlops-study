'''Data module for Food3 dataset.'''

import os
import zipfile
from pathlib import Path
import requests
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from ml.dataset import Food3Dataset


class Food3DataModule(pl.LightningDataModule):
    '''Food3 data module.'''

    def __init__(self, batch_size: int = 32, num_workers: int = 2) -> None:
        '''Initializes the data module.'''
        super().__init__()
        self.train_data = None
        self.val_data = None
        self.batch_size = batch_size
        self.num_workers = num_workers


    def prepare_data(self) -> None:
        # pylint: disable=line-too-long
        image_path = self.download_data(source="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip",
                                        destination="pizza_steak_sushi")
        self.train_data = image_path / "train"
        self.val_data = image_path / "test"

        print('[INFO] Data preparation complete!')


    def setup(self, stage=None) -> None:
        '''Sets up the data module.'''
        if stage == "fit" or stage is None:
            self.train_data = Food3Dataset(self.train_data, train=True)
            self.val_data = Food3Dataset(self.val_data, train=False)

            print('[INFO] Data setup complete!')


    def train_dataloader(self) -> DataLoader:
        '''Returns the training dataloader.'''
        return self.dataloader(self.train_data, train=True)


    def val_dataloader(self) -> DataLoader:
        '''Returns the validation dataloader.'''
        return self.dataloader(self.val_data, train=False)


    def dataloader(self, dataset: Dataset, train: bool = False) -> DataLoader:
        '''Returns a dataloader for a dataset.'''
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=train,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=train,
        )


    def download_data(
                    self, source: str,
                    destination: str,
                    remove_source: bool = True
                ) -> Path:
        # pylint: disable=line-too-long
        '''Downloads a zipped dataset from source and unzips to destination.

        Args:
            source (str): A link to a zipped file containing data.
            destination (str): A target directory to unzip data to.
            remove_source (bool): Whether to remove the source after downloading and extracting.

        Returns:
            pathlib.Path to downloaded data.

        Example usage:
            download_data(source="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip",
                        destination="pizza_steak_sushi")
        '''
        # Setup path to data folder
        data_path = Path("data/")
        image_path = data_path / destination

        # If the image folder doesn't exist, download it and prepare it...
        if image_path.is_dir():
            print(f"[INFO] {image_path} directory exists, skipping download.")
        else:
            print(
                f"[INFO] Did not find {image_path} directory, creating one...")
            image_path.mkdir(parents=True, exist_ok=True)

            # Download pizza, steak, sushi data
            target_file = Path(source).name
            with open(data_path / target_file, "wb") as file:
                request = requests.get(source,timeout=60)
                print(f"[INFO] Downloading {target_file} from {source}...")
                file.write(request.content)

            # Unzip pizza, steak, sushi data
            with zipfile.ZipFile(data_path / target_file, "r") as zip_ref:
                print(f"[INFO] Unzipping {target_file} data...")
                zip_ref.extractall(image_path)

            # Remove .zip file
            if remove_source:
                os.remove(data_path / target_file)

        return image_path
