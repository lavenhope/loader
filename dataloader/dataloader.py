import torch
import os

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

from datasets.pandadataset import PandaDataset
from datasets.augmentation import get_augmentation, Resizer, Normalizer, Augmenter, collater

BATCH_SIZE = 64

if __name__ == "__main__":
    train_dataset = PandaDataset(
                        rootdir = 'F:/Code/panda',
                        set_name = 'train',
                        transform = transforms.Compose(
                                      [Normalizer(),
                                       Augmenter(),
                                       Resizer()]))
    train_loader = DataLoader(
                        train_dataset,
                        batch_size = BATCH_SIZE,
                        shuffle = True,
                        collate_fn = collater,
                        pin_memory = True)

    print(train_dataset)
    print('-----超级分割线-----')
    print(train_loader)