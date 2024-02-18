import os
import numpy as np
import torch
from torch import Tensor
from pathlib import Path
from typing import List, Optional, Sequence, Union, Any, Callable
from torchvision.datasets.folder import default_loader
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import transforms
from torchvision.datasets import CelebA
import zipfile

def portion_data(raw_data, data_portion, time_steps, shuffle):
    if data_portion == 1.0 and time_steps == 49:
        return raw_data
    if shuffle:
        np.random.shuffle(raw_data)
    num_trajs = raw_data.shape[0]
    num_times = raw_data.shape[0]
    return raw_data[:int(num_trajs * data_portion), :int(time_steps), :, :]

def slides_t0(raw_data, lag_time, slide=1):
    t0 = np.concatenate([d[j::lag_time][:-1] for d in raw_data for j in range(slide)], axis=0)
    return t0

def slides_t1(raw_data, lag_time, slide=1):
    t1 = np.concatenate([d[j::lag_time][1:] for d in raw_data for j in range(slide)], axis=0)
    return t1


def load_phy_edges(args):
    keep_str = args.data_path.split('/')[-1].split('_', 2)[-1]
    if args.suffix != "charged":
        root_str = args.data_path[::-1].split('/', 1)[1][::-1] + '/'
    else:
        root_str = args.data_path[::-1].split('/', 1)[1][::-1]

    edges = np.load(root_str + 'edges_train_' + keep_str)
    if args.suffix != "charged":
        edges[edges > 0] = 1
    edges = np.reshape(edges, (-1))
    return edges


def load_netsims_edges(args):
    keep_str = args.data_path.split('/')[-1].split('_', 2)[-1]
    root_str = args.data_path[::-1].split('/', 1)[1][::-1] + '/'

    edges = np.load(root_str + 'edges_train_' + keep_str)
    edges[edges > 0] = 1
    edges = np.reshape(edges, (-1))
    return edges

def load_biological_edges(args):
    keep_str = args.data_path.split('/')[-1].split('_', 2)[-1]
    root_str = args.data_path[::-1].split('/', 1)[1][::-1] + '/'

    edges = np.load(root_str + keep_str)
    edges[edges > 0] = 1
    edges = np.reshape(edges, (-1))
    return edges

def load_edges(args):
    print("In load_edges.", args.suffix)
    if args.suffix in ['LI', 'LL', 'CY', 'BF', 'TF', 'BF-CV']:
        edges = load_biological_edges(args)
    elif args.suffix == "springs" or "charged":
        edges = load_phy_edges(args)
    elif args.suffix == "netsims":
        edges = load_netsims_edges(args)
    else:
        raise ValueError("Dataset not implemented yet.")
    return edges


def load_customized_springs_data(args):
    keep_str = args.data_path.split('/')[-1].split('_', 2)[-1]
    if args.suffix != "charged":
        root_str = args.data_path[::-1].split('/', 1)[1][::-1] + '/'
    else:
        root_str = args.data_path[::-1].split('/', 1)[1][::-1]

    loc_train = np.load(root_str + 'loc_train_' + keep_str)
    vel_train = np.load(root_str + 'vel_train_' + keep_str)
    edges = np.load(root_str + 'edges_train_' + keep_str)
    if args.suffix != "charged":
        edges[edges > 0] = 1

    loc_valid = np.load(root_str + 'loc_valid_' + keep_str)
    vel_valid = np.load(root_str + 'vel_valid_' + keep_str)

    loc_test = np.load(root_str + 'loc_test_' + keep_str)
    vel_test = np.load(root_str + 'vel_test_' + keep_str)

    loc_train = portion_data(loc_train, args.b_portion, args.b_time_steps, args.b_shuffle)
    vel_train = portion_data(vel_train, args.b_portion, args.b_time_steps, args.b_shuffle)

    loc_valid = portion_data(loc_valid, args.b_portion, args.b_time_steps, args.b_shuffle)
    vel_valid = portion_data(vel_valid, args.b_portion, args.b_time_steps, args.b_shuffle)

    print("We have {} training simulations, {} validation simulations, and {} test simulations.".format(
        loc_train.shape[0], loc_valid.shape[0], loc_test.shape[0]))

    loc_max = loc_train.max()
    loc_min = loc_train.min()
    vel_max = vel_train.max()
    vel_min = vel_train.min()

    loc_train = (loc_train - loc_min) * 2 / (loc_max - loc_min) - 1
    vel_train = (vel_train - vel_min) * 2 / (vel_max - vel_min) - 1

    loc_valid = (loc_valid - loc_min) * 2 / (loc_max - loc_min) - 1
    vel_valid = (vel_valid - vel_min) * 2 / (vel_max - vel_min) - 1

    loc_test = (loc_test - loc_min) * 2 / (loc_max - loc_min) - 1
    vel_test = (vel_test - vel_min) * 2 / (vel_max - vel_min) - 1

    # Reshape to: [num_sims, num_timesteps, num_nodes, num_dims]
    loc_train = np.transpose(loc_train, [0, 3, 1, 2])
    vel_train = np.transpose(vel_train, [0, 3, 1, 2])
    feat_train = np.concatenate([loc_train, vel_train], axis=3)

    loc_valid = np.transpose(loc_valid, [0, 3, 1, 2])
    vel_valid = np.transpose(vel_valid, [0, 3, 1, 2])
    feat_valid = np.concatenate([loc_valid, vel_valid], axis=3)

    loc_test = np.transpose(loc_test, [0, 3, 1, 2])
    vel_test = np.transpose(vel_test, [0, 3, 1, 2])
    feat_test = np.concatenate([loc_test, vel_test], axis=3)

    feat_train = np.reshape(feat_train, (-1, feat_train.shape[2], feat_train.shape[3]))
    feat_train_t0 = torch.FloatTensor(slides_t0(feat_train, args.lag_time, args.slide))
    feat_train_t1 = torch.FloatTensor(slides_t1(feat_train, args.lag_time, args.slide))
    train_data = TensorDataset(feat_train_t0, feat_train_t1)

    feat_valid = np.reshape(feat_valid, (-1, feat_valid.shape[2], feat_valid.shape[3]))
    feat_valid_t0 = torch.FloatTensor(slides_t0(feat_valid, args.lag_time, args.slide))
    feat_valid_t1 = torch.FloatTensor(slides_t1(feat_valid, args.lag_time, args.slide))
    valid_data = TensorDataset(feat_valid_t0, feat_valid_t1)

    feat_test = np.reshape(feat_test, (-1, feat_test.shape[2], feat_test.shape[3]))
    feat_test_t0 = torch.FloatTensor(slides_t0(feat_test, args.lag_time, args.slide))
    feat_test_t1 = torch.FloatTensor(slides_t1(feat_test, args.lag_time, args.slide))
    test_data = TensorDataset(feat_test_t0, feat_test_t1)

    # Exclude self edges: discarded
    # off_diag_idx = np.ravel_multi_index(
    #     np.where(np.ones((num_atoms, num_atoms)) - np.eye(num_atoms)),
    #     [num_atoms, num_atoms])
    # edges_train = edges_train[:, off_diag_idx]
    # edges_valid = edges_valid[:, off_diag_idx]
    # edges_test = edges_test[:, off_diag_idx]

    edges = np.reshape(edges, (-1))  # Flattened into the shape (num_nodes ** 2), np array.

    # train_data_loader = DataLoader(train_data, batch_size=args.batch_size)
    # valid_data_loader = DataLoader(valid_data, batch_size=args.batch_size)
    # test_data_loader = DataLoader(test_data, batch_size=args.batch_size)

    return train_data, valid_data, test_data, edges  #, loc_max, loc_min, vel_max, vel_min


def load_customized_netsims_data(args):

    keep_str = args.data_path.split('/')[-1].split('_', 2)[-1]
    root_str = args.data_path[::-1].split('/', 1)[1][::-1] + '/'

    bold_train = np.load(root_str + 'bold_train_' + keep_str)
    edges = np.load(root_str + 'edges_train_' + keep_str)
    edges[edges > 0] = 1

    bold_valid = np.load(root_str + 'bold_valid_' + keep_str)

    bold_test = np.load(root_str + 'bold_test_' + keep_str)
    bold_train = portion_data(bold_train, args.b_portion, args.b_time_steps, args.b_shuffle)
    bold_valid = portion_data(bold_valid, args.b_portion, args.b_time_steps, args.b_shuffle)

    bold_max = bold_train.max()
    bold_min = bold_train.min()

    bold_train = (bold_train - bold_min) * 2 / (bold_max - bold_min) - 1
    bold_valid = (bold_valid - bold_min) * 2 / (bold_max - bold_min) - 1
    bold_test = (bold_test - bold_min) * 2 / (bold_max - bold_min) - 1

    # Reshape to: [num_sims, num_timesteps, num_nodes, num_dims]
    feat_train = np.transpose(bold_train, [0, 3, 1, 2])
    feat_train = np.reshape(feat_train, (-1, feat_train.shape[2], feat_train.shape[3]))
    feat_train_t0 = torch.FloatTensor(slides_t0(feat_train, args.lag_time, args.slide))
    feat_train_t1 = torch.FloatTensor(slides_t1(feat_train, args.lag_time, args.slide))
    train_data = TensorDataset(feat_train_t0, feat_train_t1)

    feat_valid = np.transpose(bold_valid, [0, 3, 1, 2])
    feat_valid = np.reshape(feat_valid, (-1, feat_valid.shape[2], feat_valid.shape[3]))
    feat_valid_t0 = torch.FloatTensor(slides_t0(feat_valid, args.lag_time, args.slide))
    feat_valid_t1 = torch.FloatTensor(slides_t1(feat_valid, args.lag_time, args.slide))
    valid_data = TensorDataset(feat_valid_t0, feat_valid_t1)

    feat_test = np.transpose(bold_test, [0, 3, 1, 2])
    feat_test = np.reshape(feat_test, (-1, feat_test.shape[2], feat_test.shape[3]))
    feat_test_t0 = torch.FloatTensor(slides_t0(feat_test, args.lag_time, args.slide))
    feat_test_t1 = torch.FloatTensor(slides_t1(feat_test, args.lag_time, args.slide))
    test_data = TensorDataset(feat_test_t0, feat_test_t1)

    edges = np.reshape(edges, (-1))  # Flattened into the shape (num_nodes ** 2), np array.

    # Exclude self edges: discarded
    # off_diag_idx = np.ravel_multi_index(
    #     np.where(np.ones((num_atoms, num_atoms)) - np.eye(num_atoms)),
    #     [num_atoms, num_atoms])
    # edges_train = edges_train[:, off_diag_idx]
    # edges_valid = edges_valid[:, off_diag_idx]
    # edges_test = edges_test[:, off_diag_idx]

    # train_data_loader = DataLoader(train_data, batch_size=args.batch_size)
    # valid_data_loader = DataLoader(valid_data, batch_size=args.batch_size)
    # test_data_loader = DataLoader(test_data, batch_size=args.batch_size)

    return train_data, valid_data, test_data, edges  # bold_max, bold_min, bold_max, bold_min

def load_data_biological(args):

    keep_str = args.data_path.split('/')[-1].split('_', 2)[-1]
    root_str = args.data_path[::-1].split('/', 1)[1][::-1] + '/'

    train_traj = np.load(root_str + 'train.npy')
    # shape:[num_simulations, num_genes, time_steps]

    n_train = train_traj.shape[0]
    train_traj = np.transpose(train_traj, [0, 2, 1])  # change to [num_simulations, timesteps, num_genes]
    train_traj = train_traj[..., np.newaxis]  # shape: [num_sim, timesteps, num_genes, dimension]
    train_traj = np.transpose(train_traj, [0, 1, 3, 2])  # shape: [num_sim, timesteps, dimension, num_genes]
    edges = np.load(root_str + keep_str)

    valid_traj = np.load(root_str  + 'valid.npy')
    n_valid = valid_traj.shape[0]
    valid_traj = np.transpose(valid_traj, [0, 2, 1])  # change to [num_simulations, timesteps, num_genes]
    valid_traj = valid_traj[..., np.newaxis]  # shape: [num_sim, timesteps, num_genes, dimension]
    valid_traj = np.transpose(valid_traj, [0, 1, 3, 2])

    test_traj = np.load(root_str + 'test.npy')
    n_test = test_traj.shape[0]
    test_traj = np.transpose(test_traj, [0, 2, 1])  # change to [num_simulations, timesteps, num_genes]
    test_traj = test_traj[..., np.newaxis]  # shape: [num_sim, timesteps, num_genes, dimension]
    test_traj = np.transpose(test_traj, [0, 1, 3, 2])

    print("We have {} training simulations, {} validation simulations, and {} test simulations.".format(
        train_traj.shape[0], valid_traj.shape[0], test_traj.shape[0]))
    loc_max = train_traj.max()
    loc_min = train_traj.min()

    norm_train = (train_traj - loc_min) * 2 / (loc_max - loc_min) - 1

    norm_valid = (valid_traj - loc_min) * 2 / (loc_max - loc_min) - 1

    norm_test = (test_traj - loc_min) * 2 / (loc_max - loc_min) - 1

    # Reshape to: [num_sims, num_genes, num_timesteps, num_dims]

    # NOTE: added normalization on Jun.29
    # feat_train = np.transpose(train_traj, [0, 3, 1, 2])  # without normalization
    feat_train = np.transpose(norm_train, [0, 3, 1, 2])

    # feat_valid = np.transpose(valid_traj, [0, 3, 1, 2])  # without normalization
    feat_valid = np.transpose(norm_valid, [0, 3, 1, 2])

    # feat_test = np.transpose(test_traj, [0, 3, 1, 2])  # without normalization
    feat_test = np.transpose(norm_test, [0, 3, 1, 2])

    feat_train_t0 = torch.FloatTensor(slides_t0(feat_train, args.lag_time, args.slide))
    feat_train_t1 = torch.FloatTensor(slides_t1(feat_train, args.lag_time, args.slide))
    train_data = TensorDataset(feat_train_t0, feat_train_t1)

    feat_valid = np.reshape(feat_valid, (-1, feat_valid.shape[2], feat_valid.shape[3]))
    feat_valid_t0 = torch.FloatTensor(slides_t0(feat_valid, args.lag_time, args.slide))
    feat_valid_t1 = torch.FloatTensor(slides_t1(feat_valid, args.lag_time, args.slide))
    valid_data = TensorDataset(feat_valid_t0, feat_valid_t1)

    feat_test = np.reshape(feat_test, (-1, feat_test.shape[2], feat_test.shape[3]))
    feat_test_t0 = torch.FloatTensor(slides_t0(feat_test, args.lag_time, args.slide))
    feat_test_t1 = torch.FloatTensor(slides_t1(feat_test, args.lag_time, args.slide))
    test_data = TensorDataset(feat_test_t0, feat_test_t1)

    edges = np.reshape(edges, (-1))

    return train_data, valid_data, test_data, edges  #, loc_max, loc_min, vel_max, vel_min

# Add your custom dataset class here
class MyDataset(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass


class MyCelebA(CelebA):
    """
    A work-around to address issues with pytorch's celebA dataset class.

    Download and Extract
    URL : https://drive.google.com/file/d/1m8-EBPgi5MRubrm6iQjafK2QMHDBMSfJ/view?usp=sharing
    """

    def _check_integrity(self) -> bool:
        return True


class OxfordPets(Dataset):
    """
    URL = https://www.robots.ox.ac.uk/~vgg/data/pets/
    """

    def __init__(self,
                 data_path: str,
                 split: str,
                 transform: Callable,
                 **kwargs):
        self.data_dir = Path(data_path) / "OxfordPets"
        self.transforms = transform
        imgs = sorted([f for f in self.data_dir.iterdir() if f.suffix == '.jpg'])

        self.imgs = imgs[:int(len(imgs) * 0.75)] if split == "train" else imgs[int(len(imgs) * 0.75):]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = default_loader(self.imgs[idx])

        if self.transforms is not None:
            img = self.transforms(img)

        return img, 0.0  # dummy datat to prevent breaking


class VAEDataset(LightningDataModule):
    """
    PyTorch Lightning data module

    Args:
        data_dir: root directory of your dataset.
        train_batch_size: the batch size to use during training.
        val_batch_size: the batch size to use during validation.
        patch_size: the size of the crop to take from the original images.
        num_workers: the number of parallel workers to create to load data
            items (see PyTorch's Dataloader documentation for more details).
        pin_memory: whether prepared items should be loaded into pinned memory
            or not. This can improve performance on GPUs.
    """

    def __init__(
            self,
            data_path: str,
            train_batch_size: int = 8,
            val_batch_size: int = 8,
            patch_size: Union[int, Sequence[int]] = (256, 256),
            num_workers: int = 0,
            pin_memory: bool = False,
            **kwargs,
    ):
        super().__init__()

        self.data_dir = data_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        #       =========================  OxfordPets Dataset  =========================

        #         train_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
        #                                               transforms.CenterCrop(self.patch_size),
        # #                                               transforms.Resize(self.patch_size),
        #                                               transforms.ToTensor(),
        #                                                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

        #         val_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
        #                                             transforms.CenterCrop(self.patch_size),
        # #                                             transforms.Resize(self.patch_size),
        #                                             transforms.ToTensor(),
        #                                               transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

        #         self.train_dataset = OxfordPets(
        #             self.data_dir,
        #             split='train',
        #             transform=train_transforms,
        #         )

        #         self.val_dataset = OxfordPets(
        #             self.data_dir,
        #             split='val',
        #             transform=val_transforms,
        #         )

        #       =========================  CelebA Dataset  =========================

        train_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                               transforms.CenterCrop(148),
                                               transforms.Resize(self.patch_size),
                                               transforms.ToTensor(), ])

        val_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                             transforms.CenterCrop(148),
                                             transforms.Resize(self.patch_size),
                                             transforms.ToTensor(), ])

        self.train_dataset = MyCelebA(
            self.data_dir,
            split='train',
            transform=train_transforms,
            download=False,
        )

        # Replace CelebA with your dataset
        self.val_dataset = MyCelebA(
            self.data_dir,
            split='test',
            transform=val_transforms,
            download=False,
        )

    #       ===============================================================

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=144,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )


class VDEDataset(LightningDataModule):
    """
    PyTorch Lightning data module

    Args:
        data_dir: root directory of your dataset.
        train_batch_size: the batch size to use during training.
        val_batch_size: the batch size to use during validation.
        patch_size: the size of the crop to take from the original images.
        num_workers: the number of parallel workers to create to load data
            items (see PyTorch's Dataloader documentation for more details).
        pin_memory: whether prepared items should be loaded into pinned memory
            or not. This can improve performance on GPUs.
    """

    def __init__(
            self,
            args,
            train_batch_size: int = 8,
            val_batch_size: int = 8,
            num_workers: int = 28,
            pin_memory: bool = False,
            **kwargs,
    ):
        super().__init__()
        self.args = args
        self.train_batch_size = args.batch_size
        self.val_batch_size = args.batch_size
        self.test_batch_size = args.batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def prepare_data(self) -> None:
        if self.args.suffix == "springs":
            self.train_dataset, self.val_dataset, self.test_dataset, self.edges = load_customized_springs_data(self.args)
        elif self.args.suffix == "netsims":
            self.train_dataset, self.val_dataset, self.test_dataset, self.edges = load_customized_netsims_data(self.args)
        elif self.args.suffix in ['LI', 'LL', 'CY', 'BF', 'TF', 'BF-CV']:
            self.train_dataset, self.val_dataset, self.test_dataset, self.edges = load_data_biological(self.args)
        else:
            raise ValueError("Dataset not implemented yet.")

    def setup(self, stage: Optional[str] = None) -> None:
        #       =========================  OxfordPets Dataset  =========================

        #         train_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
        #                                               transforms.CenterCrop(self.patch_size),
        # #                                               transforms.Resize(self.patch_size),
        #                                               transforms.ToTensor(),
        #                                                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

        #         val_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
        #                                             transforms.CenterCrop(self.patch_size),
        # #                                             transforms.Resize(self.patch_size),
        #                                             transforms.ToTensor(),
        #                                               transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

        #         self.train_dataset = OxfordPets(
        #             self.data_dir,
        #             split='train',
        #             transform=train_transforms,
        #         )

        #         self.val_dataset = OxfordPets(
        #             self.data_dir,
        #             split='val',
        #             transform=val_transforms,
        #         )

        pass

    #       ===============================================================

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )

    def get_edges(self):
        return self.edges