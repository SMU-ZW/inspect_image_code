import torch
import numpy as np
import pandas as pd
import cv2
import h5py

from ..constants import *
from .dataset_base import DatasetBase
from omegaconf import OmegaConf
from PIL import Image
from pathlib import Path
import os
import pickle


class Dataset1D(DatasetBase):
    def __init__(self, cfg, split="train", transform=None):
        super().__init__(cfg, split)

        self.cfg = cfg
        self.df = pd.read_csv(cfg.dataset.csv_path, sep='\t')
        # print("Columns in df:", self.df.columns.tolist())

        if "rsna" not in cfg.dataset.csv_path:
            # Use image_id directly from metadata and create impression_id
            self.df["patient_datetime"] = self.df["image_id"].str.replace(".nii.gz", "")
            # self.df["patient_datetime"] = self.df["image_id"]
            # Set impression_id to be the same as patient_datetime if not present
            if 'impression_id' not in self.df.columns:
                self.df['impression_id'] = self.df["patient_datetime"]

            # duplicate patient_datetime remove
            self.df = self.df.drop_duplicates(subset=["patient_datetime"])

            # Read labels if provided separately
            if hasattr(cfg.dataset, 'label_csv') and cfg.dataset.label_csv:
                label_df = pd.read_csv(cfg.dataset.label_csv, sep='\t')
                self.df = self.df.merge(label_df, on='impression_id', how='left')

            # Read splits if provided separately
            if hasattr(cfg.dataset, 'split_csv') and cfg.dataset.split_csv:
                split_df = pd.read_csv(cfg.dataset.split_csv, sep='\t')
                self.df = self.df.merge(split_df, on='impression_id', how='left')

            if split != "all" and 'split' in self.df.columns:
                self.df = self.df[self.df["split"] == split]

        if split == "test":
            self.cfg.dataset.sample_strategy = "fix"

        # hdf5 path
        self.hdf5_path = self.cfg.dataset.hdf5_path

        if self.hdf5_path is None:
            raise Exception("Encoded slice HDF5 required")

        if "rsna" not in cfg.dataset.csv_path:
            # before = len(self.df)
            self.df = self.df[~self.df[cfg.dataset.target].isin(["Censored", "Censor"])]
            # after = len(self.df)
            # print(f"🚫 Removed {before - after} censored samples")
            self.study = self.df["patient_datetime"].tolist()
        else:
            self.study = self.df["SeriesInstanceUID"].tolist()

        self.df[cfg.dataset.target] = self.df[cfg.dataset.target].astype(str).str.lower()
        self.labels = [1 if t == "true" else 0 for t in self.df[cfg.dataset.target]]
        print(
            f"Pos: {len([t for t in self.labels if t == 1])} ; Neg: {len([t for t in self.labels if t == 0])}"
        )

    def __getitem__(self, index):
        # read featurized series
        study = self.study[index]
        # Use the image_id (without .nii.gz) as the key
        key = study.replace(".nii.gz", "")
        x = self.read_from_hdf5(key, hdf5_path=self.hdf5_path)

        # fix number of slices
        x, mask = self.fix_series_slice_number(x)

        # contextualize slices
        if self.cfg.dataset.contextualize_slice:
            x = self.contextualize_slice(x)

        # create torch tensor
        x = torch.from_numpy(x).float()

        mask = torch.tensor(mask).float()

        # get traget
        y = [self.labels[index]]
        # y = self.pe_labels[index]
        y = torch.tensor(y).float()

        return x, y, mask, study

    def __len__(self):
        return len(self.study)

    def contextualize_slice(self, arr):
        # make new empty array
        new_arr = np.zeros((arr.shape[0], arr.shape[1] * 3), dtype=np.float32)

        # fill first third of new array with original features
        for i in range(len(arr)):
            new_arr[i, : arr.shape[1]] = arr[i]

        # difference between previous neighbor
        new_arr[1:, arr.shape[1] : arr.shape[1] * 2] = (
            new_arr[1:, : arr.shape[1]] - new_arr[:-1, : arr.shape[1]]
        )

        # difference between next neighbor
        new_arr[:-1, arr.shape[1] * 2 :] = (
            new_arr[:-1, : arr.shape[1]] - new_arr[1:, : arr.shape[1]]
        )

        return new_arr

    def get_sampler(self):
        neg_class_count = (np.array(self.labels) == 0).sum()
        pos_class_count = (np.array(self.labels) == 1).sum()
        class_weight = [1 / neg_class_count, 1 / pos_class_count]
        weights = [class_weight[i] for i in self.labels]

        weights = torch.Tensor(weights).double()
        sampler = torch.utils.data.sampler.WeightedRandomSampler(
            weights, num_samples=len(weights), replacement=True
        )

        return sampler


class RSNADataset1D(DatasetBase):
    def __init__(self, cfg, split="test", transform=None):
        super().__init__(cfg, split)

        self.cfg = cfg
        self.df = pd.read_csv(cfg.dataset.csv_path)
        if "rsna" not in cfg.dataset.csv_path:
            self.df["patient_datetime"] = self.df.apply(
                lambda x: f"{x.patient_id}_{x.procedure_time}", axis=1
            )
            # duplicate patient_datetime remove
            self.df = self.df.drop_duplicates(subset=["patient_datetime"])

            if split != "all":
                self.df = self.df[self.df["split"] == split]
        elif "rsna" in cfg.dataset.csv_path:
            if split == "test":
                path = "/share/pi/nigam/projects/zphuo/data/PE/inspect/image_modality/anon_pe_features/rsna_hdf5_keys_testsplit.pkl"
                with open(path, "rb") as f:
                    keys = pickle.load(f)
                self.df = self.df[self.df["SeriesInstanceUID"].isin(keys)]

            elif split != "all":
                self.df = self.df[self.df["Split"] == split]

        if split == "test":
            self.cfg.dataset.sample_strategy = "fix"

        # hdf5 path
        model_type = self.cfg.dataset.pretrain_args.model_type
        input_size = self.cfg.dataset.pretrain_args.input_size
        channel_type = self.cfg.dataset.pretrain_args.channel_type
        # self.hdf5_path = os.path.join(
        #     self.cfg.exp.base_dir,
        #     f"{model_type}_{input_size}_{channel_type}_features/"
        #     + f"{model_type}_{input_size}_{channel_type}_features.hdf5",
        # )
        # self.hdf5_path = "/share/pi/nigam/projects/zphuo/data/PE/inspect/image_modality/anon_pe_features_full/features.hdf5"
        self.hdf5_path = self.cfg.dataset.hdf5_path

        # self.cfg.dataset.hdf5_path
        if self.hdf5_path is None:
            raise Exception("Encoded slice HDF5 required")

        if "rsna" not in cfg.dataset.csv_path:
            self.df = self.df[~self.df[cfg.dataset.target].isin(["Censored", "Censor"])]

            self.study = (
                self.df["patient_datetime"]
                .apply(lambda x: x.replace("T", " "))
                .tolist()
            )
        else:
            self.study = self.df["SeriesInstanceUID"].tolist()

        self.df[cfg.dataset.target] = self.df[cfg.dataset.target].astype(str)
        self.labels = [1 if t == "1" else 0 for t in self.df[cfg.dataset.target]]
        print(
            f"Pos: {len([t for t in self.labels if t == 1])} ; Neg: {len([t for t in self.labels if t == 0])}"
        )

    def __getitem__(self, index):
        # read featurized series
        study = self.study[index]
        x = self.read_from_hdf5(study, hdf5_path=self.hdf5_path)

        # fix number of slices
        x, mask = self.fix_series_slice_number(x)

        # contextualize slices
        if self.cfg.dataset.contextualize_slice:
            x = self.contextualize_slice(x)

        # create torch tensor
        x = torch.from_numpy(x).float()

        mask = torch.tensor(mask).float()

        # get traget
        y = [self.labels[index]]
        # y = self.pe_labels[index]
        y = torch.tensor(y).float()

        return x, y, mask, study

    def __len__(self):
        return len(self.study)

    def contextualize_slice(self, arr):
        # make new empty array
        new_arr = np.zeros((arr.shape[0], arr.shape[1] * 3), dtype=np.float32)

        # fill first third of new array with original features
        for i in range(len(arr)):
            new_arr[i, : arr.shape[1]] = arr[i]

        # difference between previous neighbor
        new_arr[1:, arr.shape[1] : arr.shape[1] * 2] = (
            new_arr[1:, : arr.shape[1]] - new_arr[:-1, : arr.shape[1]]
        )

        # difference between next neighbor
        new_arr[:-1, arr.shape[1] * 2 :] = (
            new_arr[:-1, : arr.shape[1]] - new_arr[1:, : arr.shape[1]]
        )

        return new_arr

    def get_sampler(self):
        neg_class_count = (np.array(self.labels) == 0).sum()
        pos_class_count = (np.array(self.labels) == 1).sum()
        class_weight = [1 / neg_class_count, 1 / pos_class_count]
        weights = [class_weight[i] for i in self.labels]

        weights = torch.Tensor(weights).double()
        sampler = torch.utils.data.sampler.WeightedRandomSampler(
            weights, num_samples=len(weights), replacement=True
        )

        return sampler
