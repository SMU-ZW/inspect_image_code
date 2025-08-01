import torch
import pydicom
import numpy as np
import pandas as pd
import cv2
import h5py
import nibabel as nib

from ..constants import *
from torch.utils.data import Dataset
from pathlib import Path
import os
from ..utils import read_tar_dicom
import io
import pickle


class DatasetBase(Dataset):
    def __init__(self, cfg, split="train", transform=None):
        self.cfg = cfg
        self.transform = transform
        self.split = split
        self.hdf5_dataset = None
        self.failed_files = []

        path = "/projects/eclarson/stems/STEMC/EHR/INSPECT/IMAGE/inpsect/inspectamultimodaldatasetforpulmonaryembolismdiagnosisandprog-3/full/dict_slice_thickness.pkl"
        self.dict_slice_thickness = pickle.load(open(path, "rb"))

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def read_from_hdf5(self, key, hdf5_path, slice_idx=None):
        if self.hdf5_dataset is None:
            self.hdf5_dataset = h5py.File(hdf5_path, "r")

        if slice_idx is None:
            arr = self.hdf5_dataset[key][:]
        else:
            arr = self.hdf5_dataset[key][slice_idx]

        # df_dicom_headers["patient_datetime"] = df_dicom_headers.apply(
        #     lambda x: f"{x.PatientID}_{x.StudyTime}", axis=1
        # )

        # only add slice thickness to stanford data
        if "rsna" not in self.cfg.dataset.csv_path:
            thickness_ls = []
            for idx_th in range(arr.shape[0]):
                try:
                    thickness_ls.append(self.dict_slice_thickness[key] * idx_th)
                    # print(
                    #     key,
                    #     self.dict_slice_thickness[key],
                    #     arr.shape[0],
                    #     "=========have thickness info=============================",
                    # )
                except:
                    print(
                        key,
                        idx_th,
                        "=========no thickness info=============================",
                    )
                    thickness_ls.append(0)
            thickness_ls = np.array(thickness_ls)
            arr = np.concatenate([arr, thickness_ls[:, None]], axis=1)
        elif "rsna" in self.cfg.dataset.csv_path:
            thickness_ls = []
            for idx_th in range(arr.shape[0]):
                # try:
                #     thickness_ls.append(self.dict_slice_thickness[key] * idx_th)
                # except:
                #     thickness_ls.append(0)
                thickness_ls.append(0)
            thickness_ls = np.array(thickness_ls)
            arr = np.concatenate([arr, thickness_ls[:, None]], axis=1)

        return arr

    def read_dicom(self, file_path: str, resize_size=None, channels=None):
        """Legacy DICOM reader, kept for compatibility"""
        if resize_size is None:
            resize_size = self.cfg.dataset.transform.resize_size
        if channels is None:
            channels = self.cfg.dataset.transform.channels

        # read dicom
        if "rsna" in self.cfg.dataset.csv_path.lower():
            dcm = pydicom.dcmread(file_path)
        else:
            # print("#######")
            # print(self.cfg.dataset.csv_path)
            patient_id = str(file_path).split("/")[-1].split("_")[0]
            tar_content = read_tar_dicom(
                os.path.join(self.cfg.dataset.dicom_dir, patient_id + ".tar")
            )
            dcm = pydicom.dcmread(io.BytesIO(tar_content[file_path]))

        try:
            pixel_array = dcm.pixel_array
        except:
            print(file_path)
            if channels == "repeat":
                pixel_array = np.zeros((resize_size, resize_size))
            else:
                pixel_array = np.zeros((3, resize_size, resize_size))

        # rescale
        try:
            intercept = dcm.RescaleIntercept
            slope = dcm.RescaleSlope
        except:
            intercept = 0
            slope = 1

        pixel_array = pixel_array * slope + intercept

        # resize
        if resize_size != pixel_array.shape[-1]:
            pixel_array = cv2.resize(
                pixel_array, (resize_size, resize_size), interpolation=cv2.INTER_AREA
            )

        return pixel_array

#     def read_image(self, file_path: str, resize_size=None, channels=None):
#         """Read medical image in DICOM or NIfTI format"""
#         print("===============================================================================")
#         print(file_path)
#         if resize_size is None:
#             resize_size = self.cfg.dataset.transform.resize_size
#         if channels is None:
#             channels = self.cfg.dataset.transform.channels

#         # Read image based on format
#         if file_path.endswith('.nii.gz'):
#             # Read NIfTI
#             nifti_img = nib.load(file_path)
#             pixel_array = nifti_img.get_fdata()
#             # NIfTI doesn't use rescale slope/intercept like DICOM
#             # but we'll keep consistent array preprocessing
#             if len(pixel_array.shape) > 2:
#                 # Take first volume/timepoint if 4D
#                 pixel_array = pixel_array[:, :, 0] if len(pixel_array.shape) == 3 else pixel_array[:, :, 0, 0]
#         else:
#             # Read DICOM
#             if "rsna" in self.cfg.dataset.csv_path:
#                 dcm = pydicom.dcmread(file_path)
#             else:
#                 patient_id = file_path.split("/")[-1].split("_")[0]
#                 tar_content = read_tar_dicom(
#                     os.path.join(self.cfg.dataset.dicom_dir, patient_id + ".tar")
#                 )
#                 dcm = pydicom.dcmread(io.BytesIO(tar_content[file_path]))

#             try:
#                 pixel_array = dcm.pixel_array
#                 try:
#                     intercept = dcm.RescaleIntercept
#                     slope = dcm.RescaleSlope
#                 except:
#                     intercept = 0
#                     slope = 1
#                 pixel_array = pixel_array * slope + intercept
#             except:
#                 print(f"Error reading {file_path}")
#                 if channels == "repeat":
#                     pixel_array = np.zeros((resize_size, resize_size))
#                 else:
#                     pixel_array = np.zeros((3, resize_size, resize_size))

#         # Resize image
#         if resize_size != pixel_array.shape[-1]:
#             pixel_array = cv2.resize(
#                 pixel_array, (resize_size, resize_size), interpolation=cv2.INTER_AREA
#             )

#         return pixel_array
    
#     def read_image(self, file_path: str, resize_size=None, channels=None):
#         # print("===============================================================================")
#         # print(file_path)
#         if resize_size is None:
#             resize_size = self.cfg.dataset.transform.resize_size
#         if channels is None:
#             channels = self.cfg.dataset.transform.channels

#         # ===== 强制非 RSNA 用 NIfTI 读取 =====
#         if "rsna" not in self.cfg.dataset.csv_path:
#             # 自动加上 .nii.gz 后缀（如果没有）
#             if not file_path.endswith('.nii.gz'):
#                 file_path += '.nii.gz'

#             # 不论是否以 .nii.gz 结尾，都用 nibabel 加载
#             nifti_img = nib.load(file_path)
#             pixel_array = nifti_img.get_fdata()
#             if len(pixel_array.shape) > 2:
#                 pixel_array = pixel_array[:, :, 0] if len(pixel_array.shape) == 3 else pixel_array[:, :, 0, 0]
#         else:
#             # ===== 原始 RSNA DICOM 读取逻辑 =====
#             dcm = pydicom.dcmread(file_path)
#             try:
#                 pixel_array = dcm.pixel_array
#                 try:
#                     intercept = dcm.RescaleIntercept
#                     slope = dcm.RescaleSlope
#                 except:
#                     intercept = 0
#                     slope = 1
#                 pixel_array = pixel_array * slope + intercept
#             except:
#                 print(f"Error reading {file_path}")
#                 if channels == "repeat":
#                     pixel_array = np.zeros((resize_size, resize_size))
#                 else:
#                     pixel_array = np.zeros((3, resize_size, resize_size))

#         # ===== Resize =====
#         if resize_size != pixel_array.shape[-1]:
#             pixel_array = cv2.resize(
#                 pixel_array, (resize_size, resize_size), interpolation=cv2.INTER_AREA
#             )

#         return pixel_array
    
    def read_image(self, file_path: str, resize_size=None, channels=None):
        if resize_size is None:
            resize_size = self.cfg.dataset.transform.resize_size
        if channels is None:
            channels = self.cfg.dataset.transform.channels

        if "rsna" not in self.cfg.dataset.csv_path:
            if not file_path.endswith('.nii.gz'):
                file_path += '.nii.gz'
            try:
                nifti_img = nib.load(file_path)
                pixel_array = nifti_img.get_fdata()
                if len(pixel_array.shape) > 2:
                    pixel_array = pixel_array[:, :, 0] if len(pixel_array.shape) == 3 else pixel_array[:, :, 0, 0]
            except Exception as e:
                print(f"[ERROR] Failed to read NIfTI: {file_path} | {e}")
                if not hasattr(self, "failed_files"):
                    self.failed_files = []
                self.failed_files.append(file_path)
                pixel_array = np.zeros((resize_size, resize_size))  # 用空白图像跳过

        else:
            dcm = pydicom.dcmread(file_path)
            try:
                pixel_array = dcm.pixel_array
                try:
                    intercept = dcm.RescaleIntercept
                    slope = dcm.RescaleSlope
                except:
                    intercept = 0
                    slope = 1
                pixel_array = pixel_array * slope + intercept
            except:
                print(f"Error reading {file_path}")
                if channels == "repeat":
                    pixel_array = np.zeros((resize_size, resize_size))
                else:
                    pixel_array = np.zeros((3, resize_size, resize_size))

        if resize_size != pixel_array.shape[-1]:
            pixel_array = cv2.resize(pixel_array, (resize_size, resize_size), interpolation=cv2.INTER_AREA)

        return pixel_array





    def windowing(self, pixel_array: np.array, window_center: int, window_width: int):
        lower = window_center - window_width // 2
        upper = window_center + window_width // 2
        pixel_array = np.clip(pixel_array.copy(), lower, upper)
        pixel_array = (pixel_array - lower) / (upper - lower)

        return pixel_array

    def process_numpy(self, numpy_path, idx):
        slice_array = np.load(numpy_path)[idx]

        resize_size = self.cfg.dataset.transform.resize_size
        channels = self.cfg.dataset.transform.channels

        if resize_size != slice_array.shape[-1]:
            slice_array = cv2.resize(
                slice_array, (resize_size, resize_size), interpolation=cv2.INTER_AREA
            )

        # window
        if self.cfg.dataset.transform.channels == "repeat":
            ct_slice = self.windowing(
                slice_array, 400, 1000
            )  # use PE window by default
            # create 3 channels after converting to Tensor
            # using torch.repeat won't take up 3x memory
        else:
            ct_slice = [
                self.windowing(slice_array, -600, 1500),  # LUNG window
                self.windowing(slice_array, 400, 1000),  # PE window
                self.windowing(slice_array, 40, 400),  # MEDIASTINAL window
            ]
            ct_slice = np.stack(ct_slice)

        return ct_slice

    def process_slice(
        self,
        slice_info: pd.Series = None,
        dicom_dir: Path = None,
        slice_path: str = None,
    ):
        """process slice with windowing, resize and tranforms"""

        if slice_path is None:
            slice_path = dicom_dir / slice_info[INSTANCE_PATH_COL]
        slice_array = self.read_dicom(slice_path)

        # window
        if self.cfg.dataset.transform.channels == "repeat":
            ct_slice = self.windowing(
                slice_array, 400, 1000
            )  # use PE window by default
            # create 3 channels after converting to Tensor
            # using torch.repeat won't take up 3x memory
        else:
            ct_slice = [
                self.windowing(slice_array, -600, 1500),  # LUNG window
                self.windowing(slice_array, 400, 1000),  # PE window
                self.windowing(slice_array, 40, 400),  # MEDIASTINAL window
            ]
            ct_slice = np.stack(ct_slice)

        return ct_slice

    def fix_slice_number(self, df: pd.DataFrame):
        num_slices = min(self.cfg.dataset.num_slices, df.shape[0])
        if self.cfg.dataset.sample_strategy == "random":
            slice_idx = np.random.choice(
                np.arange(df.shape[0]), replace=False, size=num_slices
            )
            slice_idx = list(np.sort(slice_idx))
            df = df.iloc[slice_idx, :]
        elif self.cfg.dataset.sample_strategy == "fix":
            df = df.iloc[:num_slices, :]
        else:
            raise Exception("Sampling strategy either 'random' or 'fix'")
        return df

    def fix_series_slice_number(self, series):
        num_slices = min(self.cfg.dataset.num_slices, series.shape[0])
        if num_slices == self.cfg.dataset.num_slices:
            if self.cfg.dataset.sample_strategy == "random":
                slice_idx = np.random.choice(
                    np.arange(series.shape[0]), replace=False, size=num_slices
                )
                slice_idx = list(np.sort(slice_idx))
                features = series[slice_idx, :]
            elif self.cfg.dataset.sample_strategy == "fix":
                pad = int((series.shape[0] - num_slices) / 2)  # select middle slices
                start = pad
                end = pad + num_slices
                features = series[start:end, :]
            else:
                raise Exception("Sampling strategy either 'random' or 'fix'")
            mask = np.ones(num_slices)
        else:
            mask = np.zeros(self.cfg.dataset.num_slices)
            mask[:num_slices] = 1
            shape = [self.cfg.dataset.num_slices] + list(series.shape[1:])
            features = np.zeros(shape)

            features[:num_slices] = series

        return features, mask

    def fill_series_to_num_slicess(self, series, num_slices):
        x = torch.zeros(()).new_full((num_slices, *series.shape[1:]), 0.0)
        x[: series.shape[0]] = series
        return x
