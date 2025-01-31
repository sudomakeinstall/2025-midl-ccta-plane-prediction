# System
import pathlib as pl
import csv

# Third Party
import pydantic as pc
import tomlkit as tk
import torch as t
import monai as mn
import numpy as np
import roma as rm

# Internal
from . import utils
from . import geometry


class EulerAngles(pc.BaseModel):
    angles: list[float]
    axes: str
    name: str
    augment: bool

    def quat(self, aug=None, shortest=True):
        q = rm.euler_to_unitquat(self.axes, self.angles)
        if self.augment and aug is not None:
            q = rm.quat_product(aug, q)
        if shortest:
            q = geometry.quat_to_shortest(q)
        return q


class EulerAnglesList(pc.BaseModel):
    angles_list: list[EulerAngles]

    def quats(self, aug=None):
        return t.stack([a.quat(aug) for a in self.angles_list])

    def names(self):
        return [i.name for i in self.angles_list]


def load_euler_angles(path: pl.Path) -> EulerAnglesList:
    with open(path, mode="rt", encoding="utf-8") as fp:
        angles_dict = tk.load(fp)
    try:
        angles = EulerAnglesList(**angles_dict)
    except pc.ValidationError as e:
        print(e.errors())
    return angles


def read_euler_angles(path: pl.Path, return_tensor=True):
    with open(path, newline="") as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)
        row = next(reader, None)
        row = [float(r) for r in row]
    if return_tensor:
        return t.Tensor(row)
    else:
        return row


class BasicDataset(t.utils.data.Dataset):
    def __init__(self, exp: utils.Experiment):
        """
        Initialize a dataset from an experiment class instantiation.

        Args:
            exp: Experiment instance.
        """
        self.exp = exp
        series_trn = list(sorted(self.exp.dir_trn.glob("*/")))
        series_val = list(sorted(self.exp.dir_val.glob("*/")))
        series_tst = list(sorted(self.exp.dir_tst.glob("*/")))
        series = series_trn + series_val + series_tst

        self.series_partition = (
            [0] * len(series_trn) + [1] * len(series_val) + [2] * len(series_tst)
        )
        self.series_partition = np.array(self.series_partition)
        self.indices = np.array(range(len(series)))

        self.img_paths = [s / self.exp.img_file for s in series]
        self.seg_paths = [s / self.exp.seg_file for s in series]
        if hasattr(self.exp, "ang_file"):
            self.ang_paths = [s / self.exp.ang_file for s in series]
        self.run_checks()

        self.augmentation = False

        self.setup_transforms()

    def setup_transforms(self):
        self.load = mn.transforms.LoadImaged(
            keys=("img", "seg"),
            image_only=True,
            ensure_channel_first=True,
            simple_keys=True,
        )
        self.orient = mn.transforms.Orientationd(
            keys=("img", "seg"),
            axcodes=self.exp.orientation,
        )
        self.remap = mn.transforms.MapLabelValued(
            keys=("seg"),
            orig_labels=self.exp.label_map.keys(),
            target_labels=self.exp.label_map.values(),
        )
        self.onehot = mn.transforms.AsDiscreted(
            keys=("seg"),
            to_onehot=max(self.exp.label_map.values()) + 1,
        )
        self.space = mn.transforms.Spacingd(
            keys=("img", "seg"),
            pixdim=(self.exp.unet_fine.spacing for x in range(self.exp.dim)),
            mode="bilinear",
        )
        self.crop = mn.transforms.ResizeWithPadOrCropd(
            keys=("img", "seg"),
            spatial_size=self.exp.lg_fov_hi_res_shape,
        )
        self.argmax = mn.transforms.AsDiscreted(
            keys=("seg"),
            argmax=True,
        )
        self.normalize = mn.transforms.ScaleIntensityRanged(
            keys=("img"),
            a_min=self.exp.clip_min,
            a_max=self.exp.clip_max,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        )
        self.affine_layer = mn.networks.layers.AffineTransform(
            zero_centered=True,
        )
        self.noise = mn.transforms.RandGaussianNoised(
            keys=("img"),
            prob=self.exp.augmentation_noise_prob,
            std=100.0,
            mean=0.0,
            sample_std=True,
        )

    def run_pipeline(self, data_dict):
        if not self.augmentation:
            pipeline = mn.transforms.Compose(
                [
                    self.load,
                    self.orient,
                    self.remap,
                    self.onehot,
                    self.space,
                    self.crop,
                    self.argmax,
                    self.onehot,
                    self.normalize,
                ]
            )
            return pipeline(data_dict)
        else:
            if np.random.uniform() > self.exp.augmentation_rotation_prob:
                data_dict["Q_aug"] = rm.identity_quat()
                data_dict["Q_aug_inv"] = rm.identity_quat()
            else:
                # data_dict["Q_aug"] = rm.random_unitquat()
                data_dict["Q_aug"] = geometry.random_unitquat()
                data_dict["Q_aug"] = geometry.quat_to_shortest(data_dict["Q_aug"])
                data_dict["Q_aug_inv"] = rm.quat_conjugation(data_dict["Q_aug"])
            T_aug = rm.RigidUnitQuat(
                linear=data_dict["Q_aug"],
                translation=t.zeros((self.exp.dim)),
            )
            affine = mn.transforms.Lambdad(
                keys=("img", "seg"),
                func=lambda x: self.affine_layer(
                    x.unsqueeze(0), T_aug.to_homogeneous()
                ).squeeze(0),
            )
            pipeline = mn.transforms.Compose(
                [
                    self.load,
                    self.orient,
                    self.remap,
                    self.onehot,
                    self.space,
                    affine,
                    self.crop,
                    self.argmax,
                    self.onehot,
                    self.noise,
                    self.normalize,
                ]
            )
            return pipeline(data_dict)

    def get_trn_indices(self):
        """Get the indices corresponding to the training set."""
        return self.indices[self.series_partition == 0]

    def get_val_indices(self):
        """Get the indices corresponding to the validation set."""
        return self.indices[self.series_partition == 1]

    def get_tst_indices(self):
        """Get the indices corresponding to the testing set."""
        return self.indices[self.series_partition == 2]

    def run_checks(self):
        """
        Ensure that all paths correspond to an existing file, and that there are an
        equal number of images and segmentations.
        """
        for i in self.img_paths:
            assert i.exists(), f"The file {i} doesn't exist."
        for s in self.seg_paths:
            assert s.exists(), f"The file {s} doesn't exist."
        assert len(self.img_paths) == len(
            self.seg_paths
        ), "The number of image and segmentation paths must be equal."
        if hasattr(self, "ang_paths"):
            for a in self.ang_paths:
                assert a.exists(), f"The file {a} doesn't exist."
                angles = load_euler_angles(a)
                ang_lambda = lambda x: x.name in self.exp.quaternion_names
                angles.angles_list = list(filter(ang_lambda, angles.angles_list))
                for m, e in zip(angles.names(), self.exp.quaternion_names):
                    assert m == e
            assert len(self.img_paths) == len(
                self.ang_paths
            ), "The number of image and angle paths must be equal."

    def __len__(self) -> int:
        """
        Return the total number of datapoints.
        """
        return len(self.img_paths)

    def __getitem__(self, index: int):
        """Get a dictionary containing the image and segmentation for a given index.

        Args:
            index: Integer index to retrieve.
        """

        return_dict = dict()
        return_dict["index"] = index
        return_dict["img_path"] = self.img_paths[index]
        return_dict["img"] = self.img_paths[index]
        return_dict["seg_path"] = self.seg_paths[index]
        return_dict["seg"] = self.seg_paths[index]

        return_dict = self.run_pipeline(return_dict)

        if hasattr(self, "ang_paths"):
            return_dict["ang_path"] = str(self.ang_paths[index])
            angles = load_euler_angles(self.ang_paths[index])
            ang_lambda = lambda x: x.name in self.exp.quaternion_names
            angles.angles_list = list(filter(ang_lambda, angles.angles_list))
            if self.augmentation:
                return_dict["Q_all"] = angles.quats(return_dict["Q_aug_inv"])
            else:
                return_dict["Q_all"] = angles.quats()

        for k in ["img_path", "seg_path"]:
            return_dict[k] = str(return_dict[k])
        return return_dict


