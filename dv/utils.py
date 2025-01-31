# System
import pathlib as pl
import functools as ft

# Third Party
import numpy as np
import pydantic as pc
import torch as t
import tomlkit as tk
import monai as mn
import roma as rm

# Internal
from . import geometry
from . import enums


@pc.dataclasses.dataclass(config=dict(frozen=True))
class UNetParams:
    """Parameters for defining a UNet model."""

    spacing: float
    shape: tuple[int, int, int]
    features: list[int]
    frozen: bool
    attention: bool
    residual: bool

    @pc.computed_field
    @ft.cached_property
    def fc_length(self) -> int:
        return (
            np.prod([s // 2 ** len(self.features) for s in self.shape])
            * self.features[-1]
            * 2
        )


@pc.dataclasses.dataclass(config=dict(frozen=True))
class LossMetric:
    weight: pc.NonNegativeFloat
    should_eval: bool

    @pc.computed_field
    @ft.cached_property
    def should_calc(self) -> bool:
        return self.should_eval or self.weight > 0.0


@pc.dataclasses.dataclass(config=dict(frozen=True))
class QuaternionComposition:
    name: str
    indices: list[int]
    mse: LossMetric
    geodesic: LossMetric
    angle: LossMetric


@pc.dataclasses.dataclass(config=dict(frozen=True))
class Label:
    """Segmentation label."""

    o: int
    rgba: tuple[float, float, float, float]
    text: str
    legend: bool

    @pc.computed_field
    @ft.cached_property
    def rgba_str(self) -> str:
        return "rgba(" + ", ".join([str(c) for c in self.rgba]) + ")"

    @pc.computed_field
    @ft.cached_property
    def rgba_255_str(self) -> str:
        return "rgba(" + ", ".join([str(c * 255.0) for c in self.rgba]) + ")"


@pc.dataclasses.dataclass(config=dict(arbitrary_types_allowed=True, frozen=True))
class Experiment:
    """Experiment file, instantiated from a TOML file path."""

    dir_trn: pl.Path
    dir_val: pl.Path
    dir_tst: pl.Path
    dir_output: pl.Path
    exp_name: str
    img_file: pl.Path
    seg_file: pl.Path
    ang_file: pl.Path | None
    quaternion_names: list[str] | None
    checkpoint_pretrain_file_name: pl.Path
    checkpoint_last_file_name: pl.Path
    checkpoint_best_file_name: pl.Path
    dataframe_file_name: pl.Path
    device: str
    num_workers: pc.NonNegativeInt
    pin_memory: bool
    seed: int
    epochs: pc.PositiveInt
    channels: pc.PositiveInt
    classes: pc.PositiveInt
    orientation: str
    batch_size: pc.PositiveInt
    air_hu: int
    clip_min: int
    clip_max: int
    augmentation: bool
    augmentation_noise_prob: pc.confloat(ge=0.0, le=1.0)
    augmentation_rotation_prob: pc.confloat(ge=0.0, le=1.0)
    normalization: enums.NormalizationEnum
    nonlinearity: enums.NonlinearityEnum
    pass_features: bool
    transform: enums.TransformEnum
    hidden_layers: list[pc.PositiveInt]
    fc_frozen: bool
    stop_after_coarse: bool
    stop_after_fc: bool
    c2f_key: str
    unet_coarse: UNetParams
    unet_fine: UNetParams
    quat_compositions: list[QuaternionComposition]
    labels: pc.conlist(Label, min_length=2)
    label_map_data: list[tuple[int, int]]
    coarse_dc: LossMetric
    coarse_jd: LossMetric
    coarse_hf: LossMetric
    coarse_cd: LossMetric
    fine_dc: LossMetric
    fine_jd: LossMetric
    fine_hf: LossMetric
    checkpoint_best_metric: str
    lr_strategy: str
    lr_initial: pc.NonNegativeFloat
    lr_warmup: bool
    lr_warmup_start_factor: pc.confloat(ge=0.0, le=1.0) | None = None
    lr_warmup_total_iters: pc.PositiveInt | None = None
    lr_exponential_gamma: pc.NonNegativeFloat | None = None
    lr_step_size: int | None = None
    lr_onecycle_max: pc.NonNegativeFloat | None = None
    clip_grad_norm: pc.NonNegativeFloat | None = None

    @pc.computed_field
    @ft.cached_property
    def fc_features(self) -> list[int]:
        return (
            [self.unet_coarse.fc_length]
            + self.hidden_layers
            + [len(self.quaternion_names) * 4]
        )

    @pc.computed_field
    @ft.cached_property
    def checkpoint_pretrain_path(self) -> pl.Path:
        """Get the path to the pretrained model."""
        return self.dir_output / self.checkpoint_pretrain_file_name

    @pc.computed_field
    @ft.cached_property
    def checkpoint_last_path(self) -> pl.Path:
        """Get the path to the last saved model."""
        return self.dir_output / self.checkpoint_last_file_name

    @pc.computed_field
    @ft.cached_property
    def checkpoint_best_path(self) -> pl.Path:
        """Get the path to the beast saved model."""
        return self.dir_output / self.checkpoint_best_file_name

    @pc.computed_field
    @ft.cached_property
    def dataframe_path(self) -> pl.Path:
        """Get the path to the dataframe."""
        return self.dir_output / self.dataframe_file_name

    @pc.computed_field
    @ft.cached_property
    def dim(self) -> int:
        """Return the dimension of the input image."""
        dim_c = len(self.unet_coarse.shape)
        dim_f = len(self.unet_fine.shape)
        assert dim_c == dim_f
        assert dim_c in {2, 3}
        return dim_c

    @pc.computed_field
    @ft.cached_property
    def fine_channels(self) -> int:
        """Calculate the number of channels, depending on whether feature sharing is enabled."""
        if self.pass_features:
            return self.channels + self.classes
        else:
            return self.channels

    @pc.computed_field
    @ft.cached_property
    def new_axis(self) -> t.Tensor:
        """Get the new axis to which the points will be aligned."""
        return (
            t.tensor([[0, 0, 1]], device=self.device).repeat(self.batch_size, 1).half()
        )

    @pc.computed_field
    @ft.cached_property
    def identity_quat(self) -> t.Tensor:
        """Get the identity quaternion."""
        return rm.identity_quat(
            size=self.batch_size, dtype=t.float16, device=self.device
        )

    ###
    ### Label Helpers
    ###

    @pc.computed_field
    @ft.cached_property
    def label_map(self) -> dict[int, int]:
        return dict(self.label_map_data)

    @pc.computed_field
    @ft.cached_property
    def plotly_colorscale(self) -> list[tuple[float, str]]:
        n = len(self.labels)
        lt = lambda x: [x.o / n, x.rgba_255_str]
        rt = lambda x: [(x.o + 1.0) / n, x.rgba_255_str]
        scale = [b(l) for l in self.labels for b in (lt, rt)]
        return scale

    def get_label(self, val: int) -> Label:
        return list(filter(lambda x: x.o == val, self.labels))[0]

    ##
    ## Data Helpers
    ##

    @pc.computed_field
    @ft.cached_property
    def A_full_to_coarse(self) -> t.Tensor:
        return mn.transforms.utils.create_scale(
            self.dim,
            [
                self.unet_coarse.spacing / self.unet_fine.spacing
                for x in range(self.dim)
            ],
            device=self.device,
            backend="torch",
        ).float()

    @pc.computed_field
    @ft.cached_property
    def A_coarse_to_full(self) -> t.Tensor:
        return self.A_full_to_coarse.inverse()

    @pc.computed_field
    @ft.cached_property
    def coarse_fov(self) -> t.Tensor:
        return (
            t.tensor(self.unet_coarse.shape, device=self.device)
            * self.unet_coarse.spacing
        )

    @pc.computed_field
    @ft.cached_property
    def fine_fov(self) -> t.Tensor:
        return (
            t.tensor(self.unet_fine.shape, device=self.device) * self.unet_fine.spacing
        )

    @pc.computed_field
    @ft.cached_property
    def fov_ratio(self) -> t.Tensor:
        return self.fine_fov / self.coarse_fov

    @pc.computed_field
    @ft.cached_property
    def lg_fov_hi_res_shape(self) -> t.Tensor:
        """Calculate the input shape."""
        return (self.coarse_fov / self.unet_fine.spacing).long()

    def get_affine_grid(
        self, shape: t.Tensor, transform: t.Tensor | None = None
    ) -> t.Tensor:
        """Calculate an affine grid, optionally transformed by a given tensor.

        Args:
            shape: Shape of the affine grid.
            transform: Optional transformation to apply to the affine grid.
        """
        if transform is None:
            transform = geometry.affine_identity(self.dim, self.batch_size)
        return (
            t.nn.functional.affine_grid(
                transform,
                [self.batch_size, self.channels, *shape],
                align_corners=False,
            )
            .half()
            .to(self.device)
        )

    @pc.computed_field
    @ft.cached_property
    def lg_fov_lo_res_points(self) -> t.Tensor:
        """Calculate the points of the full FOV, low-resolution image."""
        return self.get_affine_grid(self.unet_coarse.shape)


def load_experiment(exp_path: pl.Path) -> Experiment:
    with open(exp_path, mode="rt", encoding="utf-8") as fp:
        exp_dict = tk.load(fp)
    try:
        exp = Experiment(**exp_dict)
    except pc.ValidationError as e:
        print(e.errors())
    return exp


def tex(macro: str, value):
    """Create a LaTeX macro.

    Args:
        macro: Name of the LaTeX macro.
        value: Value of the LaTeX macro.
    """
    return f"\\newcommand{{\\{macro}}}{{{value}}}"
