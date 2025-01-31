# Third Party
import torch as t
import numpy as np
import roma as rm
import monai as mn

# Internal
from . import enums
from . import layers
from . import geometry
from . import utils
from . import visualization


class CCTANet(t.nn.Module):
    """A neural network containing coarse segmentation, transformation, and fine segmentation modules."""

    def __init__(self, exp: utils.Experiment):
        """Initialization.

        Args:
            exp: Experiment instance.
        """
        super(CCTANet, self).__init__()
        self.exp = exp

        self.unet_coarse = UNet(
            self.exp.channels,
            self.exp.classes,
            self.exp.dim,
            self.exp.unet_coarse.features,
            self.exp.normalization,
            self.exp.nonlinearity,
            self.exp.unet_coarse.attention,
            self.exp.unet_coarse.residual,
        )

        self.affine_layer = mn.networks.layers.AffineTransform(
            zero_centered=True,
        )

        if self.exp.transform == enums.TransformEnum.rigid:
            self.build_fc()

        self.unet_fine = UNet(
            self.exp.fine_channels,
            self.exp.classes,
            self.exp.dim,
            self.exp.unet_fine.features,
            self.exp.normalization,
            self.exp.nonlinearity,
            self.exp.unet_fine.attention,
            self.exp.unet_fine.residual,
        )

    def forward(self, x):
        data_dict = dict()

        ##
        ## Coarse
        ##

        img_coarse = self.affine_layer(
            x,
            self.exp.A_full_to_coarse,
            spatial_size=self.exp.unet_coarse.shape,
        )

        coarse = self.unet_coarse(img_coarse)
        data_dict["centroid"] = (
            self.calculate_centroid(coarse["seg"].softmax(1)[:, 1:, :].sum(1))
            * self.exp.coarse_fov
            / 2.0
        ).flip(1)
        data_dict["seg_coarse"] = coarse["seg"]
        # visualization.quick_show(img_coarse[0], coarse["seg"][0].softmax(0))
        # visualization.quick_show(coarse["seg"].softmax(1)[0,0:1])

        if self.exp.stop_after_coarse:
            return data_dict

        ##
        ## FC
        ##

        match self.exp.transform:
            case enums.TransformEnum.rigid:
                data_dict["Q_all"] = self.fc(
                    t.flatten(coarse["hook"], start_dim=1)
                ).reshape(self.exp.batch_size, len(self.exp.quaternion_names), 4)
                for qc in self.exp.quat_compositions:
                    kname = f"Q_{qc.name}"
                    if len(qc.indices) > 1:
                        data_dict[kname] = rm.quat_composition(
                            [data_dict["Q_all"][:, i, ...] for i in qc.indices]
                        )
                    else:
                        data_dict[kname] = data_dict["Q_all"][:, qc.indices[0], ...]
            case enums.TransformEnum.pca:
                assert False
                transform = geometry.transform_from_seg(
                    coarse["seg"].softmax(1)[:, 1:, :].sum(1),
                    self.exp.new_axis,
                )
            case enums.TransformEnum.orthogonal:
                assert False
                transform = dict()
                transform["centroid"] = self.calculate_centroid(
                    coarse["seg"].softmax(1)[:, 1:, :].sum(1)
                )
                transform["quaternion"] = self.exp.identity_quat.to(self.exp.device)
            case _:
                assert False, f"Transformation {self.exp.transform} not recognized."

        data_dict["T_c2f"] = rm.RigidUnitQuat(
            linear=data_dict[self.exp.c2f_key], translation=data_dict["centroid"]
        )
        data_dict["T_c2f_hom"] = data_dict["T_c2f"].normalize().to_homogeneous()

        if self.exp.stop_after_fc:
            return data_dict

        ##
        ## Fine
        ##

        img_fine = self.affine_layer(
            x,
            data_dict["T_c2f_hom"],
            spatial_size=self.exp.unet_fine.shape,
        )

        if self.exp.pass_features:
            features_fine = self.affine_layer(
                coarse["seg"],
                (self.exp.A_coarse_to_full @ data_dict["T_c2f_hom"]).float(),
                spatial_size=self.exp.unet_fine.shape,
            )
            # visualization.quick_show(img_fine[0], features_fine[0])
            fine = self.unet_fine(t.cat((features_fine, img_fine), dim=1))
            # visualization.quick_show(img_fine[0], fine["seg"][0])
        else:
            fine = self.unet_fine(img_fine)

        data_dict["img_fine"] = img_fine
        data_dict["seg_fine"] = fine["seg"]

        return data_dict

    def build_fc(self):
        self.fc = t.nn.Sequential()
        for i in range(len(self.exp.fc_features) - 1):
            self.fc.append(
                t.nn.Linear(self.exp.fc_features[i], self.exp.fc_features[i + 1])
            )
            if i < len(self.exp.fc_features) - 2:
                self.fc.append(t.nn.LeakyReLU(inplace=True))

    def transform_from_seg(self, seg: t.Tensor, e: t.Tensor):
        """Determine the transformation required to align the major axis of `seg` with the provided tensor `e`.

        Args:
            seg: Segmentation.
            e: Vector to align with.

        Returns:
            A dictionary containing the rotating quaternion ("quaternion"), the major axis of the provided points ("axis"), the angle between the major axis and the provided `newaxis` ("angle"), and the centroid of the points ("centroid").

        Examples:
            >>> seg = t.zeros((2, 3, 3, 3))
            >>> seg[:,:,1,1] = 1
            >>> e = t.Tensor([1, 0, 0]).unsqueeze(0).repeat(2, 1)
            >>> t = transform_from_seg(seg, e)
            >>> t["quaternion"]
            tensor([[-0.0000, 0.7070, 0.0000, 0.7075],
                    [-0.0000, 0.7070, 0.0000, 0.7075]], dtype=torch.float16)
        """
        assert seg.dim() == 4
        I = geometry.affine_identity(
            dim=self.exp.dim, batch_size=self.exp.batch_size
        ).to(self.exp.device)

        # Note that the result of `affine_grid` is somewhat counterintuitive.
        # The first dimension is the batch, the last dimension is the spatial
        # coordinate, and the middle dimensions are the image axes. Though the
        # image axes are ordered as
        # [Caudal -> Cranial] x [Posterior -> Anterior] x [Right -> Left]
        # the ordering of the coordinates in the final vector are reversed. More
        # concretely, the following will all yield constant images:
        #
        # `points_all[0,0,:,:,2]`
        # `points_all[0,:,0,:,1]`
        # `points_all[0,:,:,0,0]`
        #
        # Of course, this ording must be accounted for when interpreting the resulting
        # centroid and rotation.

        centroid, pts_flat, wts_flat = self.calculate_centroid(seg, return_all=True)
        pts_flat = pts_flat - centroid.unsqueeze(1)
        pts_weighted = pts_flat * wts_flat**0.5

        return_dict = geometry.normalizing_quaternion_from_points(pts_weighted, e)
        return_dict["centroid"] = centroid

        return return_dict

    def calculate_centroid(self, seg: t.Tensor, return_all=False):
        # https://stats.stackexchange.com/questions/113485/weighted-principal-components-analysis
        pts_flat = self.exp.lg_fov_lo_res_points.flatten(start_dim=1, end_dim=-2)
        wts_flat = seg.flatten(start_dim=1).unsqueeze(-1)
        centroid = (pts_flat * wts_flat / wts_flat.sum(axis=1).unsqueeze(-1)).sum(1)
        if return_all:
            return centroid, pts_flat, wts_flat
        else:
            return centroid

    def coarse_requires_grad(self, requires_grad: bool):
        """Method for freezing the coarse module.

        Args:
            requires_grad: If `False`, freeze the coarse module.
        """
        self.unet_coarse.set_requires_grad(requires_grad)

    def fine_requires_grad(self, requires_grad: bool):
        """Method for freezing the fine module.

        Args:
            requires_grad: If `False`, freeze the fine module.
        """
        self.unet_fine.set_requires_grad(requires_grad)

    def fc_requires_grad(self, requires_grad: bool):
        """Method for freezing the fully connected layer.

        Args:
            requires_grad: If `False`, freeze the fully connected layer.
        """
        for p in self.fc.parameters():
            p.requires_grad = requires_grad


class UNet(t.nn.Module):
    """A basic UNet implementation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dimension: int,
        features: list[int],
        norm: enums.NormalizationEnum,
        nonlinearity: enums.NonlinearityEnum = enums.NonlinearityEnum.relu,
        attention: bool = False,
        residual: bool = False,
    ):
        """Initialization.

        Args:
            in_channels: Number of channels in the input.
            out_channels: Number of channels in the output.
            dimension: Dimensionality of the input.
            features: List of feature lengths in each UNet layer.
            norm: Type of normalization.
        """
        super(UNet, self).__init__()
        self.attention = attention
        if self.attention:
            self.attention_modules = t.nn.ModuleList()
            for feature in reversed(features):
                self.attention_modules.append(
                    layers.Attn(feature, feature, feature, dimension)
                )
        self.dn_path = t.nn.ModuleList()
        self.up_path = t.nn.ModuleList()
        self.pool = layers.max_pool_dict[dimension](kernel_size=2, stride=2)

        # Downsampling path
        for ipt, opt in zip([in_channels] + features, features):
            self.dn_path.append(
                layers.CvNmAcxN(
                    ipt,
                    opt,
                    dimension,
                    norm=norm,
                    nonlinearity=nonlinearity,
                    residual=residual,
                )
            )

        # Upsampling path
        for feature in reversed(features):
            self.up_path.append(
                layers.conv_t_dict[dimension](
                    feature * 2, feature, kernel_size=2, stride=2
                )
            )
            self.up_path.append(
                layers.CvNmAcxN(
                    feature * 2,
                    feature,
                    dimension,
                    norm=norm,
                    nonlinearity=nonlinearity,
                    residual=residual,
                )
            )

        # Bottleneck
        self.bottleneck = layers.CvNmAcxN(
            features[-1],
            features[-1] * 2,
            dimension,
            norm=norm,
            nonlinearity=nonlinearity,
            residual=residual,
        )

        # Final
        self.final_conv = layers.conv_dict[dimension](
            features[0], out_channels, kernel_size=1
        )

        self.initialize_weights(dimension, nonlinearity)

    def initialize_weights(self, dim, nonlinearity):
        for m in self.modules():
            if isinstance(m, layers.conv_dict[dim]):
                t.nn.init.kaiming_uniform_(m.weight, nonlinearity=nonlinearity)
                if m.bias is not None:
                    t.nn.init.constant_(m.bias, 0)

    def forward(self, x):
        skip_connections = []
        for down in self.dn_path:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        hook = x.clone()
        skip_connections = skip_connections[::-1]
        for idx in range(0, len(self.up_path), 2):
            x = self.up_path[idx](x)
            skip_connection = skip_connections[idx // 2]

            if self.attention:
                skip_connection = self.attention_modules[idx // 2](x, skip_connection)

            concat_skip = t.cat((skip_connection, x), dim=1)
            x = self.up_path[idx + 1](concat_skip)

        return {"seg": self.final_conv(x), "hook": hook}

    def set_requires_grad(self, requires_grad: bool):
        for p in self.parameters():
            p.requires_grad = requires_grad


