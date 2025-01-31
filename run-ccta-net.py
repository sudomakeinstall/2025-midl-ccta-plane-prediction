#!/usr/bin/env python

# System
import argparse
import random
import pathlib as pl
import typing as ty

# Third Party
import torch as t
import tqdm
import numpy as np
import monai as mn
import roma as rm
import pydantic as pc
import pandas as pd

# Internal
import dv


class CCTANetRunner:
    """
    A class for CCTANet training and inference.
    """

    def __init__(self, exp_path: pl.Path):
        """Initialize the CCTANetRunner class.

        Args:
            exp_path: Path to the `toml` format experiment file.
        """
        self.exp_path = exp_path
        self.exp = dv.utils.load_experiment(exp_path)

        self.set_seeds()

        self.model = dv.models.CCTANet(self.exp).to(self.exp.device)
        self.model.coarse_requires_grad(not self.exp.unet_coarse.frozen)
        self.model.fc_requires_grad(not self.exp.fc_frozen)
        self.model.fine_requires_grad(not self.exp.unet_fine.frozen)

        # Setup Losses
        self.affine_layer = mn.networks.layers.AffineTransform(
            zero_centered=True,
        )
        self.loss_dc = mn.losses.DiceLoss(
            include_background=True,
            reduction="none",
            batch=False,
            jaccard=False,
            weight=None,
        )
        self.loss_jd = mn.losses.DiceLoss(
            include_background=True,
            reduction="none",
            batch=False,
            jaccard=True,
            weight=None,
        )
        self.loss_hf = mn.losses.HausdorffDTLoss(
            include_background=True,
            reduction="none",
            batch=False,
            alpha=2.0,
        )

        if self.exp.transform == dv.enums.TransformEnum.rigid:
            self.loss_ms = t.nn.MSELoss(reduction="none")

        # Setup Datasets
        self.dataset = dv.datasets.BasicDataset(self.exp)

        # Training
        self.dataset_trn = t.utils.data.Subset(
            self.dataset,
            self.dataset.get_trn_indices(),
        )
        self.dataloader_trn = t.utils.data.DataLoader(
            self.dataset_trn,
            batch_size=self.exp.batch_size,
            shuffle=True,
            num_workers=self.exp.num_workers,
            pin_memory=self.exp.pin_memory,
        )

        # Validation
        self.dataset_val = t.utils.data.Subset(
            self.dataset,
            self.dataset.get_val_indices(),
        )
        self.dataloader_val = t.utils.data.DataLoader(
            self.dataset_val,
            batch_size=self.exp.batch_size,
            shuffle=False,
            num_workers=self.exp.num_workers,
            pin_memory=self.exp.pin_memory,
        )

        # Testing
        self.dataset_tst = t.utils.data.Subset(
            self.dataset,
            self.dataset.get_tst_indices(),
        )
        self.dataloader_tst = t.utils.data.DataLoader(
            self.dataset_tst,
            batch_size=self.exp.batch_size,
            shuffle=False,
            num_workers=self.exp.num_workers,
            pin_memory=self.exp.pin_memory,
        )

        self.optimizer = t.optim.Adam(self.model.parameters(), lr=self.exp.lr_initial)
        self.scaler = t.amp.GradScaler(self.exp.device)
        self.set_lr(self.exp.lr_initial)
        schedulers = []
        if self.exp.lr_warmup:
            schedulers += [
                t.optim.lr_scheduler.LinearLR(
                    self.optimizer,
                    start_factor=self.exp.lr_warmup_start_factor,
                    total_iters=self.exp.lr_warmup_total_iters,
                )
            ]

        match self.exp.lr_strategy:
            case "constant":
                pass
            case "step":
                schedulers += [
                    t.optim.lr_scheduler.StepLR(
                        self.optimizer,
                        self.exp.lr_step_size,
                    )
                ]
            case "exponential":
                schedulers += [
                    t.optim.lr_scheduler.ExponentialLR(
                        self.optimizer,
                        gamma=self.exp.lr_exponential_gamma,
                    )
                ]
            case _:
                assert False, f"LR strategy '{self.exp.lr_strategy}' not recognized"

        match len(schedulers):
            case 0:
                pass
            case 1:
                self.scheduler = schedulers[0]
            case 2:
                self.scheduler = t.optim.lr_scheduler.SequentialLR(
                    self.optimizer,
                    schedulers,
                    milestones=[self.exp.lr_warmup_total_iters],
                )
        if len(schedulers) > 0:
            self.set_lr(self.scheduler.get_last_lr()[0])

        self.df = pd.DataFrame(
            columns=["Partition", "Epoch", "Sample", "Quantity", "Value"]
        )

        self.epoch = 0

    def set_seeds(self):
        """Set PyTorch, Numpy, and Python seeds to reduce stochastic effects."""
        t.manual_seed(self.exp.seed)
        random.seed(self.exp.seed)
        np.random.seed(self.exp.seed)

    def train(self):
        """Alternate between training and validation runs for the number of epochs specified in the experiment file."""
        self.exp.dir_output.mkdir(parents=True, exist_ok=True)
        for e in range(self.epoch, self.exp.epochs):
            self.run_epoch(dv.enums.TrainingModeEnum.trn)
            self.run_epoch(dv.enums.TrainingModeEnum.val)
            if hasattr(self, "scheduler"):
                self.scheduler.step()
                self.set_lr(self.scheduler.get_last_lr()[0])
            self.epoch += 1
            self.save_checkpoint()

    def test(self):
        """Run on the test set."""
        self.epoch -= 1
        self.run_epoch(dv.enums.TrainingModeEnum.tst)
        self.save_checkpoint(df_only=True)
        self.epoch += 1

    def run_epoch(self, mode: dv.enums.TrainingModeEnum):
        """Run a single epoch.

        Args:
            mode: Specify training, validation, or testing.
        """

        match mode:
            case dv.enums.TrainingModeEnum.trn:
                self.model.train()
                loop = tqdm.tqdm(self.dataloader_trn)
                self.dataset.augmentation = self.exp.augmentation
            case dv.enums.TrainingModeEnum.val:
                self.model.eval()
                loop = tqdm.tqdm(self.dataloader_val)
                self.dataset.augmentation = False
            case dv.enums.TrainingModeEnum.tst:
                self.model.eval()
                loop = tqdm.tqdm(self.dataloader_tst)
                self.dataset.augmentation = False
            case _:
                assert False, f"Mode {mode} not recognized."

        N = len(loop)

        if mode in {"trn", "val"}:
            loop.set_postfix(
                lr=self.lr,
                epoch=self.epoch,
            )

        err_rows = []

        todevice = mn.transforms.ToDeviced(
            keys=("img", "seg", "Q_all"),
            device=self.exp.device,
            allow_missing_keys=True,
        )

        for i, gt in enumerate(loop):
            gt = todevice(gt)
            # dv.visualization.quick_show(gt["img"][0], gt["seg"][0])

            gt["seg_coarse"] = self.affine_layer(
                gt["seg"],
                self.exp.A_full_to_coarse,
                spatial_size=self.exp.unet_coarse.shape,
            )
            gt["centroid"] = (
                self.model.calculate_centroid(gt["seg_coarse"][:, 1:, :].sum(1))
                * self.exp.coarse_fov
                / 2.0
            ).flip(1)
            # gt["img_coarse"] = self.affine_layer(
            #     gt["img"],
            #     self.exp.A_full_to_coarse,
            #     spatial_size=self.exp.unet_coarse.shape,
            # )
            # dv.visualization.quick_show(gt["img_coarse"][0], gt["seg_coarse"][0])

            match self.exp.transform:
                case dv.enums.TransformEnum.rigid:
                    for qc in self.exp.quat_compositions:
                        kname = f"Q_{qc.name}"
                        if len(qc.indices) > 1:
                            gt[kname] = rm.quat_composition(
                                [gt["Q_all"][:, i, ...] for i in qc.indices]
                            )
                        else:
                            gt[kname] = gt["Q_all"][:, qc.indices[0], ...]
                        gt[kname] = dv.geometry.quat_to_shortest(gt[kname])
                    gt["T_c2f"] = rm.RigidUnitQuat(
                        linear=gt[self.exp.c2f_key],
                        translation=gt["centroid"],
                    )
                    gt["T_c2f_hom"] = gt["T_c2f"].to_homogeneous()
                case dv.enums.TransformEnum.pca:
                    assert False
                    gt_transform = self.model.transform_from_seg(
                        gt["seg_coarse"][:, 1:, :].sum(1),
                        self.exp.new_axis,
                    )
                    gt["transform"] = rm.RigidUnitQuat(
                        linear=gt_transform["quaternion"],
                        translation=gt_transform["centroid"],
                    )
                case dv.enums.TransformEnum.orthogonal:
                    assert False
                case _:
                    assert False, f"Transform {self.exp.Transform} not recognized."

            # img_fine = self.affine_layer(
            #     gt["img"],
            #     gt["T_c2f_hom"],
            #     spatial_size=self.exp.unet_fine.shape,
            # )
            # seg_fine = self.affine_layer(
            #     gt["seg"],
            #     gt["T_c2f_hom"],
            #     spatial_size=self.exp.unet_fine.shape,
            # )
            # dv.visualization.quick_show(img_fine[0], seg_fine[0])
            # dv.visualization.quick_show(seg_fine[0,5:6], window=9)

            # https://pytorch.org/docs/main/notes/amp_examples.html
            with t.amp.autocast(self.exp.device):
                with t.inference_mode(mode in dv.enums.TrainingModeEnum.inf()):
                    pn = self.model(gt["img"])

                sum_loss = t.tensor(0, device=self.exp.device)
                err = dict()

                # gt["img_coarse"] = self.affine_layer(
                #     gt["img"],
                #     self.exp.A_full_to_coarse,
                #     spatial_size=self.exp.unet_coarse.shape,
                # )
                # dv.visualization.quick_show(gt["img_coarse"][0], pn["seg_coarse"][0])
                # dv.visualization.quick_show(pn["img_fine"][0], pn["seg_fine"][0], seg_scale=self.exp.plotly_colorscale)

                ##
                ## Coarse
                ##

                # Upsample segmentation to original resolution.
                pn["seg_coarse_upsampled"] = self.affine_layer(
                    pn["seg_coarse"],
                    self.exp.A_coarse_to_full,
                    spatial_size=self.exp.lg_fov_hi_res_shape,
                ).softmax(1)
                # dv.visualization.quick_show(gt["img"][0], pn["seg_coarse_upsampled"][0])

                if self.exp.coarse_dc.should_calc:
                    dc_c = self.loss_dc(pn["seg_coarse_upsampled"], gt["seg"])[
                        :, :, 0, 0, 0
                    ]
                    for i in range(dc_c.shape[1]):
                        err[f"coarse_dc_{self.exp.get_label(i).text}"] = dc_c[
                            :, i
                        ].tolist()
                    dc_c = dc_c.mean(1)
                    err[f"coarse_dc_avg"] = dc_c.tolist()
                    if self.exp.coarse_dc.weight > 0.0:
                        sum_loss = sum_loss + dc_c.mean() * self.exp.coarse_dc.weight
                if self.exp.coarse_jd.should_calc:
                    jd_c = self.loss_jd(pn["seg_coarse_upsampled"], gt["seg"])[
                        :, :, 0, 0, 0
                    ]
                    for i in range(jd_c.shape[1]):
                        err[f"coarse_jd_{self.exp.get_label(i).text}"] = jd_c[
                            :, i
                        ].tolist()
                    jd_c = jd_c.mean(1)
                    err[f"coarse_jd_avg"] = jd_c.tolist()
                    if self.exp.coarse_jd.weight > 0.0:
                        sum_loss = sum_loss + jd_c.mean() * self.exp.coarse_jd.weight
                if self.exp.coarse_hf.should_calc:
                    hf_c = self.loss_hf(pn["seg_coarse_upsampled"], gt["seg"])[
                        :, :, 0, 0, 0
                    ]
                    for i in range(hf_c.shape[1]):
                        err[f"coarse_hf_{self.exp.get_label(i).text}"] = hf_c[
                            :, i
                        ].tolist()
                    hf_c = hf_c.mean(1)
                    err[f"coarse_hf_avg"] = hf_c.tolist()
                    if self.exp.coarse_hf.weight > 0.0:
                        sum_loss = sum_loss + hf_c.mean() * self.exp.coarse_hf.weight
                if self.exp.coarse_cd.should_calc:
                    cd_l = gt["centroid"] - pn["centroid"]
                    cd_l = cd_l.pow(2).sum(dim=1).pow(0.5)
                    err["coarse_cd"] = cd_l.tolist()
                    if self.exp.coarse_cd.weight > 0.0:
                        sum_loss = sum_loss + cd_l.mean() * self.exp.coarse_cd.weight

                ##
                ## Quaternion
                ##

                if not self.exp.stop_after_coarse:
                    if self.exp.transform == dv.enums.TransformEnum.rigid:
                        pn["T_c2f_inv"] = pn["T_c2f"].normalize().inverse()
                        for qc in self.exp.quat_compositions:
                            kname = f"Q_{qc.name}"
                            if qc.mse.should_calc:
                                mse_qc = self.loss_ms(gt[kname], pn[kname]).mean(-1)
                                err[f"quat_ms_{qc.name}"] = mse_qc.tolist()
                                if qc.mse.weight > 0.0:
                                    sum_loss = sum_loss + mse_qc.mean() * qc.mse.weight
                            if qc.geodesic.should_calc:
                                geodesic_qc = rm.unitquat_geodesic_distance(
                                    rm.quat_normalize(pn[kname]),
                                    gt[kname],
                                )
                                err[f"quat_gc_{qc.name}"] = geodesic_qc.tolist()
                                if qc.geodesic.weight > 0.0:
                                    sum_loss = (
                                        sum_loss
                                        + geodesic_qc.mean() * qc.geodesic.weight
                                    )
                            if qc.angle.should_calc:
                                angle_qc = dv.geometry.unitquat_angle(
                                    rm.quat_normalize(pn[kname]),
                                    gt[kname],
                                )
                                err[f"quat_al_{qc.name}"] = angle_qc.tolist()
                                if qc.angle.weight > 0.0:
                                    sum_loss = (
                                        sum_loss + angle_qc.mean() * qc.angle.weight
                                    )

                ##
                ## Fine
                ##

                if not self.exp.stop_after_coarse and not self.exp.stop_after_fc:
                    bg = t.zeros((1, self.exp.classes, 1, 1, 1), device=self.exp.device)
                    bg[:, 0, ...] = 1.0
                    pn["seg_fine_upsampled"] = pn["seg_fine"].softmax(1)
                    pn["seg_fine_upsampled"] = pn["seg_fine_upsampled"].add(
                        bg, alpha=-1.0
                    )
                    pn["seg_fine_upsampled"] = self.affine_layer(
                        pn["seg_fine_upsampled"],
                        pn["T_c2f_inv"].to_homogeneous(),
                        spatial_size=self.exp.lg_fov_hi_res_shape,
                    )
                    pn["seg_fine_upsampled"] = pn["seg_fine_upsampled"].add(bg)
                    # dv.visualization.quick_show(pn["seg_fine_upsampled"][0,0:1], window=9)

                    if self.exp.fine_dc.should_calc:
                        dc_f = self.loss_dc(pn["seg_fine_upsampled"], gt["seg"])[
                            :, :, 0, 0, 0
                        ]
                        for i in range(dc_f.shape[1]):
                            err[f"fine_dc_{self.exp.get_label(i).text}"] = dc_f[
                                :, i
                            ].tolist()
                        dc_f = dc_f.mean(1)
                        err[f"fine_dc_avg"] = dc_f.tolist()
                        if self.exp.fine_dc.weight > 0.0:
                            sum_loss = sum_loss + dc_f.mean() * self.exp.fine_dc.weight
                    if self.exp.fine_jd.should_calc:
                        jd_f = self.loss_jd(pn["seg_fine_upsampled"], gt["seg"])[
                            :, :, 0, 0, 0
                        ]
                        for i in range(jd_f.shape[1]):
                            err[f"fine_jd_{self.exp.get_label(i).text}"] = jd_f[
                                :, i
                            ].tolist()
                        jd_f = jd_f.mean(1)
                        err[f"fine_jd_avg"] = jd_f.tolist()
                        if self.exp.fine_jd.weight > 0.0:
                            sum_loss = sum_loss + jd_f.mean() * self.exp.fine_jd.weight
                    if self.exp.fine_hf.should_calc:
                        hf_f = self.loss_hf(pn["seg_fine_upsampled"], gt["seg"])[
                            :, :, 0, 0, 0
                        ]
                        for i in range(hf_f.shape[1]):
                            err[f"fine_hf_{self.exp.get_label(i).text}"] = hf_f[
                                :, i
                            ].tolist()
                        hf_f = hf_f.mean(1)
                        err[f"fine_hf_avg"] = hf_f.tolist()
                        if self.exp.fine_hf.weight > 0.0:
                            sum_loss = sum_loss + hf_f.mean() * self.exp.fine_hf.weight

                # big_splash(
                #    gt["img"],
                #    gt["seg"],
                #    gt["Q_sax"],
                #    gt["Q_2ch"],
                #    gt["Q_3ch"],
                #    gt["Q_4ch"],
                #    gt["centroid"],
                #    self.exp.unet_fine.shape,
                #    self.exp.plotly_colorscale,
                # )

                # big_splash(
                #    gt["img"],
                #    pn["seg_fine_upsampled"],
                #    pn["Q_sax"],
                #    pn["Q_2ch"],
                #    pn["Q_3ch"],
                #    pn["Q_4ch"],
                #    pn["centroid"],
                #    self.exp.unet_fine.shape,
                #    self.exp.plotly_colorscale,
                # )

                # Backward
                if mode == "trn":
                    # sum_loss_scale = dv.utils.uneven_batch_size(
                    #    i, self.exp.batch_size_virtual, N
                    # )
                    # self.scaler.scale(sum_loss / sum_loss_scale).backward()
                    # if ((i + 1) % self.exp.batch_size_virtual == 0) or (
                    #    (i + 1) == N
                    # ):
                    #    self.scaler.step(self.optimizer)
                    #    self.scaler.update()
                    #    self.optimizer.zero_grad()
                    self.scaler.scale(sum_loss).backward()

                    if self.exp.clip_grad_norm is not None:
                        self.scaler.unscale_(self.optimizer)
                        t.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.exp.clip_grad_norm
                        )

                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()

                for b in range(self.exp.batch_size):
                    for k, v in err.items():
                        data = [
                            mode,
                            self.epoch,
                            gt["index"].tolist()[b],
                            k,
                            v[b],
                        ]
                        err_rows += [data]

        err_df = pd.DataFrame(err_rows, columns=self.df.columns)
        self.df = pd.concat([_df for _df in [self.df, err_df] if not _df.empty])

    def save_checkpoint(self, df_only=False):
        """Save checkpoint."""
        self.df.to_csv(self.exp.dataframe_path, index=False)
        if df_only:
            return
        checkpoint = {
            "epoch": self.epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        if self.exp.lr_strategy != "constant":
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        t.save(checkpoint, self.exp.checkpoint_last_path)

        df_filtered = self.df[self.df.Partition == "val"]
        df_filtered = df_filtered[
            df_filtered.Quantity == self.exp.checkpoint_best_metric
        ]
        df_filtered = df_filtered.groupby("Epoch")["Value"].mean()
        if df_filtered.iloc[-1] <= df_filtered.min():
            print("Saving new best checkpoint.")
            t.save(checkpoint, self.exp.checkpoint_best_path)

    def load_checkpoint(self, path: pl.Path):
        """Load checkpoint from path.

        Args:
            path: Path to checkpoint.
        """
        checkpoint = t.load(path, weights_only=True)

        self.epoch = checkpoint["epoch"]
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.exp.lr_strategy != "constant":
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            self.set_lr(self.scheduler.get_last_lr()[0])
        self.df = pd.read_csv(self.exp.dataframe_path, index_col=False)

    def set_lr(self, lr):
        self.lr = f"{lr:.2e}"

    def load_checkpoint_helper(self, option: dv.enums.CheckpointEnum):
        """Load a checkpoint.

        Args:
            option: Specify which checkpoint to load.
        """
        match option:
            case dv.enums.CheckpointEnum.pretrain:
                self.load_checkpoint(self.exp.checkpoint_pretrain_path)
            case dv.enums.CheckpointEnum.last:
                self.load_checkpoint(self.exp.checkpoint_last_path)
            case dv.enums.CheckpointEnum.best:
                self.load_checkpoint(self.exp.checkpoint_best_path)
            case _:
                assert False, f"Option {option} not recognized."

    def save_segmentations(self, option: dv.enums.TrainingModeEnum):
        assert False, "Not implemented."

    def process_4d(self, path: pl.Path):
        load = mn.transforms.LoadImage(
            image_only=True,
            ensure_channel_first=True,
            simple_keys=True,
        )
        orient = mn.transforms.Orientation(
            axcodes=self.exp.orientation,
        )
        space = mn.transforms.Spacing(
            pixdim=(self.exp.unet_fine.spacing for x in range(self.exp.dim)),
            mode="bilinear",
        )
        crop = mn.transforms.ResizeWithPadOrCrop(
            spatial_size=self.exp.lg_fov_hi_res_shape,
        )
        normalize = mn.transforms.ScaleIntensityRange(
            a_min=self.exp.clip_min,
            a_max=self.exp.clip_max,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        )
        i = load(path)
        i = orient(i)
        i = space(i)
        i = crop(i)
        i = normalize(i)
        i = i.unsqueeze(1)

        orient_viewer = mn.transforms.Orientation(
            axcodes="SAL",
        )
        numpify = mn.transforms.ToNumpy()
        img_splashes = []
        seg_splashes = []

        for idx in range(i.shape[0]):
            f = i[idx : idx + 1, ...]
            f = f.to(self.exp.device)
            self.model.eval()
            with t.amp.autocast(self.exp.device):
                with t.inference_mode(True):
                    o = self.model(f)

            o["img_fine"] = orient_viewer(o["img_fine"][0])
            o["img_fine"] = numpify(o["img_fine"])[0]
            img_splashes += [splash(o["img_fine"])]

            o["seg_fine"] = o["seg_fine"].softmax(1)
            o["seg_fine"] = orient_viewer(o["seg_fine"][0])
            o["seg_fine"] = numpify(o["seg_fine"].argmax(0))
            seg_splashes += [splash(o["seg_fine"])]

        img_splashes = np.concatenate(img_splashes)
        seg_splashes = np.concatenate(seg_splashes)
        hm_img = dv.visualization.HeatmapData(
            data=img_splashes,
            range_min=0.0,
            range_max=1.0,
            colorscale="gray",
            opacity=1.0,
        )
        hm_seg = dv.visualization.HeatmapData(
            data=seg_splashes,
            range_min=0,
            range_max=6,
            colorscale=self.exp.plotly_colorscale,
            opacity=0.5,
        )
        viewer = dv.visualization.PlaneViewer(0, 7, [hm_img, hm_seg])
        # viewer = dv.visualization.PlaneViewer(0, 7, [hm_img])
        viewer.app.run_server(debug=True, port=8050)


def big_splash(i, s, sax, vla, oft, hla, c, shape, seg_map="jet"):
    affine_layer = mn.networks.layers.AffineTransform(
        zero_centered=True,
    )
    numpify = mn.transforms.ToNumpy()

    T_sax = (
        rm.RigidUnitQuat(
            linear=sax,
            translation=c,
        )
        .normalize()
        .to_homogeneous()
    )
    T_vla = (
        rm.RigidUnitQuat(
            linear=vla,
            translation=c,
        )
        .normalize()
        .to_homogeneous()
    )
    T_oft = (
        rm.RigidUnitQuat(
            linear=oft,
            translation=c,
        )
        .normalize()
        .to_homogeneous()
    )
    T_hla = (
        rm.RigidUnitQuat(
            linear=hla,
            translation=c,
        )
        .normalize()
        .to_homogeneous()
    )

    img_sax = affine_layer(
        i,
        T_sax,
        spatial_size=shape,
    )
    seg_sax = affine_layer(
        s,
        T_sax,
        spatial_size=shape,
    )
    img_vla = affine_layer(
        i,
        T_vla,
        spatial_size=shape,
    )
    seg_vla = affine_layer(
        s,
        T_vla,
        spatial_size=shape,
    )
    img_oft = affine_layer(
        i,
        T_oft,
        spatial_size=shape,
    )
    seg_oft = affine_layer(
        s,
        T_oft,
        spatial_size=shape,
    )
    img_hla = affine_layer(
        i,
        T_hla,
        spatial_size=shape,
    )
    seg_hla = affine_layer(
        s,
        T_hla,
        spatial_size=shape,
    )
    orient_sax = mn.transforms.Orientation(
        axcodes="SAL",
    )
    seg_sax = seg_sax.softmax(1)
    seg_sax = orient_sax(seg_sax[0])
    seg_sax = numpify(seg_sax.argmax(0))
    length = np.count_nonzero(np.any(seg_sax, axis=(1, 2)))
    buffer = np.nonzero(np.any(seg_sax, axis=(1, 2)))[0][0]
    seg_sax = splash(seg_sax, ncols=3, length=length, buffer=buffer)

    img_sax = orient_sax(img_sax[0])
    img_sax = numpify(img_sax)[0]
    img_sax = splash(img_sax, ncols=3, length=length, buffer=buffer)

    orient_vla = mn.transforms.Orientation(
        axcodes="LSP",
    )
    img_vla = orient_vla(img_vla[0])
    img_vla = numpify(img_vla)[0, 63:64]
    seg_vla = seg_vla.softmax(1)
    seg_vla = orient_vla(seg_vla[0])
    seg_vla = numpify(seg_vla.argmax(0))[63:64]

    orient_oft = mn.transforms.Orientation(
        axcodes="LSP",
    )
    img_oft = orient_oft(img_oft[0])
    img_oft = numpify(img_oft)[0, 63:64]
    seg_oft = seg_oft.softmax(1)
    seg_oft = orient_oft(seg_oft[0])
    seg_oft = numpify(seg_oft.argmax(0))[63:64]

    orient_hla = mn.transforms.Orientation(
        axcodes="LSA",
    )
    img_hla = orient_hla(img_hla[0])
    img_hla = numpify(img_hla)[0, 63:64]
    seg_hla = seg_hla.softmax(1)
    seg_hla = orient_hla(seg_hla[0])
    seg_hla = numpify(seg_hla.argmax(0))[63:64]

    img_lax = np.concatenate([img_vla, img_oft, img_hla], axis=2)
    seg_lax = np.concatenate([seg_vla, seg_oft, seg_hla], axis=2)

    img = np.concatenate([img_lax, img_sax], axis=1)
    seg = np.concatenate([seg_lax, seg_sax], axis=1)

    hm_img = dv.visualization.HeatmapData(
        data=img,
        range_min=0.0,
        range_max=1.0,
        colorscale="gray",
        opacity=1.0,
    )
    hm_seg = dv.visualization.HeatmapData(
        data=seg,
        range_min=0,
        range_max=4,
        colorscale=seg_map,
        opacity=0.4,
    )
    viewer = dv.visualization.PlaneViewer(0, 7, [hm_img, hm_seg])
    viewer.app.run_server(debug=True, port=8050)
    assert False


def splash(i, nrows=4, ncols=4, length=None, buffer=None):
    if length is None:
        length = i.shape[0]
    skip = length // (nrows * ncols)
    if buffer is None:
        buffer = (i.shape[0] - skip * (nrows * ncols - 1)) // 2
    print(skip, buffer)
    i = i[buffer : buffer + skip * (nrows * ncols - 1) + 1 : skip, ...]
    print(i.shape)
    w = i.shape[1] * nrows
    h = i.shape[2] * ncols
    i = i.reshape(nrows, ncols, i.shape[1], i.shape[2])
    i = i[:, ::-1]
    i = i.transpose(0, 2, 1, 3)
    i = i.reshape(1, w, h)
    return i


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type=pl.Path)
    parser.add_argument("--load-checkpoint", choices=("pretrain", "last", "best"))
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--save", choices=("trn", "val", "tst"))
    parser.add_argument("--process-4d", type=pl.Path)
    args = parser.parse_args()

    runner = CCTANetRunner(args.experiment)

    if args.load_checkpoint is not None:
        runner.load_checkpoint_helper(args.load_checkpoint)
    if args.train:
        runner.train()
    if args.test:
        runner.test()
    if args.save is not None:
        runner.save_segmentations(args.save)
    if args.process_4d is not None:
        runner.process_4d(args.process_4d)


if __name__ == "__main__":
    main()
