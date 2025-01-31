# System
import functools as ft
import pathlib as pl

# Third Party
import pydantic as pc
import pandas as pd
import numpy as np
import dash
import plotly.graph_objects as go
import plotly.express as px
import monai as mn
import tomlkit as tk
import torch as t

# Custom
from . import utils


@pc.dataclasses.dataclass(config=dict(frozen=True))
class WindowLevel:
    name: str
    window: float
    level: float

    @pc.computed_field
    @ft.cached_property
    def lower(self) -> float:
        return self.level - self.window / 2

    @pc.computed_field
    @ft.cached_property
    def upper(self) -> float:
        return self.level + self.window / 2


def WindowLevelPresets():
    return {
        1: WindowLevel("Abdomen", 400, 40),
        2: WindowLevel("Lung", 1500, -700),
        3: WindowLevel("Liver", 100, 110),
        4: WindowLevel("Bone", 1500, 500),
        5: WindowLevel("Brain", 85, 42),
        6: WindowLevel("Stroke", 36, 28),
        7: WindowLevel("Vascular", 800, 200),
        8: WindowLevel("Subdural", 160, 60),
        9: WindowLevel("Normalized", 0, 1),
    }


def segmentation_viewer_factory(
    heatmaps,
    dim,
    figsize=700,
):
    assert len(heatmaps) >= 1, "At least one heatmap must be supplied."

    fig = go.Figure()

    frames = [
        go.Frame(
            data=[
                go.Heatmap(
                    z=hm.data.take(f, axis=dim),
                    zmin=hm.range_min,
                    zmax=hm.range_max,
                    colorscale=hm.colorscale,
                    showscale=False,
                    opacity=hm.opacity,
                )
                for hm in heatmaps
            ],
            name=str(f),
        )
        for f in range(heatmaps[0].data.shape[dim])
    ]

    fig.frames = frames

    for d in fig.frames[0].data:
        fig.add_trace(d)

    def frame_args(duration):
        return {
            "frame": {"duration": duration},
            "mode": "immediate",
            "fromcurrent": True,
            "transition": {"duration": duration, "easing": "linear"},
        }

    sliders = [
        {
            "pad": {"b": 0, "t": 0, "l": 0, "r": 0},
            "len": 1.0,
            "x": 0,
            "y": 0,
            "steps": [
                {
                    "args": [[f.name], frame_args(0)],
                    "label": str(k),
                    "method": "animate",
                }
                for k, f in enumerate(fig.frames)
            ],
        }
    ]

    menus = [
        dict(
            type="buttons",
            buttons=[
                dict(
                    args=[
                        None,
                        {
                            "frame": {"duration": 10, "redraw": True},
                            "fromcurrent": True,
                            "transition": {"duration": 10},
                        },
                    ],
                    label="Play",
                    method="animate",
                )
            ],
        )
    ]

    fig.update_layout(
        width=figsize,
        height=figsize,
        sliders=sliders,
        yaxis_scaleanchor="x",
        updatemenus=menus,
    )

    return fig


@pc.dataclasses.dataclass(config=dict(arbitrary_types_allowed=True))
class HeatmapData:
    data: np.ndarray
    range_min: float
    range_max: float
    colorscale: str | list[tuple[float, str]]
    opacity: float


@pc.dataclasses.dataclass(config=dict(arbitrary_types_allowed=True))
class PlaneViewerHelper:
    img_path: pl.Path
    seg_path: pl.Path
    exp_path: pl.Path
    seg_opacity: float
    dim: int
    spacing: float
    orientation: str
    window: int
    device: str

    @pc.computed_field
    @ft.cached_property
    def heatmaps(self) -> list[HeatmapData]:
        hm_img = HeatmapData(
            data=self.data["img"],
            range_min=self.wl_presets[self.window].lower,
            range_max=self.wl_presets[self.window].upper,
            colorscale="gray",
            opacity=1.0,
        )
        hm_seg = HeatmapData(
            data=self.data["seg"],
            range_min=self.label_min,
            range_max=self.label_max,
            colorscale=self.exp.plotly_colorscale,
            opacity=self.seg_opacity,
        )
        return [hm_img, hm_seg]

    @pc.computed_field
    @ft.cached_property
    def exp(self) -> utils.Experiment:
        """Experiment class instance."""
        with open(self.exp_path, mode="rt", encoding="utf-8") as fp:
            exp_dict = tk.load(fp)
        try:
            exp = utils.Experiment(**exp_dict)
        except pc.ValidationError as e:
            print(e.errors())
        return exp

    @pc.computed_field
    @ft.cached_property
    def label_min(self) -> int:
        return min(self.exp.label_map.values())

    @pc.computed_field
    @ft.cached_property
    def label_max(self) -> int:
        return max(self.exp.label_map.values()) + 1

    @pc.computed_field
    @ft.cached_property
    def wl_presets(self) -> WindowLevelPresets:
        return WindowLevelPresets()

    @pc.computed_field
    @ft.cached_property
    def data(self) -> dict[str, t.Tensor]:
        load = mn.transforms.LoadImaged(
            keys=("img", "seg"),
            image_only=True,
            ensure_channel_first=True,
            simple_keys=True,
        )
        todevice = mn.transforms.ToDeviced(
            keys=("img", "seg"),
            device=self.device,
        )
        orient = mn.transforms.Orientationd(
            keys=("img", "seg"),
            axcodes=self.orientation,
        )
        remap = mn.transforms.MapLabelValued(
            keys=("seg"),
            orig_labels=self.exp.label_map.keys(),
            target_labels=self.exp.label_map.values(),
        )
        space = mn.transforms.Spacingd(
            keys=("img", "seg"),
            pixdim=(self.spacing for x in range(3)),
            mode=("bilinear", "nearest"),
        )
        squeeze = mn.transforms.SqueezeDimd(
            keys=("img", "seg"),
        )
        fromdevice = mn.transforms.ToDeviced(
            keys=("img", "seg"),
            device="cpu",
        )
        numpify = mn.transforms.ToNumpyd(
            keys=("img", "seg"),
        )
        pipeline = mn.transforms.Compose(
            [
                load,
                todevice,
                orient,
                # remap,
                space,
                squeeze,
                fromdevice,
                numpify,
            ]
        )
        x = pipeline({"img": self.img_path, "seg": self.seg_path})
        x["seg"][x["seg"] != 52] = 0
        x["seg"][x["seg"] == 52] = 1
        print(np.unique(x["seg"]))
        return x


@pc.dataclasses.dataclass(config=dict(arbitrary_types_allowed=True))
class PlaneViewer:
    dim: int
    window: int
    heatmaps: list[HeatmapData]

    @pc.computed_field
    @ft.cached_property
    def wl_presets(self) -> WindowLevelPresets:
        return WindowLevelPresets()

    @pc.computed_field
    @ft.cached_property
    def app(self) -> dash.Dash:
        fig_initial = segmentation_viewer_factory(
            self.heatmaps,
            self.dim,
        )
        app = dash.Dash(__name__)
        app._favicon = "favicon.svg"
        app.layout = dash.html.Div(
            [
                dash.dcc.Dropdown(
                    id="segmentation_wl",
                    options=[
                        {"label": v.name, "value": k}
                        for k, v in self.wl_presets.items()
                    ],
                    value=self.window,
                    clearable=False,
                    searchable=False,
                ),
                dash.dcc.Dropdown(
                    id="segmentation_axes",
                    options=[
                        {"label": "Axial", "value": 0},
                        {"label": "Coronal", "value": 1},
                        {"label": "Sagittal", "value": 2},
                    ],
                    value=0,
                    clearable=False,
                    searchable=False,
                ),
                dash.dcc.Graph(id="segmentation_viewer", figure=fig_initial),
            ]
        )
        app.callback(
            dash.Output("segmentation_viewer", "figure"),
            dash.Input("segmentation_axes", "value"),
            dash.Input("segmentation_wl", "value"),
        )(self.update)
        return app

    def update(self, dimension, window):
        fig = segmentation_viewer_factory(
            self.heatmaps,
            dimension,
        )
        return fig


def quick_show(
    img, seg=None, img_scale="gray", seg_scale="jet", dim=0, window=7, port=8050
):
    """Expects `img` and optionally `seg` to be single 4D images (channel+3D)."""

    data_dict = {"img": img}
    if seg is not None:
        data_dict["seg"] = seg

    todevice = mn.transforms.ToDeviced(
        keys=("img", "seg"),
        device="cpu",
        allow_missing_keys=True,
    )
    orient = mn.transforms.Orientationd(
        keys=("img", "seg"),
        axcodes="SAL",
        allow_missing_keys=True,
    )
    argmax = mn.transforms.AsDiscreted(
        keys=("seg"),
        argmax=True,
        allow_missing_keys=True,
    )
    squeeze = mn.transforms.SqueezeDimd(
        keys=("img", "seg"),
        allow_missing_keys=True,
    )
    tonumpy = mn.transforms.ToNumpyd(
        keys=("img", "seg"),
        allow_missing_keys=True,
    )
    pipeline = mn.transforms.Compose(
        [
            todevice,
            orient,
            argmax,
            squeeze,
            tonumpy,
        ]
    )

    output = pipeline(data_dict)

    maps = [
        HeatmapData(
            data=output["img"],
            range_min=0.0,
            range_max=1.0,
            colorscale=img_scale,
            opacity=1.0,
        )
    ]
    if seg is not None:
        maps += [
            HeatmapData(
                data=output["seg"],
                range_min=0,
                range_max=output["seg"].max(),
                colorscale=seg_scale,
                opacity=0.4,
            )
        ]
    viewer = PlaneViewer(dim, window, maps)
    viewer.app.run_server(debug=True, port=port)


@pc.dataclasses.dataclass(config=dict(arbitrary_types_allowed=True))
class Dashboard:
    csv_paths: list[pl.Path]

    @pc.computed_field
    @ft.cached_property
    def variables(self) -> list[str]:
        return ["Experiment", "Quantity", "Partition"]

    @pc.computed_field
    @ft.cached_property
    def df(self) -> pd.DataFrame:
        dfs = [pd.read_csv(csv) for csv in self.csv_paths]
        for csv, df in zip(self.csv_paths, dfs):
            df["Experiment"] = csv.parts[-2]
        dfs = pd.concat(dfs)
        dfs = dfs.groupby(self.variables + ["Epoch"], as_index=False).Value.agg(
            ["mean", "std", "min", "max"]
        )
        return dfs

    @pc.computed_field
    @ft.cached_property
    def app(self) -> dash.Dash:
        app = dash.Dash(__name__)
        app._favicon = "favicon.svg"
        components = [
            dash.dcc.Dropdown(
                id=f"{x}_dropdown",
                options=self.df[x].unique(),
                value=self.df[x].unique()[0],
                multi=False,
            )
            for x in self.variables
        ]
        dt_col_ops = [{"name": i, "id": i} for i in self.df.columns]
        for t, op in zip(self.df.dtypes, dt_col_ops):
            if t == "float64":
                op["type"] = "numeric"
                op["format"] = dict(specifier=".4f")
        app.layout = dash.html.Div(
            [
                dash.dcc.Graph(id="graph"),
            ]
            + components
            + [
                dash.dcc.Dropdown(
                    id="series_dropdown",
                    options=self.variables,
                    value="Partition",
                    multi=False,
                ),
                dash.html.Button("Save Plot", id="save_button"),
                dash.html.Button("Print LaTeX", id="print_latex"),
            ]
            + [
                dash.dash_table.DataTable(
                    id="table",
                    columns=dt_col_ops,
                    # fixed_rows=dict(headers = True),
                    sort_action="native",
                    sort_mode="multi",
                    fill_width=False,
                )
            ]
        )
        app.callback(
            dash.Output("graph", "figure"),
            dash.Output("table", "data"),
            dash.Input("series_dropdown", "value"),
            *[dash.Input(f"{x}_dropdown", "value") for x in self.variables],
        )(self.update_figure)
        app.callback(
            *[dash.Output(f"{x}_dropdown", "multi") for x in self.variables],
            *[dash.Output(f"{x}_dropdown", "value") for x in self.variables],
            dash.Input("series_dropdown", "value"),
            *[dash.State(f"{x}_dropdown", "value") for x in self.variables],
        )(self.update_series)
        app.callback(
            dash.Input("save_button", "n_clicks"),
        )(self.save_plot)
        app.callback(
            dash.Input("print_latex", "n_clicks"),
        )(self.print_latex)
        return app

    def update_figure(self, series, *args):
        ensure_list = lambda x: x if isinstance(x, list) else [x]
        args = [ensure_list(arg) for arg in args]
        masks = [self.df[k].isin(v) for k, v in zip(self.variables, args)]
        masks = np.logical_and.reduce(masks)
        dff = self.df[masks]

        self.fig = px.line(
            dff,
            x="Epoch",
            y="mean",
            error_y="std",
            color=series,
            render_mode="svg",
            markers=True,
        )

        return self.fig, dff[dff.Epoch == dff.Epoch.max()].to_dict("records")

    def update_series(self, series, *args):
        ensure_not_list = lambda x: x if not isinstance(x, list) else x[0]
        args = tuple(ensure_not_list(x) for x in args)
        returns = [False] * len(self.variables)
        returns[self.variables.index(series)] = True
        return (*returns, *args)

    def save_plot(self, n_clicks):
        print("Saving plot...")
        self.fig.write_image("out.svg")

    def print_latex(self, n_clicks):
        quantities_fine = [
            "fine_dc_Bloodpool",
            "fine_dc_Myocardium",
            "fine_dc_Trabeculations",
        ]
        quantities_quat = [
            "quat_al_sax",
            "quat_al_2ch",
            "quat_al_3ch",
            "quat_al_4ch",
        ]
        quantities = (
            [
                "coarse_cd",
            ]
            + quantities_quat
            + quantities_fine
        )
        partition = "tst"
        epoch = 23

        df = self.df[self.df.Quantity.isin(quantities)]
        df = df[df.Partition == partition]
        df = df[df.Epoch == epoch]
        df = df[["Experiment", "Quantity", "mean"]]
        df = pd.pivot(df, index="Experiment", columns="Quantity", values="mean")
        for x in quantities_fine:
            df[x] = 1 - df[x]
        for x in quantities_quat:
            df[x] = df[x] * 180 / np.pi
        df = df[[*quantities]]
        df = df.to_latex(
            index=True,
            # formatters={"name": str.upper},
            float_format="{:.3f}".format,
        )
        print(df)
