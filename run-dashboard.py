#!/usr/bin/env python

# System
import argparse
import pathlib as pl

# Custom
import dv


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csvs", type=pl.Path, nargs="+")
    parser.add_argument("--port", type=int, default=8050)
    args = parser.parse_args()

    dashboard = dv.visualization.Dashboard(csv_paths=args.csvs)
    dashboard.app.run_server(debug=True, port=args.port)
