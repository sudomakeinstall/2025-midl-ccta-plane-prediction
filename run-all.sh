#!/usr/bin/env bash

echo "Hidden 64"
./run-ccta-net.py \
    ./experiments/midl-2025/AtnY-ResY-CsdY-Hdn-64.toml \
    --train

echo "Hidden 64-64"
./run-ccta-net.py \
    ./experiments/midl-2025/AtnY-ResY-CsdY-Hdn-64-64.toml \
    --train

echo "Hidden 128"
./run-ccta-net.py \
    ./experiments/midl-2025/AtnY-ResY-CsdY-Hdn-128.toml \
    --train

echo "Hidden 128-128"
./run-ccta-net.py \
    ./experiments/midl-2025/AtnY-ResY-CsdY-Hdn-128-128.toml \
    --train

echo "Hidden 256"
./run-ccta-net.py \
    ./experiments/midl-2025/AtnY-ResY-CsdY-Hdn-256.toml \
    --train

echo "Hidden 256-256"
./run-ccta-net.py \
    ./experiments/midl-2025/AtnY-ResY-CsdY-Hdn-256-256.toml \
    --train

echo "No Attention"
./run-ccta-net.py \
    ./experiments/midl-2025/AtnN-ResY-CsdY-Hdn-128-128.toml \
    --train

echo "No Residual"
./run-ccta-net.py \
    ./experiments/midl-2025/AtnY-ResN-CsdY-Hdn-128-128.toml \
    --train

echo "No Cascading"
./run-ccta-net.py \
    ./experiments/midl-2025/AtnY-ResY-CsdN-Hdn-128-128.toml \
    --train

echo "Non-End-to-End: Coarse"
./run-ccta-net.py \
    ./experiments/midl-2025/AtnY-ResY-CsdY-Hdn-128-128-COARSE.toml \
    --train

echo "Non-End-to-End: Transform"
mkdir -p ~/projects/midl-2025/AtnY-ResY-CsdY-Hdn-128-128-TRANSFORM/
cp \
    ~/projects/midl-2025/AtnY-ResY-CsdY-Hdn-128-128-COARSE/checkpoint-last.pth.tar \
    ~/projects/midl-2025/AtnY-ResY-CsdY-Hdn-128-128-TRANSFORM/checkpoint-pretrain.pth.tar
cp \
    ~/projects/midl-2025/AtnY-ResY-CsdY-Hdn-128-128-COARSE/data.csv \
    ~/projects/midl-2025/AtnY-ResY-CsdY-Hdn-128-128-TRANSFORM/data.csv

./run-ccta-net.py \
    ./experiments/midl-2025/AtnY-ResY-CsdY-Hdn-128-128-TRANSFORM.toml \
    --load-checkpoint=pretrain \
    --train

echo "Non-End-to-End: Fine"
mkdir -p ~/projects/midl-2025/AtnY-ResY-CsdY-Hdn-128-128-FINE/
cp \
    ~/projects/midl-2025/AtnY-ResY-CsdY-Hdn-128-128-TRANSFORM/checkpoint-last.pth.tar \
    ~/projects/midl-2025/AtnY-ResY-CsdY-Hdn-128-128-FINE/checkpoint-pretrain.pth.tar
cp \
    ~/projects/midl-2025/AtnY-ResY-CsdY-Hdn-128-128-TRANSFORM/data.csv \
    ~/projects/midl-2025/AtnY-ResY-CsdY-Hdn-128-128-FINE/data.csv
./run-ccta-net.py \
    ./experiments/midl-2025/AtnY-ResY-CsdY-Hdn-128-128-FINE.toml \
    --train

echo "No Hidden"
./run-ccta-net.py \
    ./experiments/midl-2025/AtnY-ResY-CsdY-HdnN.toml \
    --train
