#!/usr/bin/env bash

readonly EXPERIMENTS=(
AtnY-ResY-CsdY-IndY-DirY-Hdn-64.toml
AtnY-ResY-CsdY-IndY-DirY-Hdn-64-64.toml
AtnY-ResY-CsdY-IndY-DirY-Hdn-128.toml
AtnY-ResY-CsdY-IndY-DirY-Hdn-128-128.toml
AtnY-ResY-CsdY-IndY-DirY-Hdn-256.toml
AtnY-ResY-CsdY-IndY-DirY-Hdn-256-256.toml
AtnN-ResY-CsdY-IndY-DirY-Hdn-128-128.toml
AtnY-ResN-CsdY-IndY-DirY-Hdn-128-128.toml
AtnY-ResY-CsdN-IndY-DirY-Hdn-128-128.toml
AtnY-ResY-CsdY-IndN-DirY-Hdn-128-128.toml
AtnY-ResY-CsdY-IndY-DirN-Hdn-128-128.toml
AtnY-ResY-CsdY-IndY-DirY-HdnN.toml
AtnY-ResY-CsdY-IndY-DirY-Hdn-128-128-NO-FINE.toml
AtnY-ResY-CsdY-IndY-DirY-Hdn-128-128-2CH.toml
AtnY-ResY-CsdY-IndY-DirY-Hdn-128-128-3CH.toml
AtnY-ResY-CsdY-IndY-DirY-Hdn-128-128-4CH.toml
AtnY-ResY-CsdY-IndY-DirY-Hdn-128-128-SAX.toml
)

for e in ${EXPERIMENTS[*]};
do

echo ${plane}

./run-ccta-net.py \
    ./experiments/midl-2025/${e} \
    --train

./run-ccta-net.py \
    ./experiments/midl-2025/${e} \
    --load-checkpoint=last \
    --test

done

./run-ccta-net.py \
    ./experiments/midl-2025/AtnY-ResY-CsdY-IndY-DirY-Hdn-128-128-COARSE.toml \
    --train

mkdir -p ~/projects/midl-2025/AtnY-ResY-CsdY-IndY-DirY-Hdn-128-128-TRANSFORM/
cp \
    ~/projects/midl-2025/AtnY-ResY-CsdY-IndY-DirY-Hdn-128-128-COARSE/checkpoint-last.pth.tar \
    ~/projects/midl-2025/AtnY-ResY-CsdY-IndY-DirY-Hdn-128-128-TRANSFORM/checkpoint-pretrain.pth.tar
cp \
    ~/projects/midl-2025/AtnY-ResY-CsdY-IndY-DirY-Hdn-128-128-COARSE/data.csv \
    ~/projects/midl-2025/AtnY-ResY-CsdY-IndY-DirY-Hdn-128-128-TRANSFORM/data.csv

./run-ccta-net.py \
    ./experiments/midl-2025/AtnY-ResY-CsdY-IndY-DirY-Hdn-128-128-TRANSFORM.toml \
    --load-checkpoint=pretrain \
    --train

mkdir -p ~/projects/midl-2025/AtnY-ResY-CsdY-IndY-DirY-Hdn-128-128-FINE/
cp \
    ~/projects/midl-2025/AtnY-ResY-CsdY-IndY-DirY-Hdn-128-128-TRANSFORM/checkpoint-last.pth.tar \
    ~/projects/midl-2025/AtnY-ResY-CsdY-IndY-DirY-Hdn-128-128-FINE/checkpoint-pretrain.pth.tar
cp \
    ~/projects/midl-2025/AtnY-ResY-CsdY-IndY-DirY-Hdn-128-128-TRANSFORM/data.csv \
    ~/projects/midl-2025/AtnY-ResY-CsdY-IndY-DirY-Hdn-128-128-FINE/data.csv
./run-ccta-net.py \
    ./experiments/midl-2025/AtnY-ResY-CsdY-IndY-DirY-Hdn-128-128-FINE.toml \
    --train
