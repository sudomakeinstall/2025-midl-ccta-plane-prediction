# CCTA Plane Prediction and Segmentation

This repository contains the code to reproduce results for the following article, which has been submitted to *Medical Imaging with Deep Learning* 2025 and is currently under review:

    Vigneault DM, Manohar A, Hernandez A. Cardiac Computed Tomography Angiography Plane Prediction and Comprehensive LV Segmentation. MIDL 2025. Salt Lake City, UT, USA. (Under Review)

Experiment files are found in `./experiments/midl-2025/`.  After modifying the paths in the experiment files to match the paths to input and output data folders, an experiment may be trained as follows:

    ./run-ccta-net.py \
        ./experiments/midl-2025/AtnY-ResY-CsdY-Hdn-128-128.toml \
        --train

Testing may be performed as follows

    ./run-ccta-net.py \
        ./experiments/midl-2025/AtnY-ResY-CsdY-Hdn-128-128.toml \
        --load-checkpoint=last \
        --test

Please see the above referenced article for further details.
