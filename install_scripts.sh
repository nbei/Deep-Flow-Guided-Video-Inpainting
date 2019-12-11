#!/usr/bin/env bash
cd ./models/FlowNet2_Models/resample2d_package/
bash make.sh
cd ../correlation_package
bash make.sh
cd ../channelnorm_package
bash make.sh
cd ../..