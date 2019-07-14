#!/usr/bin/env bash
cd ./models/FlowNet2_Models/resample2d_package/
./make.sh
cd ../correlation_package
./make.sh
cd ../channelnorm_package
./make.sh
cd ../..