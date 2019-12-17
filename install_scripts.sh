#!/usr/bin/env bash
cd ./models/FlowNet2_Models/resample2d_package/
rm -rf *_cuda.egg-info build dist __pycache__
python3 setup.py develop

cd ../correlation_package
rm -rf *_cuda.egg-info build dist __pycache__
python3 setup.py develop

cd ../channelnorm_package
rm -rf *_cuda.egg-info build dist __pycache__
python3 setup.py develop
cd ../..