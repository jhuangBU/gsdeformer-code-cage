# Workspace for Decimation Algorithm

## setup environment
conda env create -f environment.yaml
conda activate deep-cage

## for cage optimization
bash run.sh

## for mesh decimation
python decimate_test.py

note: the update_mesh() API effectively restarts the optimization process, recalculating error from the updated mesh instead of the original mesh starting the optimization, this might be undesirable?