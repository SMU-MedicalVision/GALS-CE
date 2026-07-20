#!/bin/bash

set -e

echo "Attention [1/2] Stage I: Running CECT synthesis..."
python ./main/train_GALS-CE_gen.py --gpu 0 --quick_test

echo "Attention [2/2] Stage II: Running LMs origin identification..."
python ./main/train_GALS-CE_cla.py --gpu 0 --quick_test

echo "GALS-CE demo completed successfully."