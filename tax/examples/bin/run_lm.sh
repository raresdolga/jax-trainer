#!/bin/bash
# Create a generic run script
ROOT_DIR=${1:-/home/rares/latte_trans}
CONFIG_FILE=${2:-lm.yaml} # specify default
NAME=${3:-${CONFIG_FILE::-5}} # default is config file name apart from .yaml

echo "Root Dir: ${ROOT_DIR} Name Experiment: ${NAME} Conf File: ${CONFIG_FILE}"
# nohup # 0
JAX_TRACEBACK_FILTERING=off WANDB_MODE="online" XLA_PYTHON_CLIENT_MEM_FRACTION="0.97" CUDA_VISIBLE_DEVICES="4,5,6,7" pdm run python -u -m tax.examples.gemma.exp \
    --base_dir $ROOT_DIR/data/ \
    --config_file $ROOT_DIR/tax/examples/configs/${CONFIG_FILE}\
    --name $NAME \
#>$ROOT_DIR/'data/logs_labl/'$NAME'_'$BASHPID'.log'  2>&1 &