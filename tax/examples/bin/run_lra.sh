#!/bin/bash
# Create a generic run script
ROOT_DIR=${1:-/home/rares/latte_trans}
CONFIG_FILE=${2:-lm.yaml} # specify default
NAME=${3:-${CONFIG_FILE::-5}} # default is config file name apart from .yaml

# bash bash/run.sh /home/rares/latte train_lm lm.yaml experimet_name
echo "Root Dir: ${ROOT_DIR} Name Experiment: ${NAME} Conf File: ${CONFIG_FILE}"
# nohup # 0,1,2,3,4,5,6,7  -  0,1,2,3  4,5,6,7 nohup - 0,1,2,3,
WANDB_MODE="online" XLA_PYTHON_CLIENT_MEM_FRACTION="0.90" CUDA_VISIBLE_DEVICES="0" pdm run python -u -m tax.examples.lra.experiment \
    --base_dir $ROOT_DIR/data/ \
    --config_file $ROOT_DIR/tax/examples/configs/${CONFIG_FILE}\
    --name $NAME \
# >$ROOT_DIR/'data/logs/'$NAME'_'$BASHPID'.log'  2>&1 &
# /data_rares/data/ 
# /mnt/data 
# logs_match
# /mnt/data  \