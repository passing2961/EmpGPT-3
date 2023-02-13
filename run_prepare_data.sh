#!/bin/bash

DOWNLOAD_DATA_DIR=./data/raw_data
PREPARE_DATA_DIR=./data/annotated_data

TASK_NAMES=("empathetic_dialogues")
DATA_TYPES=("train:ordered" "valid" "test")
SPECIAL_TASK="empathetic_dialogues"

for TASK_NAME in "${TASK_NAMES[@]}"
do
    for DATA_TYPE in "${DATA_TYPES[@]}"
    do
        echo "Downloading ${TASK_NAME}:${DATA_TYPE}"
        if [ ${TASK_NAME} == ${SPECIAL_TASK} ]; then
            python3 prepare_data.py --task ${TASK_NAME} --datatype ${DATA_TYPE} --train_experiencer_only True --datapath $DOWNLOAD_DATA_DIR --prepare_data_dir $PREPARE_DATA_DIR
        else
            python3 prepare_data.py --task ${TASK_NAME} --datatype ${DATA_TYPE} --datapath $DOWNLOAD_DATA_DIR --prepare_data_dir $PREPARE_DATA_DIR
        fi
        echo "Done preparing ${TASK_NAME}:${DATA_TYPE}"
    done
done