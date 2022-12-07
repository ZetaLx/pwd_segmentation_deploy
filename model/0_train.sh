#!/bin/bash

MODEL_NAME=segformer
PROJECT_NAME=${MODEL_NAME}_deploy

DATASET_NAME=ceshi
ROOT_DATASET=/home/zeta1996/20221012/unet_deploy/datasets/

ROOT=/home/zeta1996/20221012/unet_deploy/

TRAIN_DATASET=/home/jovyan/datasets/${DATASET_NAME}
VOC_DATASET=/home/jovyan/logs/${DATASET_NAME}
OUTPUT_ROOT=/home/jovyan/logs/${PROJECT_NAME}_${DATASET_NAME}


docker run -it -u 0 --rm \
	--shm-size 4G \
	-v ${ROOT}/model:/home/jovyan/model \
	-v ${ROOT_DATASET}:/home/jovyan/datasets \
	-v ${ROOT}/logs:/home/jovyan/logs \
    -v /etc/localtime:/etc/localtime:ro \
	--gpus all \
 mmsegmentation \
	python /home/jovyan/model/train/train.py --model=${MODEL_NAME}  \
    --input=${TRAIN_DATASET} \
    --voc=${VOC_DATASET} \
    --output=${OUTPUT_ROOT}
