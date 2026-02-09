#!/bin/bash
# -*- coding: utf-8 -*-
#
# @author: Yang Liu <yangliu.real@gmail.com>
# @date: 2024/04/03

# Use custom proxy and hosted mirror of HF for model downloading
#export https_proxy=http://127.0.0.1:7890 http_proxy=http://127.0.0.1:7890 all_proxy=socks5://127.0.0.1:7890
# Use custom proxy mirror of the Hugging Face
#export HF_ENDPOINT=https://hf-mirror.com

set -x

#export CUDA_VISIBLE_DEVICES=0
#export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Disable tokenizers parallelism
export TOKENIZERS_PARALLELISM="false"

# Enable NCCL socket communication
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
export CUDA_LAUNCH_BLOCKING=1
export NCCL_IB_DISABLE=1
#export NCCL_NET_GDR_LEVEL=0
#export NCCL_DEBUG=INFO
#export NCCL_DEBUG_SUBSYS=NET


#WORLD_SIZE=1
#WORLD_SIZE=4
WORLD_SIZE=8

MODEL_TYPE=""
MODEL_NAME_OR_PATH=""
RUN_NAME=""
DO_TRAIN="False"
DO_EVAL="False"
DO_PREDICT="False"
UPLOAD_CKPT="False"
ENABLE_DEEPSPEED="False"
FSDP=""
FSDP_WRAP_LAYER=""
ENABLE_DISTRIBUTED_TRAINING="False"
ENABLE_DISTRIBUTED_INFERENCE="False"

# Hyperparameter settings
BATCH_SIZE=
PER_DEVICE_TRAIN_BATCH_SIZE=
PER_DEVICE_EVAL_BATCH_SIZE=
EVAL_SAMPLES=
EPOCHS=
LR=
MAX_SEQ_LEN=
EVAL_TIMES=
LOGGING_STEPS=
GRAD_ACCUMULATION_STEPS=1
TRAIN_ON_INPUT="True"
MASK_TERM="False"
SEQ2SEQ_TRAINING="False"
TRAINABLE_LAYERS="all"

# Path settings
OUTPUT_DIR=""
OUTPUT_FNAME_INFER=""
DATA_PATH_TRAIN=""
DATA_PATH_VALID=""
DATA_PATH_TEST=""
PRED_RESULT_PATH=""
MODEL_CONFIG="run_config/model_config.json"
#DEEPSPEED_CONFIG_PATH="run_config/deepspeed_config_zero1.json"  # ✅
DEEPSPEED_CONFIG_PATH="run_config/deepspeed_config_zero2.json"  # ✅
#DEEPSPEED_CONFIG_PATH="run_config/deepspeed_config_zero3_without_offload.json"  # ✅
#DEEPSPEED_CONFIG_PATH="run_config/deepspeed_config_zero3_with_offload.json"  # ✅
#DEEPSPEED_CONFIG_PATH="run_config/deepspeed_config_zeropp.json"  # ❌

if [ $# -lt 1 ]; then
    echo "[ERROR] Your command line contains $# arguments, please double check it!"
    exit 1
fi

# Generate configs for model & deepspeed
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_type)
            MODEL_TYPE=$2
            shift # past argument
            shift # past value
            ;;
        --model_name_or_path)
            MODEL_NAME_OR_PATH=$2
            shift # past argument
            shift # past value
            ;;
        --run_name)
            RUN_NAME=$2
            shift # past argument
            shift # past value
            ;;
        --do_train)
            DO_TRAIN="True"
            shift # past argument
            ;;
        --do_eval)
            DO_EVAL="True"
            shift # past argument
            ;;
        --do_predict)
            DO_PREDICT="True"
            shift # past argument
            ;;
        --upload_ckpt)
            UPLOAD_CKPT="True"
            shift # past argument
            ;;
        --fsdp)
            FSDP=$2
            shift
            ;;
        --fsdp_wrap_layer)
            FSDP_WRAP_LAYER=$2
            shift
            ;;
        --enable_deepspeed)
            ENABLE_DEEPSPEED="True"
            shift
            ;;
        --enable_distributed_training)
            ENABLE_DISTRIBUTED_TRAINING="True"
            shift
            ;;
        --enable_distributed_inference)
            ENABLE_DISTRIBUTED_INFERENCE="True"
            shift
            ;;
        --data_path_train)
            DATA_PATH_TRAIN=$2
            shift # past argument
            shift # past value
            ;;
        --data_path_valid)
            DATA_PATH_VALID=$2
            shift # past argument
            shift # past value
            ;;
        --data_path_test)
            DATA_PATH_TEST=$2
            shift # past argument
            shift # past value
            ;;
        --pred_result_path)
            PRED_RESULT_PATH=$2
            shift
            shift
            ;;
        -o|--output_dir)
            OUTPUT_DIR=$2
            shift # past argument
            shift # past value
            ;;
        --output_fname_infer)
            OUTPUT_FNAME_INFER=$2
            shift # past argument
            shift # past value
            ;;
        --mask_term)
            MASK_TERM=$2
            shift # past argument
            ;;
        -b|--batch_size)
            BATCH_SIZE=$2
            shift # past argument
            shift # past value
            ;;
        --per_device_train_batch_size)
            PER_DEVICE_TRAIN_BATCH_SIZE=$2
            shift # past argument
            shift # past value
            ;;
        --per_device_eval_batch_size)
            PER_DEVICE_EVAL_BATCH_SIZE=$2
            shift # past argument
            shift # past value
            ;;
        -e|--epochs)
            EPOCHS=$2
            shift # past argument
            shift # past value
            ;;
        --learning_rate)
            LR=$2
            shift # past argument
            shift # past value
            ;;
        -l|--max_seq_len)
            MAX_SEQ_LEN=$2
            shift # past argument
            shift # past value
            ;;
        --dev_set_size)
            DEV_SET_SIZE=$2
            shift # past argument
            shift # past value
            ;;
        --eval_times)
            EVAL_TIMES=$2
            shift # past argument
            shift # past value
            ;;
        --eval_samples)
            EVAL_SAMPLES=$2
            shift # past argument
            shift # past value
            ;;
        --train_on_input)
            TRAIN_ON_INPUT=$2
            shift # past argument
            shift # past value
            ;;
        --seq2seq_training)
            SEQ2SEQ_TRAINING=$2
            shift # past argument
            shift # past value
            ;;
        --trainable_layers)
            TRAINABLE_LAYERS=$2
            shift # past argument
            shift # past value
            ;;
        --logging_steps)
            LOGGING_STEPS=$2
            shift # past argument
            shift # past value
            ;;
        --grad_accumulation_steps)
            GRAD_ACCUMULATION_STEPS=$2
            shift # past argument
            shift # past value
            ;;
        -*|--*)
            echo "[ERROR] Unknown option $1"
            exit 1
            ;;
        *)
            POSITIONAL_ARGS+=("$1") # save positional arg
            shift # past argument
            ;;
    esac
done

JSON_DATA=$(cat <<EOF
{
    "do_train": "${DO_TRAIN}",
    "do_eval": "${DO_EVAL}",
    "do_predict": "${DO_PREDICT}",
    "model_type": "${MODEL_TYPE}",
    "model_name_or_path": "${HDFS_PATH_PREFIX}${MODEL_NAME_OR_PATH}",
    "run_name": "${RUN_NAME}",
    "data_path_train": "${HDFS_PATH_PREFIX}${DATA_PATH_TRAIN}",
    "data_path_valid": "${HDFS_PATH_PREFIX}${DATA_PATH_VALID}",
    "data_path_test": "${HDFS_PATH_PREFIX}${DATA_PATH_TEST}",
    "mask_term": "${MASK_TERM}",
    "output_dir": "${OUTPUT_DIR}",
    "batch_size": ${BATCH_SIZE},
    "per_device_train_batch_size": ${PER_DEVICE_TRAIN_BATCH_SIZE},
    "per_device_eval_batch_size": ${PER_DEVICE_EVAL_BATCH_SIZE},
    "num_epochs": ${EPOCHS},
    "learning_rate": ${LR},
    "cutoff_len": ${MAX_SEQ_LEN},
    "train_on_input": "${TRAIN_ON_INPUT}",
    "seq2seq_training": "${SEQ2SEQ_TRAINING}",
    "trainable_layers": "${TRAINABLE_LAYERS}",
    "eval_times": ${EVAL_TIMES},
    "eval_samples": ${EVAL_SAMPLES},
    "logging_steps": ${LOGGING_STEPS},
    "gradient_accumulation_steps": ${GRAD_ACCUMULATION_STEPS},
    "fsdp": "${FSDP}",
    "fsdp_wrap_layer": "${FSDP_WRAP_LAYER}"
}
EOF
)

echo $JSON_DATA
echo ${JSON_DATA} > ${MODEL_CONFIG}

# Prepare environment
echo "[INFO] Checking essential binaries path..."
echo `which pip`
echo `which conda`
echo `which python`
echo `python --version`
echo "[CMD] cat run_config/model_config.json"
cat ${MODEL_CONFIG}
echo "[CMD] cat run_config/deepspeed_config.json"
cat ${DEEPSPEED_CONFIG_PATH}
echo "[SUCC] Checking essential binaries path success!"

# do training or not
echo "python path: `which python`"
echo "deepspeed path: `which deepspeed`"

# DDP
#OMP_NUM_THREADS=12 /home/liuyang/app/anaconda3/envs/dm/bin/python -m torch.distributed.run --nnodes=1 --nproc_per_node=4 finetune.py --model_config ${MODEL_CONFIG} || exit -1

# DeepSpeed training for models size < 10B
deepspeed --num_gpus=${WORLD_SIZE} finetune.py --model_config ${MODEL_CONFIG} --deepspeed ${DEEPSPEED_CONFIG_PATH} || exit -1

# Run with naive pipeline parallelism implemented in transformers (w/ `device_map="auto"`)
#/home/liuyang/app/anaconda3/envs/dm/bin/python -u finetune.py --model_config ${MODEL_CONFIG} || exit -1
#/home/liuyang/app/anaconda3/envs/dm/bin/python -u finetune.py --model_config ${MODEL_CONFIG} || exit -1

# Resume from specific checkpoint
#/home/liuyang/app/anaconda3/envs/dm/bin/python -u finetune.py --model_config ${MODEL_CONFIG} || exit -1
#/home/liuyang/app/anaconda3/envs/dm/bin/python -u finetune.py --model_config ${MODEL_CONFIG} --resume_from_checkpoint "/mnt/buffer/liuyang/Meta-Llama-3-4x8B-MoE-3D-EX-gate-ffn/Meta-Llama-3-4x8B-MoE-3D-EX/checkpoint-11872" || exit -1

# do prediction or not
 #python -u generate.py --dev_file "<test file path>" --model_type ${MODEL_TYPE} --model_name_or_path ${MODEL_NAME_OR_PATH} --max_length ${MAX_SEQ_LEN} --dev_batch_size ${PER_DEVICE_EVAL_BATCH_SIZE} --output_file ${output_path} > /dev/stdout 2>&1 &

echo "[INFO] All tasks done!"

exit 0
