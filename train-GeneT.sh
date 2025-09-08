#!/bin/bash

# 默认参数设置
DEFAULT_NPROC_PER_NODE=8
DEFAULT_MASTER_PORT=29500
DEFAULT_CUDA_DEVICES="0,1,2,3,4,5,6,7"
DEFAULT_MODEL_TYPE="qwen1half-0_5b"
DEFAULT_MODEL_PATH="qwen0.5b/"
DEFAULT_OUTPUT_DIR="GeneT-qwen0.5b-public-mode/"
DEFAULT_TRAIN_DATA="train.jsonl"
DEFAULT_VAL_DATA="test.jsonl"
DEFAULT_NUM_EPOCHS=5
DEFAULT_MAX_LENGTH=32768
DEFAULT_BATCH_SIZE=1
DEFAULT_LEARNING_RATE="1e-5"
DEFAULT_DEEPSPEED="default-zero3"

# 显示帮助信息
show_help() {
    echo "用法: $0 [选项]"
    echo "选项:"
    echo "  --nproc_per_node=<值>        每个节点的进程数 (默认: $DEFAULT_NPROC_PER_NODE)"
    echo "  --master_port=<值>           主节点端口 (默认: $DEFAULT_MASTER_PORT)"
    echo "  --cuda_devices=<值>          使用的GPU设备ID (默认: $DEFAULT_CUDA_DEVICES)"
    echo "  --model_type=<值>            模型类型 (默认: $DEFAULT_MODEL_TYPE)"
    echo "  --model_path=<路径>          模型路径 (默认: $DEFAULT_MODEL_PATH)"
    echo "  --output_dir=<路径>          输出目录 (默认: $DEFAULT_OUTPUT_DIR)"
    echo "  --train_data=<路径>          训练数据路径 (默认: $DEFAULT_TRAIN_DATA)"
    echo "  --val_data=<路径>            验证数据路径 (默认: $DEFAULT_VAL_DATA)"
    echo "  --num_epochs=<值>            训练轮数 (默认: $DEFAULT_NUM_EPOCHS)"
    echo "  --max_length=<值>            最大长度 (默认: $DEFAULT_MAX_LENGTH)"
    echo "  --batch_size=<值>            批次大小 (默认: $DEFAULT_BATCH_SIZE)"
    echo "  --learning_rate=<值>         学习率 (默认: $DEFAULT_LEARNING_RATE)"
    echo "  --deepspeed=<值>             DeepSpeed配置 (默认: $DEFAULT_DEEPSPEED)"
    echo "  --help                       显示此帮助信息"
    exit 0
}

# 解析命令行参数
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --nproc_per_node=*) NPROC_PER_NODE="${1#*=}" ;;
        --master_port=*) MASTER_PORT="${1#*=}" ;;
        --cuda_devices=*) CUDA_DEVICES="${1#*=}" ;;
        --model_type=*) MODEL_TYPE="${1#*=}" ;;
        --model_path=*) MODEL_PATH="${1#*=}" ;;
        --output_dir=*) OUTPUT_DIR="${1#*=}" ;;
        --train_data=*) TRAIN_DATA="${1#*=}" ;;
        --val_data=*) VAL_DATA="${1#*=}" ;;
        --num_epochs=*) NUM_EPOCHS="${1#*=}" ;;
        --max_length=*) MAX_LENGTH="${1#*=}" ;;
        --batch_size=*) BATCH_SIZE="${1#*=}" ;;
        --learning_rate=*) LEARNING_RATE="${1#*=}" ;;
        --deepspeed=*) DEEPSPEED="${1#*=}" ;;
        --help) show_help ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
    shift
done

# 设置参数值（使用默认值或用户提供的值）
NPROC_PER_NODE=${NPROC_PER_NODE:-$DEFAULT_NPROC_PER_NODE}
MASTER_PORT=${MASTER_PORT:-$DEFAULT_MASTER_PORT}
CUDA_DEVICES=${CUDA_DEVICES:-$DEFAULT_CUDA_DEVICES}
MODEL_TYPE=${MODEL_TYPE:-$DEFAULT_MODEL_TYPE}
MODEL_PATH=${MODEL_PATH:-$DEFAULT_MODEL_PATH}
OUTPUT_DIR=${OUTPUT_DIR:-$DEFAULT_OUTPUT_DIR}
TRAIN_DATA=${TRAIN_DATA:-$DEFAULT_TRAIN_DATA}
VAL_DATA=${VAL_DATA:-$DEFAULT_VAL_DATA}
NUM_EPOCHS=${NUM_EPOCHS:-$DEFAULT_NUM_EPOCHS}
MAX_LENGTH=${MAX_LENGTH:-$DEFAULT_MAX_LENGTH}
BATCH_SIZE=${BATCH_SIZE:-$DEFAULT_BATCH_SIZE}
LEARNING_RATE=${LEARNING_RATE:-$DEFAULT_LEARNING_RATE}
DEEPSPEED=${DEEPSPEED:-$DEFAULT_DEEPSPEED}

# 计算梯度累积步数
GRADIENT_ACCUMULATION_STEPS=$((16 / NPROC_PER_NODE))

# 执行训练命令
NPROC_PER_NODE=$NPROC_PER_NODE \
MASTER_PORT=$MASTER_PORT \
CUDA_VISIBLE_DEVICES=$CUDA_DEVICES \
swift sft \
    --model_type $MODEL_TYPE \
    --model_id_or_path $MODEL_PATH \
    --check_model_is_latest false \
    --output_dir $OUTPUT_DIR \
    --sft_type full \
    --template_type qwen \
    --dtype bf16 \
    --ddp_backend nccl \
    --custom_train_dataset_path $TRAIN_DATA \
    --custom_val_dataset_path $VAL_DATA \
    --train_dataset_sample -1 \
    --num_train_epochs $NUM_EPOCHS \
    --max_length $MAX_LENGTH \
    --check_dataset_strategy warning \
    --gradient_checkpointing true \
    --batch_size $BATCH_SIZE \
    --weight_decay 0.1 \
    --learning_rate $LEARNING_RATE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --max_grad_norm 0.5 \
    --warmup_ratio 0.03 \
    --eval_steps 5000 \
    --save_steps 5000 \
    --save_total_limit 20 \
    --logging_steps 10 \
    --use_flash_attn true \
    --deepspeed $DEEPSPEED \
    --save_strategy epoch \
    --evaluation_strategy epoch \
    --save_only_model true

sleep 5
echo "训练完成!"
