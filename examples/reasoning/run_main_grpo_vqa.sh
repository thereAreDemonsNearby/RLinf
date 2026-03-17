#! /bin/bash
set -x

tabs 4
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TOKENIZERS_PARALLELISM=false
export RAY_DEDUP_LOGS=0

CONFIG_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_PATH=$(dirname $(dirname "$CONFIG_PATH"))
MEGATRON_PATH=/opt/Megatron-LM
# Only transformers lib path is configurable; override with env TRANSFORMERS_LIB_PATH for comparison experiments
TRANSFORMERS_LIB_PATH=/mnt/project_rlinf/shiletong/transformers/src
OTHER_DEPS=/mnt/project_rlinf/shiletong/huggingface_hub-1.0.0rc6:/mnt/project_rlinf/shiletong/sglang-0.5.4-mypatch/

echo "TRANSFORMERS_LIB_PATH: $TRANSFORMERS_LIB_PATH"
echo "OTHER_DEPS: $OTHER_DEPS"
# export VERL_DEP=/mnt/public/shiletong/peft/src:/mnt/public/shiletong/Liger-Kernel/build/lib:/mnt/public/shiletong/mbridge:/mnt/public/shiletong/Megatron-LM-jingxiang:/mnt/project_rlinf/shiletong/qwen_vl_utils-0.0.14/qwen_vl_utils-0.0.14:/mnt/project_rlinf/shiletong/debugpy-1.8.1:/mnt/project_rlinf/shiletong/tensordict-0.8.3:/mnt/project_rlinf/shiletong/codetiming-1.4.0
export PYTHONPATH=${REPO_PATH}:${MEGATRON_PATH}:${TRANSFORMERS_LIB_PATH}:${OTHER_DEPS}:$PYTHONPATH
export RAY_DEBUG=legacy
# export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

if [ -z "$1" ]; then
    CONFIG_NAME="qwen2.5-vl-3b-grpo-fsdp"
else
    CONFIG_NAME=$1
fi

python ${REPO_PATH}/examples/reasoning/main_grpo.py --config-path ${CONFIG_PATH}/config/vqa/  --config-name $CONFIG_NAME
