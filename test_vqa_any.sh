# cd /mnt/public/shiletong/RLinf_ppo_new
# rm ray_utils/ray_head_ip.txt
# Usage: bash test_vqa_any.sh <num_gpu> <config> [transformers_path]
#   transformers_path: optional; colon-separated PYTHONPATH for transformers lib.
#   If omitted, uses default from run_main_grpo_vqa.sh.
#   Example: bash test_vqa_any.sh 8 qwen3-vl-2b-grpo-fsdp-geo3k /path/to/transformers/build/lib:/other/deps
export TOKENIZERS_PARALLELISM=false
bash ray_utils/start_ray.sh

num_gpu=$1
config=$2
transformers_lib_path=$3
echo "number of gpus: $num_gpu, the config file name $config"
[[ -n "$transformers_lib_path" ]] && echo "using custom TRANSFORMERS_LIB_PATH: $transformers_lib_path"

if [[ -z $num_gpu ]]; then
    echo "please pass num gpus in"
    exit 1
fi

if [[ -z $config ]]; then
    echo "please pass config file in"
    exit 1
fi
# exit 0

if [ "$RANK" -eq 0 ]; then
    bash ray_utils/check_ray.sh $num_gpu
    [[ -n "$transformers_lib_path" ]] && export TRANSFORMERS_LIB_PATH="$transformers_lib_path"
    bash examples/reasoning/run_main_grpo_vqa.sh  $config
else
    sleep 10d
    rm ray_utils/ray_head_ip.txt;
fi
# 运行环境初始化
sleep 10d
