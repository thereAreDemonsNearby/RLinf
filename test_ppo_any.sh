# cd /mnt/public/shiletong/RLinf_ppo_new
# rm ray_utils/ray_head_ip.txt
export TOKENIZERS_PARALLELISM=false
bash ray_utils/start_ray.sh

num_gpu=$1
config=$2
echo "number of gpus: $num_gpu, the config file name $config"

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
    # bash examples/reasoning/run_main_grpo_math.sh;
    # bash examples/reasoning/run_main_ppo_math.sh qwen2.5-1.5b-ppo-megatron-dynamicbatch.yaml
    bash examples/reasoning/run_main_ppo_math.sh $config
else
    sleep 10d
    rm ray_utils/ray_head_ip.txt;
fi
# 运行环境初始化
sleep 10d
