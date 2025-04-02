export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
accelerate launch --config_file=config/accelerate/zero3.yml --num_processes=8 --main_process_port=29501 finetune.py \
    --config config/finetune/default.yml