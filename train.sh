
# CUDA_VISISBLE_DEVICES=0,1,2,3 accelerate launch --config_file='configs/accelerator_config.yaml' tangoflux/train.py  --checkpointing_steps="1000" --save_every=5 --config='configs/tangoflux_config.yaml' 


CUDA_VISISBLE_DEVICES=0 accelerate launch --config_file='/home/sang/.cache/huggingface/accelerate/default_config.yaml' tangoflux/train.py  --checkpointing_steps="500" --save_every=5 --config='configs/tangoflux_config.yaml' 
