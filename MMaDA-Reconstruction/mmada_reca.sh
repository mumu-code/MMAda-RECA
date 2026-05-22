accelerate launch --config_file /home/xl/project/MaskGRPO/accelerate_configs/1_gpu.yaml --main_process_port=8888 training/train_mmada_reca.py config=./configs/sft.yaml



accelerate launch --config_file /home/xl/project/MaskGRPO/accelerate_configs/1_node_4_gpus_deepspeed_zero2.yaml --main_process_port=8888 training/train_mmada_reca.py config=./configs/reca.yaml