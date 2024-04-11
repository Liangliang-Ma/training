export DS_SYNC=1
# export CCL_OP_SYNC=0
# # export CCL_SAME_STREAM=1
# # export CCL_BLOCKING_WAIT=0
export CCL_SKIP_SCHEDULER=1
export CCL_ALLGATHERV_MEDIUM_SIZE_THRESHOLD=0
export TORCH_LLM_ALLREDUCE=1
#export ONECCL_BINDINGS_FOR_PYTORCH_ENV_VERBOSE=1
# export UR_L0_IN_ORDER_BARRIER_BY_SIGNAL=0
# export IPEX_ZE_TRACING=1

deepspeed --force_multi --hostfile hostfile_deepspeed --launcher=MPICH --launcher_args='-hostfile hostfile_mpich' scripts/train.py \
--model_path "/mllnvme0/llama2-7b" \
--dataset_name "tau/scrolls" --dataset_config_name "gov_report" \
--max_seq_len 2048 \
--bf16 True \
--logging_steps 24 \
--eval_steps 48 \
--output_dir "./output" \
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 1 \
--lr_scheduler_type "cosine" \
--learning_rate 4e-4 \
--weight_decay 0.0001 \
--max_grad_norm 0.3 \
--warmup_ratio 0 \
--use_gradient_checkpointing True \
--target_eval_loss 0.925 \
--use_peft_lora True \
--lora_r 16 \
--lora_alpha 32 \
--lora_dropout 0.1 \
--max_steps 10 \
--seed 42 \
--lora_target_modules "qkv_proj,o_proj" 2>&1 | tee log/test_unsetopsync.log

# --lora_alpha 32 \
# --lr_scheduler_type "cosine" 
# --model_path "/scratch/users/maliangl/Llama-2-70b-hf" \
# "/scratch/users/maliangl/save_lora" \
#  "./converge_home"
# /scratch/users/maliangl/llama2-70b-fused-qkv-mlperf
# "/scratch/users/tianmuli/llama2-70b-fused-qkv-mlperf" \
# export UR_L0_IN_ORDER_BARRIER_BY_SIGNAL=0
# /mllnvme0/llama2-7b
# "/mllnvme0/llama2"
# mpirun -np 8 -ppn 8 python scripts/train.py \

# mpirun -n 8 -ppn 8 -hostfile hostfile_mpich \
# -genv PYTHONPATH=/home/mll/nightly/main/training/llama2_70b_lora \
# -genv MASTER_ADDR 127.0.0.1 -genv MASTER_PORT 29500 \
# -genv WORLD_SIZE 8 -genv LOCAL_SIZE 8 -hosts localhost \
# /home/mll/pti-gpu/tools/unitrace/build/unitrace --chrome-kernel-logging \
# /home/mll/miniconda3/envs/mlp/bin/python -u /home/mll/nightly/main/training/llama2_70b_lora/DeepSpeed/deepspeed/launcher/launcher_helper.py \
# --launcher mpich scripts/train.py \
# --model_path "/mllnvme0/llama2-7b" \
# --dataset_name "tau/scrolls" --dataset_config_name "gov_report" \
# --max_seq_len 2048 \
# --bf16 True \
# --logging_steps 24 \
# --eval_steps 48 \
# --output_dir "./output" \
# --per_device_train_batch_size 1 \
# --gradient_accumulation_steps 1 \
# --lr_scheduler_type "cosine" \
# --learning_rate 4e-4 \
# --weight_decay 0.0001 \
# --max_grad_norm 0.3 \
# --warmup_ratio 0 \
# --use_gradient_checkpointing True \
# --target_eval_loss 0.925 \
# --use_peft_lora True \
# --lora_r 16 \
# --lora_alpha 32 \
# --lora_dropout 0.1 \
# --max_steps 5 \
# --seed 42 \
# --lora_target_modules "qkv_proj,o_proj" 2>&1 | tee log/test.log

# /home/mll/pti-gpu/tools/unitrace/build/unitrace --chrome-call-logging --chrome-kernel-logging --chrome-dnn-logging --chrome-sycl-logging --chrome-ccl-logging --conditional-collection --output perf/god.json \
# /home/mll/pti-gpu/tools/onetrace/build/onetrace --chrome-call-logging \
