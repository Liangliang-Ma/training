deepspeed --force_multi --hostfile hostfile_deepspeed --launcher=MPICH --launcher_args='-hostfile hostfile_mpich' scripts/train.py \
--model_path "/scratch/users/maliangl/llama2-70b-fused-qkv-mlperf" \
--dataset_name "tau/scrolls" --dataset_config_name "gov_report" \
--max_seq_len 8192 \
--bf16 True \
--logging_steps 24 \
--eval_steps 48 \
--output_dir "./converge_home" \
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
--max_steps 5 \
--seed 42 \
--lora_target_modules "qkv_proj,o_proj" 2>&1 | tee test_nosync_v2.log

# --lora_alpha 32 \
# --lr_scheduler_type "cosine" 
# --model_path "/scratch/users/maliangl/Llama-2-70b-hf" \
# "/scratch/users/maliangl/save_lora" \
#  "./converge_home"
# /scratch/users/maliangl/llama2-70b-fused-qkv-mlperf
# "/scratch/users/tianmuli/llama2-70b-fused-qkv-mlperf" \