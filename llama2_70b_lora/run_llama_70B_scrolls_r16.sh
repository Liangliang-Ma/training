deepspeed --force_multi --hostfile hostfile_deepspeed --launcher=MPICH --launcher_args='-hostfile hostfile_mpich' scripts/train.py \
--model_path /scratch/users/maliangl/Llama-2-70b-hf \
--dataset_name "tau/scrolls" --dataset_config_name "gov_report" \
--max_seq_len 2048 \
--bf16 True \
--logging_steps 32 \
--eval_steps 64 \
--output_dir "./results/llama-70b_scrolls_gov_report_r16_666" \
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 1 \
--lr_scheduler_type "cosine" \
--learning_rate 5e-4 \
--warmup_ratio 0 \
--use_gradient_checkpointing True \
--target_eval_loss 0.925 \
--use_peft_lora True \
--lora_r 16 \
--lora_alpha 16 \
--lora_dropout 0.1 \
--max_steps 800 \
--seed 666 \
--lora_target_modules "qkv_proj,o_proj" 2>&1 | tee test.log

# --dataset_text_field "input" \
# --max_seq_len 8192 \
# --use_flash_attn \