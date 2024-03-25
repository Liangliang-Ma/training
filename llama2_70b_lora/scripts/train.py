# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset
from mlperf_logging_utils import LoraLogger, MLPerfCallback
from transformers import HfArgumentParser, Trainer, TrainingArguments, AutoModelForCausalLM
from utils import create_and_prepare_model, create_datasets, peft_module_casting_to_bf16

import deepspeed
from deepspeed.accelerator import get_accelerator
import os 
import torch
from peft import PeftModel
from peft import LoraConfig, get_peft_model

# distributed setup
local_rank = int(os.getenv("LOCAL_RANK", "0"))
world_size = int(os.getenv("WORLD_SIZE", "1"))

os.environ['CCL_PROCESS_LAUNCHER'] = 'none'
os.environ['CCL_LOCAL_SIZE'] = str(world_size)
os.environ['CCL_LOCAL_RANK'] = str(local_rank)

get_accelerator().set_device(local_rank)
deepspeed.init_distributed()
device = get_accelerator().current_device_name()


@dataclass
class ScriptArguments:
    """
    These arguments vary depending on how many GPUs you have, what their capacity and features are, and what size model you want to train.
    """

    local_rank: Optional[int] = field(
        default=-1, metadata={"help": "Used for multi-gpu"}
    )
    per_device_train_batch_size: Optional[int] = field(default=1)
    per_device_eval_batch_size: Optional[int] = field(default=1)
    gradient_accumulation_steps: Optional[int] = field(default=1)
    learning_rate: Optional[float] = field(default=2e-4)
    max_grad_norm: Optional[float] = field(default=0.0)
    weight_decay: Optional[float] = field(default=0.001)
    lora_alpha: Optional[int] = field(default=32)
    lora_dropout: Optional[float] = field(default=0.1, metadata={"help": "lora dropout is a fixed to 0.1 in closed submission"})
    lora_r: Optional[int] = field(default=16, metadata={"help": "lora rank is a fixed to 16 in closed submission"})
    lora_target_modules: Optional[str] = field(
        default=None,
        metadata={
            "help": "comma separated list of target modules to apply LoRA layers to"
        },
    )
    max_seq_length: Optional[int] = field(default=8192)
    model_path: Optional[str] = field(
        default="./llama-v2-fused-qkv",
        metadata={"help": "Path to the model directory."},
    )
    dataset_name: Optional[str] = field(
        default="tau/scrolls",
        metadata={"help": "The preference dataset to use."},
    )
    config_path: Optional[str] = field(
        default="./configs/default_config.yaml",
        metadata={"help": "path to model config"},
    )
    num_train_epochs: Optional[int] = field(
        default=1,
        metadata={"help": "The number of training epochs for the reward model."},
    )
    fp16: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables fp16 training."},
    )
    bf16: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables bf16 training."},
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True,
        metadata={"help": "Enables gradient checkpointing."},
    )
    optim: Optional[str] = field(
        default="adamw_torch",
        metadata={"help": "The optimizer to use."},
    )
    lr_scheduler_type: str = field(
        default="cosine",
        metadata={
            "help": "Learning rate schedule. Constant a bit better than cosine, and has advantage for analysis"
        },
    )
    max_steps: int = field(
        default=-1, metadata={"help": "How many optimizer update steps to take"}
    )
    warmup_ratio: float = field(
        default=0.03, metadata={"help": "Fraction of steps to do a warmup for"}
    )
    save_steps: int = field(
        default=10, metadata={"help": "Save checkpoint every X updates steps."}
    )
    eval_steps: int = field(default=22, metadata={"help": "Eval model every X steps."})
    logging_steps: int = field(
        default=10, metadata={"help": "Log every X updates steps."}
    )
    target_eval_loss: float = field(
        default=0.92, metadata={"help": "target eval loss - NOT FINAL."}
    )
    output_dir: str = field(
        default="results", metadata={"help": "Where to store the final model."}
    )
    use_flash_attn: Optional[bool] = field(
        default=True,
        metadata={"help": "Enables Flash attention for training."},
    )
    use_peft_lora: Optional[bool] = field(
        default=True,
        metadata={"help": "Enables PEFT LoRA for training."},
    )
    use_gradient_checkpointing: Optional[bool] = field(
        default=True,
        metadata={"help": "Enables Gradient Checkpointing."},
    )
    push_to_hub: Optional[bool] = field(
        default=False,
        metadata={"help": "If True, pushes the model to the HF Hub"},
    )
    num_workers: int = field(
        default=4, metadata={"help": "Number of dataset workers to use."}
    )
    debug: Optional[bool] = field(
        default=False,
        metadata={
            "help": "If True, tests things like proper saving/loading/logging of model"
        },
    )
    dataset_config_name: Optional[str] = field(default="gov_report")
    hub_model_id: Optional[str] = field(default=None)
    seed: Optional[int] = field(default=42)


def main(args):
    loralogger = LoraLogger(target_eval_loss=args.target_eval_loss)
    training_arguments = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        optim=args.optim,
        learning_rate=args.learning_rate,
        fp16=args.fp16,
        bf16=args.bf16,
        max_grad_norm=args.max_grad_norm,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        num_train_epochs=args.num_train_epochs,
        evaluation_strategy="steps",
        save_strategy="no",
        max_steps=args.max_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        push_to_hub=args.push_to_hub,
        gradient_checkpointing=args.use_gradient_checkpointing,
        hub_model_id=args.hub_model_id,
        report_to="tensorboard",
        seed=args.seed,
        deepspeed="ds_config.json"
    )

    model, peft_config, tokenizer = create_and_prepare_model(args)
    model.config.use_cache = False

    #load
    # savelora_path = "/scratch/users/maliangl/save_lora"
    # model.load_adapter(args.output_dir)
    # model.enable_adapters()
    
    # model = PeftModel.from_pretrained(model, "./converge_home", is_trainable=True)

    # datasets
    ## ToDo uncomment once drive goes public
    # train_url = "https://drive.google.com/file/d/1-JgY1mEafcJ7qhggt6UR3OEKAciIPd5s/view?usp=sharing"
    # eval_url =  "https://drive.google.com/file/d/1jrm6Lacrq49AYv0uB_Qy22xRmfPixQvs/view?usp=sharing"
    # dataset = load_dataset("parquet", data_files={'train': train_url, 'validation': eval_url})
    # dataset = load_dataset(
    #     "parquet",
    #     data_files={
    #         "train": f"{args.dataset_path}/train-00000-of-00001.parquet",
    #         "validation": f"{args.dataset_path}/validation-00000-of-00001.parquet",
    #     },
    # )
     # datasets
    train_dataset, eval_dataset = create_datasets(tokenizer, args)
    # train_dataset, eval_dataset = dataset["train"], dataset["validation"]

    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        # wtf="wtf"
        # callbacks=[MLPerfCallback(loralogger, len(train_dataset), len(eval_dataset),args.lora_alpha)],
    )
    trainer.accelerator.print(f"{trainer.model}")
    # if args.use_peft_lora:
    #     trainer.model.print_trainable_parameters()

    if args.use_peft_lora:
        peft_module_casting_to_bf16(trainer.model, args)
    print("prepare all done!")
    trainer.train()

    # save_path = "/scratch/users/maliangl/save"
    # # savelora_path = "/scratch/users/maliangl/save_lora_1"
    
    # # model.save_pretrained(save_path)
    # merged_model = model.float32().merge_and_unload()
    # merged_model.save_pretrained(save_path)
    # model.save_pretrained(savelora_path, save_adapter=True, save_config=True)
 
    # trainer.save_model(save_path) # will only save from the main process
    # from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
    # from peft import get_peft_model_state_dict
    # if local_rank == 0:
    #     print(model.state_dict().keys())
    #     if trainer.deepspeed and not os.path.exists(save_path + "/pytorch_model.bin"):
    #         print("CONVERT Deepspeed Checkpoint to FP32")
    #         state_dict = get_fp32_state_dict_from_zero_checkpoint(save_path) # already on cpu
    #     else:
    #         print("TRY to use the model directly")
    #         state_dict = model.cpu().state_dict()
    #     print("Number of elements in the state dict", sum(p.numel() for p in state_dict.values()))
    #     d = get_peft_model_state_dict(model, state_dict=state_dict)

    #     model.save_pretrained(savelora_path)
    #     torch.save(d, savelora_path + "/adapter_model.bin")

    # from deepspeed.utils import safe_get_full_fp32_param
    # output_state_dict = {
    #     k: safe_get_full_fp32_param(v).cpu() for k, v in model.named_parameters()
    # }

    # if get_local_rank() != 0:
    #     return

    # torch.save(output_state_dict, os.path.join(save_path, WEIGHTS_NAME))
    # if trainer.args.process_index == 0:
    trainer.accelerator.wait_for_everyone()
    state_dict = trainer.accelerator.get_state_dict(trainer.deepspeed)
    if trainer.accelerator.is_main_process:
        unwrapped_model = trainer.accelerator.unwrap_model(
            trainer.deepspeed
        )
        unwrapped_model.save_pretrained(args.output_dir, state_dict=state_dict)


    print("save all done!")
    # new_model, peft_config, tokenizer = create_and_prepare_model(args)
    # # model = AutoModelForCausalLM.from_pretrained(save_path)
    # model_to_merge = PeftModel.from_pretrained(new_model, savelora_path)
    import datetime
    datetime.datetime.now()
    print("all done!!")

if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]
    main(args)
