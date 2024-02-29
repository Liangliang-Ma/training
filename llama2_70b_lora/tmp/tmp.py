from datasets import load_dataset
from transformers import AutoTokenizer
from functools import partial
from itertools import chain
seq_len = 8192

def group_texts(examples, block_size):
    # Concatenate all texts.
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
    # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    if "labels" not in result:
        result["labels"] = result["input_ids"].copy()
    return result

def create_datasets(tokenizer):
    dataset = load_dataset(
        "tau/scrolls",
        "gov_report",
        use_auth_token=True,
        num_proc=8,
    )
    train_dataset = dataset["train"]
    valid_dataset = dataset["validation"]
    column_names = train_dataset.features

    def tokenize_function(example, eval=False):
        output_texts = []
        mask_labels_sizes = []
        for i in range(len(example["input"])):
            if "gov_report" in "gov_report":
                output_texts.append(
                    f"### Summarize the following text:\n {example['input'][i]}\n ### Summary:\n {example['output'][i]}{tokenizer.eos_token}"
                )
                if eval:
                    mask_labels_sizes.append(
                        f"### Summarize the following text:\n {example['input'][i]}\n ### Summary:\n"
                    )
            else:
                output_texts.append(
                    f"### {example['input'][i]}\n ### The answer is:\n {example['output'][i]}{tokenizer.eos_token}"
                )

        input_ids = tokenizer(output_texts).input_ids

        if eval:
            labels_ids = tokenizer(mask_labels_sizes).input_ids
            masked_labels = []
            for out, lb in zip(input_ids, labels_ids):
                ml = out.copy()
                ml[: len(lb)] = [-100] * len(lb)
                ml[-1] = -100
                masked_labels.append(ml)
            return {"input_ids": input_ids, "labels": masked_labels}
        else:
            return {"input_ids": input_ids}

    train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        num_proc=8,
        remove_columns=column_names,
    )
    valid_dataset = valid_dataset.map(
        partial(tokenize_function, eval=True),
        batched=True,
        num_proc=2,
        remove_columns=column_names,
    )

    def filter_function(example):
        to_keep = []
        for i in range(len(example["input_ids"])):
            if len(example["input_ids"][i]) > seq_len:
                to_keep.append(False)
            else:
                to_keep.append(True)
        return to_keep

    train_dataset = train_dataset.filter(
        filter_function,
        batched=True,
        # with_indices=True,
        num_proc=8,
        # remove_columns=column_names,
    )
    valid_dataset = valid_dataset.filter(
        filter_function,
        batched=True,
        # with_indices=True,
        num_proc=2,
        # remove_columns=column_names,
    )
    print(
        f"Before packing, Size of the train set: {len(train_dataset)}. Size of the validation set: {len(valid_dataset)}"
    )

    packing_method = partial(group_texts, block_size=seq_len)
    # Packing
    train_dataset = train_dataset.map(
        packing_method,
        batched=True,
        num_proc=8,
    )
    valid_dataset = valid_dataset.map(
        packing_method,
        batched=True,
        num_proc=2,
    )

    print(
        f"Size of the train set: {len(train_dataset)}. Size of the validation set: {len(valid_dataset)}"
    )

    return train_dataset, valid_dataset

model_path = "/scratch/users/maliangl/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
train_dataset, valid_dataset = create_datasets(tokenizer)