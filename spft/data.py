import torch


def generate_prompt_csr170k(data_point):
    # sorry about the formatting disaster gotta move fast
    return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.  

                ### Instruction:
                {data_point["instruction"]}
                
                ### Response:
                {data_point["answer"]}"""  # noqa: E501
    
def generate_prompt_math10k(data_point):
    # sorry about the formatting disaster gotta move fast
    return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.  

                ### Instruction:
                {data_point["instruction"]}
                
                ### Response:
                {data_point["output"]}"""  # noqa: E501
                
def generate_prompt_ultrachat(data_point):
    # sorry about the formatting disaster gotta move fast
    return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.  

                ### Instruction:
                {data_point["instruction"]}
                
                ### Response:
                {data_point["output"]}"""  # noqa: E501
                
def generate_prompt_codefeedback(data_point):
    # sorry about the formatting disaster gotta move fast
    # return f"""Please provide a self-contained Python script that solves the following problem in a markdown code block:
    #             {data_point["x"]}
                
    #             Below is a Python script with a self-contained function that solves the problem and passes corresponding tests:
    #             {data_point["y"]}"""  # noqa: E501
    return f"""{data_point["x"]} {data_point["y"]}"""  # noqa: E501

def tokenize(tokenizer, prompt, add_eos_token=True):
    # there's probably a way to do this with the tokenizer settings
    # but again, gotta move fast
    cutoff_len = 256

    result = tokenizer(
        prompt,
        truncation=True,
        max_length=cutoff_len,
        padding=True,
        return_tensors=None,
    )
    if result["input_ids"][-1] != tokenizer.eos_token_id and len(result["input_ids"]) < cutoff_len and add_eos_token:
        result["input_ids"].append(tokenizer.eos_token_id)

    result["labels"] = result["input_ids"].copy()

    return {"input_ids": torch.as_tensor(result["input_ids"]), "labels": torch.as_tensor(result["labels"])}


def tokenizeCode(tokenizer, prompt, seq_len=2048, add_eos_token=True):
    # there's probably a way to do this with the tokenizer settings
    # but again, gotta move fast
    cutoff_len = seq_len - 1  # -1 for the eos token

    result = tokenizer(
        prompt,
        truncation=True,
        max_length=cutoff_len,
        padding=True,
        return_tensors=None,
    )
    if result["input_ids"][-1] != tokenizer.eos_token_id and add_eos_token:
        result["input_ids"].append(tokenizer.eos_token_id)

    result["labels"] = result["input_ids"].copy()

    return {"input_ids": torch.as_tensor(result["input_ids"]), "labels": torch.as_tensor(result["labels"])}
    # return {"input_ids": torch.as_tensor(result["input_ids"]), "labels": torch.as_tensor(result["labels"])}


def generate_and_tokenize_prompt_csr170k(tokenizer, data_point):
    full_prompt = generate_prompt_csr170k(data_point)
    tokenized_full_prompt = tokenize(tokenizer, full_prompt)

    user_prompt = generate_prompt_csr170k({**data_point, "answer": ""})
    tokenized_user_prompt = tokenize(tokenizer, user_prompt, add_eos_token=False)

    user_prompt_len = len(tokenized_user_prompt["input_ids"])
    tokenized_full_prompt["labels"][:user_prompt_len] = -100
    return tokenized_full_prompt

def generate_and_tokenize_prompt_math10k(tokenizer, data_point):
    full_prompt = generate_prompt_math10k(data_point)
    tokenized_full_prompt = tokenize(tokenizer, full_prompt)

    user_prompt = generate_prompt_math10k({**data_point, "output": ""})
    tokenized_user_prompt = tokenize(tokenizer, user_prompt, add_eos_token=False)

    user_prompt_len = len(tokenized_user_prompt["input_ids"])
    tokenized_full_prompt["labels"][:user_prompt_len] = -100
    return tokenized_full_prompt


def generate_and_tokenize_prompt_ultrachat(tokenizer, seq_len, data_point):
    full_prompt = generate_prompt_ultrachat(data_point)
    tokenized_full_prompt = tokenizeCode(tokenizer, full_prompt, seq_len=seq_len)

    user_prompt = generate_prompt_ultrachat({**data_point, "output": ""})
    tokenized_user_prompt = tokenizeCode(tokenizer, user_prompt, seq_len=seq_len, add_eos_token=False)

    user_prompt_len = len(tokenized_user_prompt["input_ids"])
    tokenized_full_prompt["labels"][:user_prompt_len] = -100
    return tokenized_full_prompt

def generate_and_tokenize_prompt_codefeedback(tokenizer, seq_len, data_point):
    # print(data_point)
    full_prompt = generate_prompt_codefeedback(data_point)
    tokenized_full_prompt = tokenizeCode(tokenizer, full_prompt, seq_len=seq_len)

    user_prompt = generate_prompt_codefeedback({**data_point, "y": ""})
    tokenized_user_prompt = tokenizeCode(tokenizer, user_prompt, seq_len=seq_len, add_eos_token=False)

    user_prompt_len = len(tokenized_user_prompt["input_ids"])
    tokenized_full_prompt["labels"][:user_prompt_len] = -100
    return tokenized_full_prompt


#* Dummy Dataset Collection:

DATA_COLLECTION = {
    "CodeFeedback-Filtered-Instruction": (1024, 512),
    "WizardLM_evol_instruct_70k": (2048, 512),
    "math_10k": (512, 128),
    "commonsense_170k": (512, 3),
    "commonsense_15k": (512, 3),
}
    
def generate_and_tokenize_prompt_simulated(seq_len=512, out_len=3):
    """ Generates a dummy input_ids and labels for testing purposes."""
    input_ids = list(range(100, 100 + seq_len))  # Dummy token ids
    ctx_len = seq_len - out_len  # Length of the context part
    labels = [-100] * ctx_len+ input_ids[ctx_len:]

    assert len(input_ids) == seq_len
    assert len(labels) == seq_len

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": [1] * seq_len,
    }