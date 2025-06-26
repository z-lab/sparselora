from spft.train.args import DataTrainingArguments, ModelArguments, TrainingArguments
from peft import LoraConfig, get_peft_model
import torch
from liger_kernel.transformers import apply_liger_kernel_to_llama
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from .io import rank0_print

def create_model_and_tokenizer(model_args: ModelArguments, data_args: DataTrainingArguments, training_args: TrainingArguments):
    """
    Create and prepare the model and tokenizer.
    """
    
    if training_args.enable_unsloth:
        return unsloth_initialize(model_args, data_args, training_args)
    
    apply_liger_kernel_to_llama(
            rope=True,
            swiglu=False,
            cross_entropy=True,
            fused_linear_cross_entropy=False,
            rms_norm=True
        )
        
    # Set up model
    if training_args.peft == "qlora":
        print("executing qlora")
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,

            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4'
            ),
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
        )
    
    # Set up tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=data_args.model_max_length,
        padding_side="left",
        use_fast=False,
    )
    
    if len(tokenizer) > 32000: #* Llama3
        print("Using LLaMA 3 tokenizer")
        tokenizer.pad_token = "<|reserved_special_token_0|>"
        tokenizer.pad_token_id = 128002
    
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    if model_args.chat_template_format != "none":
        rank0_print(f"Using custom chat template format: {model_args.chat_template_format}")
        tokenizer.chat_template = model_args.chat_template_format
    
    # Set up PEFT
    if training_args.peft is not None:
        if training_args.peft == "lora" or "qlora" in training_args.peft:
            config = LoraConfig(
                r=training_args.lora_r,
                target_modules=training_args.lora_target_modules.split(","),
                lora_alpha=training_args.lora_alpha,
                lora_dropout=training_args.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
        elif training_args.peft == "dora":
            config = LoraConfig(
                r=training_args.lora_r,
                target_modules=training_args.lora_target_modules.split(","),
                lora_alpha=training_args.lora_alpha,
                lora_dropout=training_args.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                use_dora=True,
            )
        else:
            raise ValueError(f"Unsupported PEFT method: '{training_args.peft}'")
        model = get_peft_model(model, config)
        
    return model, tokenizer


def unsloth_initialize(model_args: ModelArguments, data_args: DataTrainingArguments, training_args: TrainingArguments):
    from unsloth import FastLanguageModel
    
    model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = model_args.model_name_or_path,
            dtype = torch.bfloat16,
            max_seq_length = data_args.model_max_length,
            load_in_4bit=False,
    )
    
    if len(tokenizer) > 32000: #* Llama3
        print("Using LLaMA 3 tokenizer")
        tokenizer.pad_token = "<|reserved_special_token_0|>"
        tokenizer.pad_token_id = 128002
    
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        
    if model_args.chat_template_format != "none":
        rank0_print(f"Using custom chat template format: {model_args.chat_template_format}")
        tokenizer.chat_template_format = model_args.chat_template_format

    if training_args.peft is not None:
        model = FastLanguageModel.get_peft_model(
            model,
            r=training_args.lora_r,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",],
            lora_alpha=training_args.lora_alpha,
            lora_dropout=training_args.lora_dropout,
            bias = "none",
            use_rslora = False,  # We support rank stabilized LoRA
            loftq_config = None, # And LoftQ
            use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        )
    return model, tokenizer