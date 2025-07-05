from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
import torch
import argparse
import os

def merge_weights(base_model_path, peft_model_path, output_path):
    # Load the base model and merge with PEFT weights
    
    if os.path.exists(output_path):
        print(f"Output path {output_path} already exists. Skipping merge.")
    else:
        print(f"Merging weights from {peft_model_path} into base model {base_model_path}...")
        model = AutoPeftModelForCausalLM.from_pretrained(
            peft_model_path,
            torch_dtype=torch.float16,
            device_map="cpu",
            attn_implementation="eager",
        )
        # Save the merged model
        model = model.merge_and_unload()
        model.save_pretrained(output_path)
        
        tokenizer = AutoTokenizer.from_pretrained(
            model.peft_config["default"].base_model_name_or_path,
            use_fast=True
        )
        tokenizer.save_pretrained(output_path)
        print(f"Merged model saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge base and PEFT weights.")
    parser.add_argument("--base_model", type=str, required=True, help="Path to base model")
    parser.add_argument("--peft_model", type=str, required=True, help="Path to PEFT adapter")
    args = parser.parse_args()

    output = args.peft_model + "/merged"
    merge_weights(args.base_model, args.peft_model, output)