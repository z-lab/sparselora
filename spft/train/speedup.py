try:
    import unsloth
except ImportError:
    pass

import torch
from torch import distributed as dist
import transformers
from transformers import HfArgumentParser, set_seed
from spft.api import SPFTConfig, get_spft_callback, get_spft_model
from spft.callbacks import EvaluateFirstStepCallback
from spft.train.args import DataTrainingArguments, ModelArguments, TrainingArguments
from spft.utils.io import build_runname
from spft.utils.model import create_model_and_tokenizer
import yaml
import sys
import os
import subprocess
from tools.utils.timer_util import Timers, TimedTrainer
from spft.data import generate_and_tokenize_prompt_simulated, DATA_COLLECTION
from datasets import Dataset
    

def run_warmup_and_benchmark(model, training_args, dataset, collator, callbacks):
    training_args.max_steps = 20
    trainer = TimedTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        callbacks=callbacks,
    )
    trainer.train()
    
    ##* Full Training:
    training_args.max_steps = 50
    training_args.log_level="error"
    trainer = TimedTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        callbacks=callbacks,
        
    )
    timers = Timers()
    timers(f"model").reset()
    timers(f"model").start()
    _ = trainer.train()
    timers(f"model").stop()
    
    
    try:
        avg_fwd = sum(trainer.forward_times) / len(trainer.forward_times)
        avg_bwd = sum(trainer.backward_times) / len(trainer.backward_times)
        print(f"[Local] Avg Forward: {avg_fwd:.2f} ms | Avg Backward: {avg_bwd:.2f} ms")
    except:
        pass        
    
    TOTAL_TIME = timers("model").elapsed(mode="sum") / training_args.max_steps
    print(f"[Local] Time (per step): {TOTAL_TIME:.2f}ms/step")
    
    exit(0)  # Exit after warmup and benchmark to avoid training
   
 

def main(model_args: ModelArguments, data_args: DataTrainingArguments, training_args: TrainingArguments, cli_keys: set) -> None:    # Set up SPFT
    assert training_args.spft is not None, "SPFT is not enabled. Please set --spft to a valid path."
    spft_config = SPFTConfig.from_file(training_args.spft)
    spft_config.update([model_args, data_args, training_args], prefix="spft_", cli_keys=cli_keys)
    training_args.run_name = build_runname(training_args, data_args, spft_config)
    
    peft_name = "none" if training_args.peft is None else f"{training_args.peft}"
    base_path = f"{training_args.output_dir}/{peft_name}/{training_args.run_name}/"
    training_args.output_dir = f"{base_path}{training_args.seed}/"
    spft_config.write_out(training_args.output_dir)
    
    
    print("Launching Model: ", model_args.model_name_or_path)
    
    #* Create model + LoRA
    model, tokenizer = create_model_and_tokenizer(model_args, data_args, training_args)
    
    callbacks = []
    model = get_spft_model(model, spft_config, enable_unsloth=training_args.enable_unsloth)
    callbacks.append(get_spft_callback(spft_config))
    model = model.to(torch.bfloat16)
    
    model.print_trainable_parameters()
    # Load dataset
    
    print(f"[INFO] Loading dataset: {data_args.dataset}, max_length: {data_args.model_max_length}, max_seq_length: {data_args.max_seq_length}, batch_size: {training_args.per_device_train_batch_size}")
    
    # Simulate dummy datapoints
    dataset_name = data_args.dataset.split("/")[-1].split(".")[0]
    seq_len, out_len = DATA_COLLECTION.get(dataset_name, (512, 3))
    print(f"Generating simulated data for {dataset_name} with seq_len={seq_len} and out_len={out_len}")
    samples = [generate_and_tokenize_prompt_simulated(seq_len=seq_len, out_len=out_len) for _ in range(10000)]

    # Convert to HuggingFace Dataset
    data_dict = {
        "train": Dataset.from_list(samples),
        "eval": None  # No eval set for this simulation
    }
    
    if data_dict["eval"] is not None:
        callbacks.append(EvaluateFirstStepCallback())

    # Set up data collator
    collator = transformers.DataCollatorForSeq2Seq(
        tokenizer,
        pad_to_multiple_of=512,
        return_tensors="pt",
        padding=True,
    )
     
    run_warmup_and_benchmark(
        model=model,
        training_args=training_args,
        dataset=data_dict["train"],
        collator=collator,
        callbacks=callbacks,
    )
    

if __name__ == "__main__":
    
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    
    # Get CLI args (excluding the script name)
    raw_cli_args = sys.argv[1:]
    # Collect all explicitly passed --spft_ keys
    cli_spft_keys = {
        arg.lstrip('-').replace('-', '_')
        for arg in raw_cli_args
        if arg.startswith('--spft_')
    }
    print(f"Raw CLI args: {cli_spft_keys}")
    
    # First, check for a --config flag (manually)
    if "--config" in sys.argv:
        config_idx = sys.argv.index("--config")
        config_path = sys.argv[config_idx + 1]
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)

        # Remove the config args to remain HfArgumentParser compatible
        del sys.argv[config_idx:config_idx+2]

        # Track existing CLI argument keys (e.g., "--learning-rate")
        existing_cli_keys = set(arg for arg in sys.argv if arg.startswith("--"))
        
        # Convert to argument-style format
        extra_args = []
        for k, v in config_dict.items():
            cli_key = "--" + k.replace("_", "-")
            if cli_key not in existing_cli_keys:
                v = str(v).lower() if isinstance(v, bool) else str(v)
                extra_args.extend([cli_key, v])
                
        # Inject into sys.argv
        sys.argv.extend(extra_args)

    model_args, data_args, training_args, remaining_args = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    
    # training_args.full_determinism = False
    set_seed(training_args.seed)
    
    main(model_args, data_args, training_args, cli_spft_keys)

