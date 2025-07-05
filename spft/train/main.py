import os
import torch
from torch import distributed as dist


world_size = int(os.environ.get("WORLD_SIZE", "1"))
rank = int(os.environ.get("RANK", "0"))
if world_size == 1 or rank == 0:
    try:
        import unsloth
        print("[INFO] Imported unsloth.")
    except ImportError:
        print("[WARNING] Unsloth import failed.")
else:
   print("Unsloth is not available in distributed mode. If you are training with unsloth please run in single process mode.") 
         
import transformers
from transformers import HfArgumentParser, set_seed
from spft.api import SPFTConfig, get_spft_callback, get_spft_model
from spft.callbacks import EvaluateFirstStepCallback
from spft.train.args import DataTrainingArguments, ModelArguments, TrainingArguments
from spft.utils.io import build_runname
import yaml
import sys
import subprocess
from spft.utils.data import load_dataset_by_name


def main(model_args: ModelArguments, data_args: DataTrainingArguments, training_args: TrainingArguments, cli_keys: set) -> None:    # Set up SPFT
    
    if training_args.enable_unsloth:
        #? Unsloth only patches if LoRA module present on component.
        training_args.lora_target_modules = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
    
    assert training_args.spft is not None, "SPFT is not enabled. Please set --spft to a valid path."
    spft_config = SPFTConfig.from_file(training_args.spft)
    spft_config.update([model_args, data_args, training_args], prefix="spft_", cli_keys=cli_keys)
    training_args.run_name = build_runname(training_args, data_args, spft_config)
    
    peft_name = "none" if training_args.peft is None else f"{training_args.peft}"
    base_path = f"{training_args.output_dir}/{peft_name}/{training_args.run_name}/"
    training_args.output_dir = f"{base_path}{training_args.seed}/"
    spft_config.write_out(training_args.output_dir)
    
    if not training_args.eval_only:
        from spft.utils.model import create_model_and_tokenizer
        print("Launching Model: ", model_args.model_name_or_path)
        
        #* Create model + LoRA
        model, tokenizer = create_model_and_tokenizer(model_args, data_args, training_args)
        spft_config.padding_side = tokenizer.padding_side
        
        callbacks = []
        model = get_spft_model(model, spft_config, enable_unsloth=training_args.enable_unsloth)
        callbacks.append(get_spft_callback(spft_config))
        model = model.to(torch.bfloat16)
        
        model.print_trainable_parameters()
        # Load dataset
        
        print(f"[INFO] Loading dataset: {data_args.dataset}, max_length: {data_args.model_max_length}, max_seq_length: {data_args.max_seq_length}, batch_size: {training_args.per_device_train_batch_size}")
        
        assert data_args.model_max_length == data_args.max_seq_length, "model_max_length must be equal to max_seq_length for training"
        
        data_dict = load_dataset_by_name(data_args, tokenizer)
        
        if data_dict["eval"] is not None:
            callbacks.append(EvaluateFirstStepCallback())

        # Set up data collator
        collator = transformers.DataCollatorForSeq2Seq(
            tokenizer,
            pad_to_multiple_of=data_args.max_seq_length,
            return_tensors="pt",
            padding=True,
        )
        
        print("Training save strategy: ", training_args.save_strategy)
       
        trainer = transformers.Trainer(
            model=model,
            args=training_args,
            train_dataset=data_dict["train"],
            eval_dataset=data_dict["eval"],
            data_collator=collator,
            callbacks=callbacks,
        )    
        
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        
        if "wizardlm" in data_args.dataset.lower():
            #* Save the model
            print(f"[INFO] Saving fulll model for chat: {training_args.output_dir}")
            #* Must Save the full checkpoint model to avoid chat-specific issues
            model = model.merge_and_unload()
            
            trainer.save_model(output_dir=training_args.output_dir)
            trainer.save_state()

            tokenizer.save_pretrained(training_args.output_dir)
        else:
            print(f"[INFO] Model saved to {training_args.output_dir}")
            trainer.save_model(output_dir=training_args.output_dir)
            trainer.save_state()
    
    else: 
        if "wizardlm" in data_args.dataset.lower():
            print(f"[INFO] Skipping Eval. Test for chat model must be conducted using FastChat")
            return
        n_proc_per_node = dist.get_world_size() if dist.is_initialized() else 1
        
        if n_proc_per_node > 1:
            print(f"[INFO] Running in distributed mode with {n_proc_per_node} processes.")
            prefix_cmd = ["torchrun", f"--nproc_per_node={n_proc_per_node}", f"--master_port={29501}"]
        else:
            prefix_cmd = ["python"]

        if "code" in data_args.dataset.lower():
            main_file = "spft/test/code.py"
            add_args = [
                "--base_model_name_or_path", model_args.model_name_or_path,
            ]
        elif "arc" in data_args.dataset.lower():
            main_file = "spft/test/arc_agi/arc_agi.py"
            add_args = [
                "--base_model_name_or_path", model_args.model_name_or_path,
                "--proc_count", str(torch.cuda.device_count()) #* Count number of GPUs visible
            ]
        else:
            main_file = "spft/test/main.py"
            add_args = [
                "--dataset", data_args.eval_dataset,
            ]
        
        #* Launch Testing:
        command = prefix_cmd + [    
                    main_file,
                    "--model_name_or_path", training_args.output_dir,
                ] + add_args

        
        if int(os.environ.get("RANK", 0)) == 0 or not dist.is_initialized():
            #* Only the main process should run the test
            print(f"[INFO] Launching test with: {command}")
            subprocess.run(command, check=True, stdout=None, stderr=None)
          
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
    
    set_seed(training_args.seed)
    
    main(model_args, data_args, training_args, cli_spft_keys)

