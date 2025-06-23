import argparse
import os
import sys
import yaml
from huggingface_hub import snapshot_download
from rich.console import Console
from rich.table import Table

console = Console()
BASE_REPO_ID = "z-lab/sparselora-svd-estimator"
LOCAL_PREDICTOR_BASE = "spft/modules/low_rank_weights"

def parse_config(sparsity_config):
    with open(sparsity_config, "r") as f:
        config = yaml.safe_load(f)

    mode = config.get("mode", "svd_8")
    sparsity = config.get("sparsity", {})

    mlp_layers, attn_layers = [], []
    for key, value in sparsity.items():
        if value == 0:
            continue
        try:
            parts = key.split(".")
            idx = parts[1]
            if parts[-1] == "mlp":
                mlp_layers.append(idx)
            elif parts[-1] == "self_attn":
                attn_layers.append(idx)
        except Exception as e:
            console.print(f"[red]⚠ Failed to parse: {key} ({e})[/red]")

    return mode, mlp_layers, attn_layers

def download_all_predictors():
    console.print(f"[bold cyan]🔍 Downloading all predictors from HF repo: {BASE_REPO_ID}[/bold cyan]")
    try:
        snapshot_download(
            repo_id=BASE_REPO_ID,
            local_dir=LOCAL_PREDICTOR_BASE,
            local_dir_use_symlinks=False,
        )
        console.print(f"[yellow]🔽 Predictors downloaded to: {LOCAL_PREDICTOR_BASE}[/yellow]")
    except Exception as e:
        console.print(f"[red]❌ Download failed: {e}[/red]")
        sys.exit(1)

def verify_predictors(args, mode, mlp_layers, attn_layers):
    model_id = args.model_id
    if "svd" not in mode:
        console.print("[green]✅ No predictors required for this mode.[/green]")
        return

    rank = mode.split("_")[-1]
    predictor_dir = os.path.join(LOCAL_PREDICTOR_BASE, f"{model_id}/r_{rank}")
    if not os.path.exists(predictor_dir) or not os.listdir(predictor_dir):
        console.print(f"[red]❌ Missing predictors at: {predictor_dir}[/red]")
        console.print(f"[yellow]⚠ Attempting to download all predictors from Hugging Face...[/yellow]")
        download_all_predictors()

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Predictor File")
    table.add_column("Exists", justify="center")

    all_exist = True
    for layer in mlp_layers:
        for suffix in ["a", "b"]:
            fname = f"layers.{layer}.mlp_low_rank_{suffix}.pt"
            fpath = os.path.join(predictor_dir, "mlp", fname)
            exists = os.path.exists(fpath)
            table.add_row(f"mlp/{fname}", "[green]✅[/green]" if exists else "[red]❌[/red]")
            all_exist &= exists

    for layer in attn_layers:
        for head in ["q", "k", "v"]:
            for suffix in ["a", "b"]:
                fname = f"layers.{layer}.self_attn_{head}_low_rank_{suffix}.pt"
                fpath = os.path.join(predictor_dir, "attn", fname)
                exists = os.path.exists(fpath)
                table.add_row(f"attn/{fname}", "[green]✅[/green]" if exists else "[red]❌[/red]")
                all_exist &= exists

    console.print(table)
    if all_exist:
        console.print("\n[bold green]🎉 All required SVD predictors are available![/bold green]")
    else:
        console.print("\n[bold red]⚠ Some predictors are still missing after download.[/bold red]")
        
        create_preds = console.input("[yellow]Do you want to create missing predictors? (yes/no): ").strip().lower()
        if create_preds == "yes":
            console.print("[yellow]⚠ Creating missing predictors...this may take some time.[/yellow]")
            # Here you would implement the logic to create the missing predictors
            
            create_predictor(args)
            
        else:
            console.print("[bold red]Exiting without creating predictors.[/bold red]")
            sys.exit(1)
        
        
def create_predictor(args):
    from spft.train.args import DataTrainingArguments, ModelArguments, TrainingArguments
    from transformers import HfArgumentParser
    from spft.api import SPFTConfig, get_spft_model
    from spft.utils.model import create_model_and_tokenizer
    
    args_str = sys.argv[1:]
    if "--output_dir" not in args:
        args_str += ["--output_dir", "temp"]

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args, _ = parser.parse_args_into_dataclasses(args_str, return_remaining_strings=True)
    model_args.model_name_or_path = args.model_id
    spft_config = SPFTConfig.from_file(args.sparsity_config)
    spft_config.update([model_args, data_args, training_args], prefix="spft_", cli_keys={})
    #* Create model + LoRA
    model, _ = create_model_and_tokenizer(model_args, data_args, training_args)
    model = get_spft_model(model, spft_config, enable_unsloth=training_args.enable_unsloth)
    
    return


def main():
    parser = argparse.ArgumentParser(description="Download and verify SVD predictors.")
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--sparsity_config", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(LOCAL_PREDICTOR_BASE, exist_ok=True)
    mode, mlp_layers, attn_layers = parse_config(args.sparsity_config)

    console.print(f"\n[bold cyan]🔍 Verifying predictors for model: '{args.model_id}'[/bold cyan]")
    console.print(f"[bold]Mode:[/bold] {mode}\n")
    verify_predictors(args, mode, mlp_layers, attn_layers)

if __name__ == "__main__":
    main()
