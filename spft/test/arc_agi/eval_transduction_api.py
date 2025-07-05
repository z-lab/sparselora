######################
#   Example Usage    #
######################
#
# python eval/eval_transduction_api.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --temperature 0.5 --reasoning
#

import datetime
import json
import argparse
import os
import re
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer
from concurrent.futures import ThreadPoolExecutor, as_completed
from termcolor import colored
from litellm import text_completion
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def extract_solution(solution_str):
    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)
    if matches:
        final_answer = matches[-1].group(1).strip()
    else:
        final_answer = None
    return final_answer


def extract_answer_lists(generated_text):
    """
    Extract lists of numbers that appear after the </think> tag,
    handling cases where there might be additional text between the tag and lists.
    
    Args:
        generated_text (str): The full response from the model
        
    Returns:
        str: The extracted lists if found after </think>, None otherwise
    """
    # First check if </think> tag exists
    if '</think>' not in generated_text:
        return None
    
    # Get all content after </think> tag
    post_think_content = generated_text.split('</think>', 1)[1].strip()
    
    # Find all list patterns in the post-think content
    list_pattern = r'\[\s*\d+\s*,\s*\d+\s*(?:,\s*\d+\s*)*\]'
    list_matches = re.findall(list_pattern, post_think_content)
    
    if list_matches:
        # Extract the lines with valid lists
        valid_lines = []
        
        # Process each list match
        for list_match in list_matches:
            # Validate that it's a proper list of numbers
            if list_match.startswith('[') and list_match.endswith(']'):
                content = list_match[1:-1]
                elements = [elem.strip() for elem in content.split(',')]
                if all(re.match(r'^\d+$', elem) for elem in elements):
                    valid_lines.append(list_match)
        
        if valid_lines:
            # Simply join the valid lists with newlines
            return '\n'.join(valid_lines)
    
    return None


def parse_args():
    parser = argparse.ArgumentParser(description="Transduction evaluation using sglang API")
    parser.add_argument("--model", type=str, default='barc0/Llama-3.1-ARC-Potpourri-Transduction-8B',
                        help="Base model to use")
    parser.add_argument("--tokenizer", type=str, default=None,
                        help="Tokenizer to use (defaults to model path)")
    parser.add_argument("--problem_file", type=str, 
                        default="spft/test/arc_agi/validation_transduction_prompt.jsonl",
                        help="Path to the problem file")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Top-p sampling parameter")
    parser.add_argument("--top_k", type=int, default=50,
                        help="Top-k sampling parameter")
    parser.add_argument("--num_samples", type=int, default=2,
                        help="Number of samples to generate per prompt")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for inference")
    parser.add_argument("--max_tokens", type=int, default=8192,
                        help="Maximum number of tokens to generate")
    parser.add_argument("--num_workers", type=int, default=64,
                        help="Number of worker threads for parallel processing")
    parser.add_argument("--represent_type", type=str, choices=["color", "numeric"], default="color",
                        help="representation of arc-agi problem")
    parser.add_argument("--reasoning", action="store_true",
                        help="Use reasoning to solve the problem")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed information about each sample")
    parser.add_argument("--api_base", type=str, default="http://127.0.0.1:2345/v1",
                        help="Base URL for the sglang API")
    parser.add_argument("--use_ttt", action="store_true",
                        help="Use TTT lora adapter")
    return parser.parse_args()


def generate_with_api(
    prompt, api_base, temperature, max_tokens, top_p, top_k, lora_path=None):
    """
    Generate text using the sglang API via litellm.
    
    Args:
        prompt (str): The input prompt
        api_base (str): Base URL for the API
        temperature (float): Sampling temperature
        max_tokens (int): Maximum number of tokens to generate
        top_p (float): Top-p sampling parameter
        top_k (int): Top-k sampling parameter
        lora_path (str): Path to the lora adapter
    Returns:
        str: The generated text
    """
    model_name = "model"
    try:
        result = text_completion(
            model=f"openai/{model_name}",
            prompt=prompt,
            api_base=api_base,
            api_key="api_key",  # Using a placeholder API key
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            top_k=top_k,
            lora_path=lora_path,
        )
        return result['choices'][0]['text']
    except Exception as e:
        raise Exception(f"API request failed: {str(e)}")


def process_sample(args, data_idx, d, tokenizer):
    """
    Process a single sample using the API.
    
    Args:
        args: Command line arguments
        data_idx (int): Index of the current data point
        d (dict): The data point
        tokenizer: The tokenizer to use
        
    Returns:
        tuple: (current_result, found_correct, best_match, best_response)
    """
    messages = d["messages"]
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    
    if args.represent_type == "color":
        if not args.reasoning:
            chat_sequence = [
                {"role": "system", "content": messages[0]["content"]},
                {"role": "user", "content": messages[1]["content"]},
                # {"role": "assistant", "content": messages[2]["content"]},
                {"role": "assistant", "content": ""},
            ]
        else:
            system_prompt = messages[0]["content"]
            user_prompt = messages[1]["content"].split("Directly provide")[0]
            user_prompt += "\nAnalyze what the output grids corresponding to the given test input grids should be, based on the patterns observed in the reference examples. \
Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, \
for example <answer> Yellow Green\nYellow Black\n </answer>."
            chat_sequence = [
                # {"role": "user", "content": system_prompt + "\n" + user_prompt},
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": "<think>"},
            ]
    else:
        chat_sequence = [
            {"role": "user", "content": messages[0]["content"] + "\n" + messages[1]["content"]},
            {"role": "assistant", "content": messages[2]["content"]}
        ]
    
    inputs_text = tokenizer.apply_chat_template(
        chat_sequence, tokenize=False, add_generation_prompt=False
    )
    eos_token = tokenizer.eos_token
    if eos_token and inputs_text.strip().endswith(eos_token):
        inputs_text = inputs_text.strip()[:-len(eos_token)]
    
    # Generate multiple samples
    responses = []
    # print("Temper")
    for _ in range(args.num_samples):
        try:
            generated_text = generate_with_api(
                inputs_text,
                args.api_base,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                top_p=args.top_p,
                top_k=args.top_k,
                # lora_path="lora0" if args.use_ttt else None,
            )
            responses.append(generated_text)
        except Exception as e:
            print(f"Error generating sample {data_idx}: {str(e)}")
            responses.append("")
    
    # Prepare result dictionary
    current_result = {
        "prompt": inputs_text,
        "responses": responses,
        "base_model": args.model
    }
    
    # Evaluate responses
    found_correct = False
    best_response = None
    best_match = None
    
    for generated_text in responses:
        parsed_generated_text = None
        
        if args.represent_type == "color":
            color_regex = r"\b(?:Black|Blue|Red|Green|Yellow|Grey|Gray|Pink|Orange|Purple|Brown)(?:[\s\n]+(?:Black|Blue|Red|Green|Yellow|Grey|Gray|Pink|Orange|Purple|Brown))*\b"
            match = None
            if not args.reasoning:
                matches = re.findall(color_regex, generated_text)
                if matches:
                    match = matches[-1]  # Get the last match
            else:
                output = extract_solution(generated_text)
                if output:
                    matches = re.findall(color_regex, output)
                    if matches:
                        match = matches[-1]  # Get the last match
            if match:
                parsed_generated_text = match.strip()
                tokens = parsed_generated_text.split()
                valid_colors = {"Black", "Blue", "Red", "Green", "Yellow", "Grey", "Gray", "Pink", "Orange", "Purple", "Brown"}
                if not (tokens and tokens[0] in valid_colors and tokens[-1] in valid_colors):
                    parsed_generated_text = None
        elif args.represent_type == "numeric":
            parsed_generated_text = extract_answer_lists(generated_text)
        
        if parsed_generated_text:
            if best_match is None:
                best_match = parsed_generated_text
                best_response = generated_text
            
            if parsed_generated_text.strip() == d['answer'].strip():
                found_correct = True
                best_match = parsed_generated_text
                best_response = generated_text
                break
    
    current_result["answer"] = d['answer']
    current_result["extracted_answer"] = best_match
    current_result["correct"] = found_correct
    
    return current_result, found_correct, best_match, best_response


def main():
    args = parse_args()
    
    # Set up tokenizer
    if args.tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    # Load data
    data = []
    with open(args.problem_file) as f:
        for line in f:
            data.append(json.loads(line))
    
    # Generate timestamp for output file
    datetime_str = datetime.datetime.now().strftime("%m%d%H%M%S")
    
    # Set up output file path
    if 'checkpoint' in args.model.split('/')[-1]:
        model_name = args.model.split('/')[-2] + "_" + args.model.split('/')[-1]
    else:
        model_name = args.model.split('/')[-1]
    
    problem_filename = os.path.basename(args.problem_file).replace('.jsonl', '')
    os.makedirs("results/validation_transduction", exist_ok=True)    
    saving_file = os.path.join(
        "results/validation_transduction",
        # f"{problem_filename}_{model_name}_"
        f"{model_name}_"
        f"{'ttt_' if args.use_ttt else ''}"
        f"temp{args.temperature}_"
        f"samples{args.num_samples}_"
        f"{'no_cot_' if not args.reasoning else ''}"
        f"{datetime_str}.jsonl"
    )
    print(colored(f"📝 Output file: {saving_file}", "cyan"))
    
    # Print configuration summary
    print(colored("🚀 Configuration:", "yellow"))
    print(colored(f"  • Represent type: {args.represent_type}", "yellow"))
    print(colored(f"  • Model: {args.model}", "yellow"))
    print(colored(f"  • Reasoning: {args.reasoning}", "yellow"))
    print(colored(f"  • Temperature: {args.temperature}", "yellow"))
    print(colored(f"  • Samples per prompt: {args.num_samples}", "yellow"))
    print(colored(f"  • Batch size: {args.batch_size}", "yellow"))
    print(colored(f"  • Workers: {args.num_workers}", "yellow"))
    
    correct_counter = 0
    correct_task = []
    
    # Process data in parallel
    with open(saving_file, "w") as f:
        with tqdm(total=len(data), desc=colored("Evaluating", "green"), unit="sample") as pbar:
            with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
                futures = []
                for i, d in enumerate(data):
                    future = executor.submit(process_sample, args, i, d, tokenizer)
                    futures.append(future)
                
                for i, future in enumerate(as_completed(futures)):
                    current_result, found_correct, best_match, best_response = future.result()
                    
                    if found_correct:
                        correct_counter += 1
                        correct_task.append(data[i]['uid'])
                    
                    f.write(json.dumps(current_result) + "\n")
                    f.flush()
                    
                    if best_match is None:
                        if args.represent_type == "color":
                            print(colored(f"❌ Sample {i}: No valid color format found", "red"))
                        elif args.represent_type == "numeric":
                            print(colored(f"❌ Sample {i}: No valid numeric list format found", "red"))
                    elif args.verbose:
                        if found_correct:
                            print(colored(f"✅ Sample {i}: Correct match '{best_match}'", "green"))
                        else:
                            print(colored(f"❌ Sample {i}: Found '{best_match}',\n expected '{data[i]['answer'].strip()}'", "red"))
                    
                    pbar.update(1)
                    pbar.set_postfix(accuracy=f"{(correct_counter/(i+1))*100:.2f}% ({correct_counter}/{i+1})")
    
    # Print final results with colors
    print("\n" + colored("📊 Results Summary:", "cyan", attrs=["bold"]))
    print(colored(f"  ✅ Correct: {correct_counter}/{len(data)} ({(correct_counter/len(data))*100:.2f}%)", 
                 "green" if correct_counter > 0 else "red"))
    
    if len(correct_task) > 0:
        print(colored(f"  🏆 Correctly solved {len(correct_task)} tasks", "green"))
        if args.verbose:
            print(colored(f"  🔍 UIDs: {', '.join(correct_task[:10])}" + 
                         ("..." if len(correct_task) > 10 else ""), "green"))
    
    print(colored(f"  💾 Results saved to: {saving_file}", "cyan"))
    
    # Append accuracy information to the result file
    with open(saving_file, "a") as f:
        accuracy_info = {
            "accuracy": (correct_counter/len(data))*100
        }
        f.write(json.dumps(accuracy_info) + "\n")
        print(colored(f"  📈 Accuracy information appended to result file", "cyan"))


if __name__ == "__main__":
    main() 