import subprocess
import time
import socket
import os
import argparse
from tqdm import tqdm
import sys
import threading

# Merge model weights:
def merge_model_weights(model_name, lora_path):
    print(f"[INFO] Merging model weights from {lora_path} into {model_name}...")
    subprocess.run([
        "python", "tools/merge.py",
        "--base_model", model_name,
        "--peft_model", lora_path,
    ], check=True,  stdout=None, stderr=None)
    
    output_dir = os.path.join(lora_path, "merged")
    print(f"[INFO] Model weights merged and saved to {output_dir}")
    return output_dir

def wait_for_port(port, host='localhost', timeout=600, interval=2): #* 5 Minute timeout should be sufficient for SGLang server to start
    print(f"[INFO] Waiting for port {port} to become available...")
    attempts = timeout // interval

    for _ in tqdm(range(attempts), desc=f"Polling port {port}", unit="try", ncols=80):
        try:
            with socket.create_connection((host, port), timeout=interval):
                print(f"\n[INFO] Port {port} is open, server is ready.")
                return True
        except OSError:
            time.sleep(interval)

    raise TimeoutError(f"[ERROR] Port {port} not available after {timeout} seconds.")

def stream_output(pipe):
    for line in iter(pipe.readline, b''):
        sys.stdout.write(line.decode())
    pipe.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_name_or_path", type=str, required=True)
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--proc_count", type=int, default=1, help="Number of processes to run for distributed inference.")
    args = parser.parse_args()


    output_dir = merge_model_weights(args.base_model_name_or_path, args.model_name_or_path)

    print(f"[INFO] Merged model saved to {output_dir}")
    print(f"[INFO] Starting SGLang server with {args.proc_count} processes...")
    PORT = 2345
    
    # 1. Launch SGLang server
    
    command = [
        "python", "-m", "sglang.launch_server",
        "--model-path", output_dir,
        "--dp-size", str(args.proc_count),
        "--port", str(PORT),
        "--disable-radix-cache", 
        "--attention-backend", "triton",
        "--sampling-backend", "pytorch",
        "--disable-custom-all-reduce"
    ]
    sglang_proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # 2. Stream logs to terminal
    threading.Thread(target=stream_output, args=(sglang_proc.stdout,), daemon=True).start()
    threading.Thread(target=stream_output, args=(sglang_proc.stderr,), daemon=True).start()

    # # 3. Wait until server is ready
    try:
        wait_for_port(PORT)
    except TimeoutError as e:
        print(e)
        sglang_proc.terminate()
        exit(1)

        
    print(f"[INFO] SGLang server started successfully on port {PORT} with {args.proc_count} processes.")
    
    try:
        # 4. Run evaluation
        for N in [4]:
            for T in [1.0]:
                subprocess.run([
                    "python", "spft/test/arc_agi/eval_transduction_api.py",
                    "--temperature", str(T),
                    "--model", args.base_model_name_or_path,
                    "--num_samples", str(N),
                    "--use_ttt"
                ], stdout=None, stderr=None)

    except:
        pass
    
    print("Teardown SGLang server...")
    # 4. kill the server
    sglang_proc.terminate()

if __name__ == "__main__":
    main()