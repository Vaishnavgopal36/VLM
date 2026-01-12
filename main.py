import subprocess
import sys
import os
import time

def execute_safe():
    print(">>> [MASTER] Launching Training Process...", flush=True)

    # 1. Kill old zombies
    # This ensures no previous 'run_task.py' processes are hogging the TPU
    try:
        subprocess.call("pkill -9 -f run_task.py", shell=True)
        time.sleep(2) # Give the OS a moment to reclaim resources
    except: pass

    # 2. Prepare clean environment
    # Crucial: We remove TPU env vars from THIS parent process
    # to avoid accidental initialization in the notebook kernel.
    clean_env = os.environ.copy()
    keys_to_purge = ["PJRT_DEVICE", "XLA_USE_BF16", "TPU_PROCESS_ADDRESSES"]
    for k in keys_to_purge:
        if k in clean_env: del clean_env[k]

    # 3. Launch
    # We use sys.executable to ensure we use the same Python interpreter
    try:
        subprocess.check_call([sys.executable, "run_task.py"], env=clean_env)
        print(">>> [MASTER] TRAINING COMPLETE SUCCESS.", flush=True)
    except subprocess.CalledProcessError as e:
        print(f"!!! [MASTER] Training Crashed with code {e.returncode}", flush=True)

# Run the execution
if __name__ == "__main__":
    execute_safe()
