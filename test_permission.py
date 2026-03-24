
import os

test_file = r"F:\guoji\test_write.txt"
print(f"Trying to write to {test_file}...", flush=True)

try:
    with open(test_file, "w") as f:
        f.write("Hello world")
    print("Success: File written.", flush=True)
    
    # Clean up
    os.remove(test_file)
    print("Success: File removed.", flush=True)
    
except Exception as e:
    print(f"FAIL: {e}", flush=True)
