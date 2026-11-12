import torch
import time

# --- 1. Check GPU availability ---
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("✅ MPS GPU is available. Using M4 GPU.")
else:
    device = torch.device("cpu")
    print("❌ MPS GPU not available. Using CPU.")
    exit()

# --- 2. Create large matrices ---
# We will use 5000 x 5000 matrices. This is big enough to show the difference.
size = 10000
print(f"\nCreating two {size}x{size} matrices...")

# Create on CPU
cpu_matrix_a = torch.randn(size, size)
cpu_matrix_b = torch.randn(size, size)

# Move to GPU
gpu_matrix_a = cpu_matrix_a.to(device)
gpu_matrix_b = cpu_matrix_b.to(device)

print("Matrices created and moved to GPU.\n")

# --- 3. Benchmark CPU matrix multiplication ---
print("--- CPU Matrix Multiplication ---")
start = time.time()
cpu_result = torch.matmul(cpu_matrix_a, cpu_matrix_b)
cpu_time = time.time() - start
print(f"CPU time: {cpu_time:.4f} seconds")

# --- 4. Benchmark GPU matrix multiplication ---
print("\n--- GPU Matrix Multiplication ---")
# IMPORTANT: We must synchronize the GPU before timing.
# The GPU runs asynchronously. Without sync, the timer stops before the GPU finishes.
torch.mps.synchronize() 
start = time.time()
gpu_result = torch.matmul(gpu_matrix_a, gpu_matrix_b)
torch.mps.synchronize() # Wait for the GPU to finish
gpu_time = time.time() - start
print(f"GPU time: {gpu_time:.4f} seconds")

# --- 5. Calculate speedup ---
speedup = cpu_time / gpu_time
print(f"\n🚀 Speedup: {speedup:.2f}x faster on GPU")

# --- 6. Verify results match (sanity check) ---
# Move GPU result back to CPU to compare
gpu_result_cpu = gpu_result.to("cpu")
if torch.allclose(cpu_result, gpu_result_cpu, atol=1e-4):
    print("✅ Results match between CPU and GPU.")
else:
    print("⚠️ Results differ. This is likely due to floating-point precision differences.")

# --- Warm-up run (ignore the time) ---
_ = torch.matmul(gpu_matrix_a, gpu_matrix_b)
torch.mps.synchronize()

# --- Timed run (after warm-up) ---
torch.mps.synchronize()
start = time.time()
gpu_result = torch.matmul(gpu_matrix_a, gpu_matrix_b)
torch.mps.synchronize()
gpu_time = time.time() - start