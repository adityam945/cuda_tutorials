import torch

if torch.cuda.is_available():
    gpu_count = torch.cuda.device_count()
    print(f"CUDA is available with {gpu_count} GPU(s).")
    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        print(f"--- GPU {i}: {props.name} ---")
        print(f"Compute Capability: {props.major}.{props.minor}")
        print(f"Number of Multiprocessors: {props.multi_processor_count}")
        # Note: The CUDA core count is derived from multi_processor_count
        # multiplied by cores per multiprocessor (which depends on compute capability)
else:
    print("CUDA is not available.")