import torch

print("Torch version:", torch.__version__)
print("CUDA version:", torch.version.cuda)
print("Device:", torch.cuda.get_device_name(0))
