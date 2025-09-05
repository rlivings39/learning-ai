import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
torch.set_default_device(device)
print(f"Using device = {torch.get_default_device()}")
