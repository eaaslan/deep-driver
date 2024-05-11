import torch

if torch.cuda.is_available():
    print("CUDA is available. GPU will be used for training.")
else:
    print("CUDA is not available. Training will use CPU.")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# create a tensor
tensor = torch.Tensor([1.0, 2.0])

# move the tensor to the GPU if available
tensor = tensor.to(device)

if torch.cuda.is_available():
    print("CUDA is available. GPU will be used for training.")
else:
    print("CUDA is not available. Training will use CPU.")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")