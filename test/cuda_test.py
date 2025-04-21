import torch

def test_torch_and_gpu():
    print("PyTorch version:", torch.__version__)
    if torch.cuda.is_available():
        print("CUDA is available!")
        print("GPU Name:", torch.cuda.get_device_name(0))
        print("Number of GPUs:", torch.cuda.device_count())
        print("Current GPU:", torch.cuda.current_device())
    else:
        print("CUDA is not available.")

if __name__ == "__main__":
    test_torch_and_gpu()