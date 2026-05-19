#pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/rocm5.7
#pip install transformers==4.35.2 tokenizers==0.15.2 accelerate==0.28.0 datasets==2.21.0 sentence-transformers==2.7.0 evaluate sacrebleu bert-score deep-translator tqdm numpy matplotlib PyYAML pymorphy3 rapidfuzz protobuf sentencepiece tiktoken


import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    print(f"Device capability: {torch.cuda.get_device_capability(0)}")

try:
    x = torch.tensor([1, 2, 3], device="cuda")
    print("Tensor created on GPU, no error.")
except Exception as e:
    print(f"Error creating tensor: {e}")
    exit()

try:
    embedding = torch.nn.Embedding(1000, 128).to("cuda")
    input_ids = torch.randint(0, 1000, (4, 16), device="cuda")
    out = embedding(input_ids)
    print("Embedding forward pass succeeded on GPU.")
except Exception as e:
    print(f"Embedding forward pass FAILED: {e}")