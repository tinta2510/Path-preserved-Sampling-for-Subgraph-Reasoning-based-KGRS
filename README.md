git clone https://github.com/tinta2510/Path-preserved-Sampling-for-Subgraph-Reasoning-based-KGRS.git

# ---- PyTorch 2.8.x + CUDA 12.6 (has torch_scatter wheels) ----
pip install torch==2.8.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# ---- torch_scatter: PREBUILT wheel (fast) ----
pip install torch_scatter -f https://data.pyg.org/whl/torch-2.8.0+cu128.html

# other deps your code uses
pip install numpy scipy tqdm