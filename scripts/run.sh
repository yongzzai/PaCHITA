export TORCH_USE_CUDA_DSA=0
export TORCH_CUDNN_V8_API_ENABLED=1

uv run main.py

wait

echo "Done."