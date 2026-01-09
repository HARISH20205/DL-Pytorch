torchrun --nproc_per_node=1 --nnodes=2 --node_rank=0 \
  --master_addr="your_tailscale_ip" --master_port=12355 main1.py