export CUDA_VISIBLE_DEVICES=4,5,6,7
python -m torch.distributed.launch --nproc_per_node=4 globalpointer_train_ddp.py
