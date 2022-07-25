python -m torch.distributed.launch --nproc_per_node=1 --master_port=12352 main_test_psnr_swin.py --opt options/swinir/train_swinv2ir_sr_classical_x4.json  --dist false

# python -m torch.distributed.launch --nproc_per_node=1 --master_port=12351 main_test_psnr_swin.py --opt options/swinir/train_swinv2ir_sr_classical_x4_aux_modify.json  --dist false

# python -m torch.distributed.launch --nproc_per_node=1 --master_port=12351 main_test_psnr_swin.py --opt options/swinir/train_swinv2ir_sr_classical_x4_aux_modify_hf.json  --dist false