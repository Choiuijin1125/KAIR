MODEL=60000_E.pth
echo "Swin2SR X2 upsample test ================================================="
echo "Swin2SR X2 upsample test : Set5 ================================================="
python main_test_swinir.py --task classical_sr --scale 2 --training_patch_size 64 --model_path superresolution/swinv2ir_sr_classical_patch48_x2/models/$MODEL  --folder_gt testsets/test/Set5/original --folder_lq testsets/test/Set5/LRbicx2

echo "Swin2SR X2 upsample test : Set14 ================================================="
python main_test_swinir.py --task classical_sr --scale 2 --training_patch_size 64 --model_path superresolution/swinv2ir_sr_classical_patch48_x2/models/$MODEL  --folder_gt testsets/test/Set14/original --folder_lq testsets/test/Set14/LRbicx2

echo "Swin2SR X2 upsample test : BSDS100 ================================================="
python main_test_swinir.py --task classical_sr --scale 2 --training_patch_size 64 --model_path superresolution/swinv2ir_sr_classical_patch48_x2/models/$MODEL  --folder_gt testsets/test/BSDS100/original --folder_lq testsets/test/BSDS100/LRbicx2

echo "Swin2SR X2 upsample test : manga109 ================================================="
python main_test_swinir.py --task classical_sr --scale 2 --training_patch_size 64 --model_path superresolution/swinv2ir_sr_classical_patch48_x2/models/$MODEL  --folder_gt testsets/test/manga109/original --folder_lq testsets/test/manga109/LRbicx2

echo "Swin2SR X2 upsample test : urban100 ================================================="
python main_test_swinir.py --task classical_sr --scale 2 --training_patch_size 64 --model_path superresolution/swinv2ir_sr_classical_patch48_x2/models/$MODEL  --folder_gt testsets/test/urban100/original --folder_lq testsets/test/urban100/LRbicx2
