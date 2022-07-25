## Train

```
sh train.sh
```

## Test

1. download pretrained weights 
2. unzip pretrained weights in root dir and It will show like below
    ```
    KAIR
    ├── ...
    ├── superresolution                    
    │   ├── swinv2ir_sr_classical_patch48_x4_external_data
    │     └── models/190000_G.pth
    │   ├── swinv2ir_sr_classical_patch48_x4_external_data_aux_modify
    │     └── models/310000_G.pth
    │   ├── swinv2ir_sr_classical_patch48_x4_external_data_aux_modify_hf
    │     └── models/380000_G.pth
    │   ├── ensemble_result
    ├── ensemble.py
    ```

3. put test datasets in `testsets` folder and change `test_dir` in each configs. configs are in `options/swinir`

    ```
    KAIR
    ├── ...
    ├── options               
    │   ├── train_swinv2ir_sr_classical_x4.json ## swinv2ir_sr_classical_patch48_x4_external_data
    │   ├── train_swinv2ir_sr_classical_x4_aux_modify.json ## swinv2ir_sr_classical_patch48_x4_external_data_aux_modify
    │   ├── train_swinv2ir_sr_classical_x4_aux_modify_hf.json ## swinv2ir_sr_classical_patch48_x4_external_data_aux_modify_hf
    ```

4. generate each model's prediction image
```
sh test.sh
```

4. ensemble of images
```
python ensemble.py
```
It will generate `ensemble_result` folder in superresolution folder
