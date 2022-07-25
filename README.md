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

3. generate each model's prediction image
```
sh test.sh
```

4. ensemble of images
```
python ensemble.py
```
It will generate `ensemble_result` folder in superresolution folder
