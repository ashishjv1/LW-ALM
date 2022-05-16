# Pedestrian Attribute Recognition Using Lightweight Attribute-Specific Localization Model
Compression of  Convolutional Layers in Attribute-Specific Localization Model using CPD-EPC, SVD Decompositions

## **Required Environments**
* Python 3.7 +
* Pytorch 1.11
* Flopco-pytorch 
* Numpy

## **Datasets**

* PETA: http://mmlab.ie.cuhk.edu.hk/projects/PETA.html
* PA-100K: https://github.com/xh-liu/HydraPlus-Net

## Labels 
* data_labels/peta
* data_labels/pa-100k

### **Steps to Reproduce**
1. Use `"run_main.py"` to save best checkpoint for the dataset. <br />
   ```
    python3 run_main.py --model=FULL --attr_num=26 --experiment=PA-100K --epoch=15 --checkpoint_save=path_to_save_checkpoints
   ```

2. Use `"submit_decompositios.py"` to decompose layers using SVD or CPD-EPC. <br />
   ```
   python3 submit_decompositios.py --dpath=path_to_dataset --mpath=path_to_saved_checkpoint --tlabels=path_to_train_labels --vlabels=path_to_val_labels --factors=both --attr_num=26 --experiment=PA-100K --device=cpu
   ```

3. Use `"create_compressed_model.py"` to create "Fully-Compressed" or "Partially-Compressed" Model.
   
   1. To create a Fully-Compressed Model:
      (Use create_compressed_model.py script)
      ```
      python3 create_compressed_model.py --model_type=full --save_path=path_to_save_model --ranks_dir=directory_to_load_factors&ranks --attr_num=26 or 35
      ```
   2. To create a Partially-Compressed Model:
      1. Perform step i.
      2. Use create_compressed_model.py script for partial model creation. <br />
           ```
           python3 create_compressed_model.py --full_model=path_to_full_model --model_type=partial --save_path=path_to_save_model --ranks_dir=directory_to_load_factors&ranks --attr_num=26 or 35
           ```

4. Use `"run_main.py"` to fine-tune compressed model <br />
   ``` 
   python3 main.py --model=COMPRESSED --attr_num=26 --experiment=PA-100K --epoch=15 --model_path=path_to_compressed_model --checkpoint_save=path_to_save_checkpoints
   ```

Besides,<br />
`layer_decomposer.py` can be used to decompose only specific layers. <br />

example:
```
python3 layer_decomposition.py --layer=main_branch.conv1_7x7_s2 --rank=81 --eps=0.002 --device=cpu --dpath=dataset_path --mpath=model_path --tlabels=data_train_label --vlabels=data_val/test_label --experiment=PA-100K --attr_num=26
```