# inspect_image_code
image code inspect


# Image modality experiment 
- Dataset path: https://stanfordaimi.azurewebsites.net/datasets/151848b9-8b31-4129-bc25-cefdf18f95d8
To generate image model results: 
- Make sure to change the dicom\_dir and csv\_path in configs files from image/radfusion3/configs  (classify, extract, model/resbetv2_ct and dataset/standford)
- Download weights from https://huggingface.co/StanfordShahLab/resnetv2_ct (or Train slice encoder using run_rsna.sh. Make sure the download the RSNA RESPECT dataset)
- Extract slice representation using **run_featurize.sh**.
    - Remember to change the ckpt path in image/radfusion3/configs/model/resnetv2_ct.yaml** && path in image/convert_to_hdf5.py
- After that you can using **run_classify_all.sh** to get classification results on all 8 tasks
    - If you want, run hyperparameter search with **wandb sweep sweep.yaml**. Note that line 8 specifies the prediction target. 


### Possible changes
- Skip nii.gz files which is not exist
- Change valid to val in split file
