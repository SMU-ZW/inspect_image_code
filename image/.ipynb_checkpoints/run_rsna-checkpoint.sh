# #CUDA_VISIBLE_DEVICES=1 python run_classify.py model=swinv2 dataset=rsna 
# #CUDA_VISIBLE_DEVICES=1 python run_classify.py model=dinov2 dataset=rsna dataset.transform.final_size=224 
# #python run_classify.py model=dinov2 dataset=rsna dataset.transform.resize_size=512 dataset.transform.crop_size=448 dataset.transform.final_size=448 dataset.batch_size=16
# CUDA_VISIBLE_DEVICES=3 python run_classify.py model=resnetv2 dataset=rsna dataset.transform.final_size=224 

# #dataset.tranform.final_size=224

#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

echo "Using GPUs: $CUDA_VISIBLE_DEVICES"

CUDA_VISIBLE_DEVICES=3 python run_classify.py model=resnetv2 dataset=rsna dataset.transform.final_size=224 
