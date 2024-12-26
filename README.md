# BullySense
BullySense: A Vision Language Model Approach for Detecting Campus Bullying

Campus bullying poses a significant threat to students' mental and physical well-being, necessitating timely and effective detection. This study introduces BullySense, a vision language model (VLM) based approach for detecting campus bullying incidents.



# Dataset and weights

The dataset used in this study is available for download via the following link: https://huggingface.co/datasets/Gavin6114/BullySense/tree/main. Due to the sensitive nature of the dataset, if the link becomes invalid or inaccessible, please contact the corresponding author to request access to the dataset.

The weights used in this study is available via the following link: https://huggingface.co/datasets/Gavin6114/BullySense/tree/main ，and  you also download llama-guard3 weight from https://github.com/meta-llama/llama-recipes



# Install Package



conda create -n bullysense python=3.10 -y

conda activate bullysense

pip install -e .



# Quick Start



the usage example is in the finetune_test_generate.py

You only need to modify the picture path(image_prompt).