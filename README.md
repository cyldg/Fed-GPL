# Federated-Generative-Prompt-Learning
This repo is the official implementation of "Federated Generative Prompt Learning with Vision Foundation Models: Universal Efficient Multi-Center Medical Image Analysis"

## Abstract
Federated medical AI revolutionizes multi-center collaboration, while communication cost, data scarcity, and heterogeneity still limit its practical deployment. Foundation models (FMs) offer a promising avenue for addressing these challenges, owing to their generalization capabilities and efficient adaptability to medical tasks. Here, we present Federated Generative Prompt Learning (Fed-GPL), a universal and efficient framework for multi-center medical image analysis. It collaboratively trains a prompt generator that produces customized prompts for each patient, capturing patient-specific variations and enabling precise medical diagnosis. Fed-GPL is compatible with various vision FMs and medical tasks, like Vision Transformer (ViT) for diabetic retinopathy and melanoma classification, and Segment Anything (SAM) for polyp and prostate segmentation. Fed-GPL outperforms traditional models and full fine-tuning method across classification and segmentation tasks with only 2.8\% and 7.2\% trainable parameters, while converging within just 15 communication rounds. For low-resource settings, Fed-GPL maintains its performance with 5\% of the original training data.

## Acknowledgement

* [segment-anything](https://github.com/facebookresearch/segment-anything)
* [Finetune_segment_anything_tutorial](https://github.com/xzyun2011/finetune_segment_anything_tutorial)
* [invertinggradients](https://github.com/JonasGeiping/invertinggradients)
