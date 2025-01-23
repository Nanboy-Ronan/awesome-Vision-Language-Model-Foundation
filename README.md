# Awesome Vision Language Model and Foundation Models

[![Maintenance](https://img.shields.io/badge/Maintained%3F-YES-green.svg)](https://github.com/Nanboy-Ronan/awesome-Vision-Language-Model-Foundation/graphs/commit-activity)
![PR Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen)
![ ](https://img.shields.io/github/last-commit/Nanboy-Ronan/awesome-Vision-Language-Model-Foundation)
[![GitHub stars](https://img.shields.io/github/stars/Nanboy-Ronan/awesome-Vision-Language-Model-Foundation?color=blue&style=plastic)](https://github.com/Nanboy-Ronan/awesome-Vision-Language-Model-Foundation/stargazers)
[![GitHub watchers](https://img.shields.io/github/watchers/Nanboy-Ronan/awesome-Vision-Language-Model-Foundation?color=yellow&style=plastic)](https://github.com/Nanboy-Ronan/awesome-Vision-Language-Model-Foundation)
[![GitHub forks](https://img.shields.io/github/forks/Nanboy-Ronan/awesome-Vision-Language-Model-Foundation?color=red&style=plastic)](https://github.com/Nanboy-Ronan/awesome-Vision-Language-Model-Foundation/watchers)
[![GitHub Contributors](https://img.shields.io/github/contributors/Nanboy-Ronan/awesome-Vision-Language-Model-Foundation?color=green&style=plastic)](https://github.com/Nanboy-Ronan/awesome-Vision-Language-Model-Foundation/network/members)


## Table of Contents

## Survey

<details>
<summary>Survey: 2024</summary>

- Vision-Language Models for Medical Report Generation and Visual Question Answering: A Review. [[paper]](https://arxiv.org/abs/2403.02469)
  - Iryna Hartsock, Ghulam Rasool.
  - Key Words: Medical AI; Vision-Language Models; Medical Report Generation; Visual Question Answering; Multimodal Learning.
  - <details><summary>Digest</summary>
    This review examines recent advancements in medical vision-language models (VLMs) that integrate computer vision and natural language processing to analyze visual and textual medical data. The authors focus on models designed for medical report generation and visual question answering (VQA), providing background on NLP and CV integration techniques. Key areas discussed include medical vision-language datasets, architectures, pre-training strategies, and evaluation metrics. The paper also highlights current challenges, such as enhancing clinical validity and addressing patient privacy concerns, and proposes future research directions to improve healthcare applications.
  </details>
</details>


<details>
<summary>Survey: 2023</summary>

- Medical Vision Language Pretraining: A Survey. [[paper]](https://arxiv.org/abs/2312.06224)
  - Prashant Shrestha, Sanskar Amgain, Bidur Khanal, Cristian A. Linte, Binod Bhattarai.
  - Key Words: Medical Vision-Language Pretraining; Self-Supervised Learning; Multimodal Learning; Medical Imaging; Natural Language Processing.
  - <details><summary>Digest</summary>
    This survey delves into the emerging field of Medical Vision Language Pretraining (VLP), which addresses the scarcity of labeled data in the medical domain by leveraging both visual and textual data through self-supervised learning. The authors review existing works, categorizing them based on pretraining objectives, architectures, evaluation tasks, and datasets used. They discuss current challenges in medical VLP, such as data scarcity, model interpretability, and the need for standardized evaluation metrics. The paper concludes by highlighting future directions, emphasizing the importance of developing more robust models and exploring diverse medical datasets to enhance the applicability of VLP in healthcare.
  </details>
</details>

## Existing Medical VLM and Foundation Models
### CLIP and Variants
<details>
<summary>MedCLIP: Contrastive Learning from Unpaired Medical Images and Text (2022)</summary>

- MedCLIP: Contrastive Learning from Unpaired Medical Images and Text. [[paper]](https://arxiv.org/abs/2210.10163)
  - Zifeng Wang, Zhenbang Wu, Dinesh Agarwal, Jimeng Sun.
  - Key Words: Medical AI; Contrastive Learning; Unpaired Data; Vision-Language Models; Self-Supervised Learning.
  - <details><summary>Digest</summary>
    This paper introduces **MedCLIP**, a framework designed to overcome the limitations of existing vision-text contrastive learning models like CLIP when applied to the medical domain. Traditional models rely on large-scale paired image-text datasets, which are scarce in medicine. MedCLIP addresses this by decoupling images and texts for multimodal contrastive learning, allowing the use of unpaired data and significantly expanding the training dataset. Additionally, it replaces the standard InfoNCE loss with a semantic matching loss based on medical knowledge to eliminate false negatives in contrastive learning. The framework demonstrates superior performance in zero-shot prediction, supervised classification, and image-text retrieval tasks, outperforming state-of-the-art methods even with a smaller pretraining dataset. :contentReference[oaicite:0]{index=0}
  </details>
</details>

### Text2Image Models

### Language Modeling

### Trasitional Vision Language Models

### Vision Large Language Models

### Vision Language Model with Other Modalities


## Research Areas
### Architecture
### Training
### Inference Adjustment
### Combination

## Tasks, Datasets and Metrics
### Report Generation
### VQA
### Captioning
### Diagnosis
### Img-txt Retrieval
### Embedding Extractor

## Challenges and Future Direction
