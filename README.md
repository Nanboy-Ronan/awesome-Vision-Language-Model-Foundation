# Awesome Medical Vision Language Model and Foundation Models

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
  - Modality: Chest Xray
  - Datasets: MIMIC, ChexPert
  - <details><summary>Digest</summary>
    This paper introduces **MedCLIP**, a framework designed to overcome the limitations of existing vision-text contrastive learning models like CLIP when applied to the medical domain. Traditional models rely on large-scale paired image-text datasets, which are scarce in medicine. MedCLIP addresses this by decoupling images and texts for multimodal contrastive learning, allowing the use of unpaired data and significantly expanding the training dataset. Additionally, it replaces the standard InfoNCE loss with a semantic matching loss based on medical knowledge to eliminate false negatives in contrastive learning. The framework demonstrates superior performance in zero-shot prediction, supervised classification, and image-text retrieval tasks, outperforming state-of-the-art methods even with a smaller pretraining dataset. :contentReference[oaicite:0]{index=0}
  </details>
</details>

<details>
<summary>PubMedCLIP: How Much Does CLIP Benefit Visual Question Answering in the Medical Domain? (2023)</summary>

- PubMedCLIP: How Much Does CLIP Benefit Visual Question Answering in the Medical Domain? [[paper]](https://aclanthology.org/2023.findings-eacl.88.pdf)
  - Sedigheh Eslami, Christoph Meinel, Gerard de Melo.
  - Modality: Chest X-rays, PET scans, CT scans, MRI, angiography
  - Datasets: the Radiology Objects in COntext (ROCO)
  - <details><summary>Digest</summary>
    This study evaluates the effectiveness of Contrastive Language–Image Pre-training (CLIP) in the medical domain by introducing **PubMedCLIP**, a version of CLIP fine-tuned on image–text pairs from PubMed articles. The authors assess PubMedCLIP's performance on two Medical Visual Question Answering (MedVQA) benchmark datasets, demonstrating that it improves overall accuracy by up to 3% compared to state-of-the-art Model-Agnostic Meta-Learning (MAML) networks trained solely on visual data. The findings suggest that incorporating domain-specific textual and visual information enhances the performance of vision–language models in medical applications.
  </details>
</details>


<details>
<summary>BiomedCLIP: A Multimodal Biomedical Foundation Model Pretrained from Fifteen Million Scientific Image-Text Pairs (2025)</summary>

- BiomedCLIP: A Multimodal Biomedical Foundation Model Pretrained from Fifteen Million Scientific Image-Text Pairs. [[paper]](https://arxiv.org/abs/2303.00915)
  - Sheng Zhang, Yanbo Xu, Naoto Usuyama, Hanwen Xu, Jaspreet Bagga, Robert Tinn, Sam Preston, Rajesh Rao, Mu Wei, Naveen Valluri, Cliff Wong, Andrea Tupini, Yu Wang, Matt Mazzola, Swadheen Shukla, Lars Liden, Jianfeng Gao, Angela Crabtree, Brian Piening, Carlo Bifulco, Matthew P. Lungren, Tristan Naumann, Sheng Wang, Hoifung Poon.
  - Modality: pathology, MRI, Chest Xray, Dermatoscope, BreastUltrasound.
  - Datasets: PMC-15M
  - <details><summary>Digest</summary>
    This paper introduces **BiomedCLIP**, a multimodal biomedical foundation model pretrained on **PMC-15M**, a dataset comprising 15 million biomedical image-text pairs extracted from 4.4 million scientific articles. The authors highlight that PMC-15M is two orders of magnitude larger than existing biomedical multimodal datasets and encompasses a diverse range of biomedical image types. BiomedCLIP incorporates domain-specific adaptations tailored to biomedical vision-language processing. Extensive experiments demonstrate that BiomedCLIP achieves state-of-the-art results across various biomedical imaging tasks, including retrieval, classification, and visual question-answering, outperforming prior approaches. Notably, despite its broad pretraining, BiomedCLIP surpasses radiology-specific models like BioViL in tasks such as RSNA pneumonia detection. The model and dataset are fully open-access, aiming to facilitate future research in multimodal biomedical AI.
  </details>
</details>

<details>
<summary>PMC-CLIP: Contrastive Language-Image Pre-training using Biomedical Documents (2023)</summary>

- PMC-CLIP: Contrastive Language-Image Pre-training using Biomedical Documents. [[paper]](https://arxiv.org/abs/2303.07240)
  - Weixiong Lin, Ziheng Zhao, Xiaoman Zhang, Chaoyi Wu, Ya Zhang, Yanfeng Wang, Weidi Xie.
  - Modality: Colon Pathology, Dermatoscope, Retinal OCT, CT, MRI, X-ray.
  - Datasets: ROCO, MedICaT, MIMIC-CXR
  - <details><summary>Digest</summary>
    This paper introduces **PMC-CLIP**, a vision-language model pretrained on **PMC-OA**, a biomedical dataset comprising 1.6 million image-caption pairs extracted from PubMed Central's Open Access subset. PMC-OA encompasses diverse medical modalities and diseases, with fine-grained alignments between subfigures and subcaptions. PMC-CLIP employs contrastive learning to align visual and textual representations, achieving state-of-the-art performance across various downstream tasks, including image-text retrieval on ROCO, MedMNIST image classification, and medical visual question answering (VQA). Notably, it improves R@10 by 8.1% in image-text retrieval and accuracy by 3.9% in image classification compared to previous methods. The study highlights the potential of large-scale, domain-specific pretraining in advancing biomedical AI applications.
  </details>
</details>

<details>
<summary>Developing Generalist Foundation Models from a Multimodal Dataset for 3D Computed Tomography (2024)</summary>
- Developing Generalist Foundation Models from a Multimodal Dataset for 3D Computed Tomography. [[paper]](https://arxiv.org/abs/2403.17834)
  - Ibrahim Ethem Hamamci, Sezgin Er, Furkan Almas, Ayse Gulnihan Simsek, Sevval Nil Esirgun, Irem Dogan, Muhammed Furkan Dasdelen, Omer Faruk Durugol, Bastian Wittmann, Tamaz Amiranashvili, Enis Simsar, Mehmet Simsar, Emine Bensu Erdemir, Abdullah Alanbay, Anjany Sekuboyina, Berkan Lafci, Christian Bluethgen, Mehmet Kemal Ozdemir, Bjoern Menze.
  - Modality: Chest CT.
  - Datasets: CT-RATE
  - <details><summary>Digest</summary>
    This study introduces **CT-RATE**, the first dataset pairing 3D chest CT scans with corresponding radiology reports, comprising 25,692 non-contrast 3D chest CT scans from 21,304 unique patients, expanded to 50,188 volumes through various reconstructions. Leveraging this dataset, the authors develop **CT-CLIP**, a contrastive language-image pretraining framework designed for broad applications without the need for task-specific training. CT-CLIP outperforms state-of-the-art fully supervised models in multi-abnormality detection and efficiently retrieves relevant cases using image or textual queries. Additionally, by combining CT-CLIP's vision encoder with a pretrained large language model, the authors create **CT-CHAT**, a vision-language foundational chat model for 3D chest CT volumes, fine-tuned on over 2.7 million question-answer pairs derived from the CT-RATE dataset. CT-CHAT surpasses other multimodal AI assistants, underscoring the necessity for specialized methods in 3D medical imaging. The open-source release of CT-RATE, CT-CLIP, and CT-CHAT aims to address critical challenges in 3D medical imaging and lay the groundwork for future innovations in medical AI and improved patient care.
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
### Trustworthiness
#### Fairness
<details>
<summary>Fair Text to Medical Image Diffusion Model with Subgroup Distribution Aligned Tuning (2024)</summary>

- Fair Text to Medical Image Diffusion Model with Subgroup Distribution Aligned Tuning. [[paper]](https://arxiv.org/abs/2406.14847)
  - Xu Han, Fangfang Fan, Jingzhao Rong, Zhen Li, Georges El Fakhri, Qingyu Chen, Xiaofeng Liu.
  - Key Words: Text-to-Image Generation; Medical Imaging; Diffusion Models; Bias Mitigation; Subgroup Distribution Alignment.
  - <details><summary>Digest</summary>
    This study addresses biases in text-to-medical image (T2MedI) generation models, particularly concerning underrepresented subgroups in training datasets. The authors develop a T2MedI model based on the pre-trained Imagen framework, fine-tuning it with medical images from the Radiology Objects in Context (ROCO) dataset. They identify gender bias in the generated images and propose a Subgroup Distribution Aligned Tuning (SDAT) method to mitigate this issue. SDAT fine-tunes the model to align the distribution of sensitive subgroups in generated images with those in a target dataset, guided by a sensitivity-subgroup classifier and maintained through a CLIP-consistency regularization term. Evaluation using the BraTS18 dataset demonstrates that SDAT effectively reduces gender representation inconsistencies in generated brain MR images, aligning them more closely with the target dataset's distribution. :contentReference[oaicite:0]{index=0}
  </details>
</details>
#### Privacy
#### Security

## Tasks, Datasets and Metrics
### Report Generation
<details>
<summary>MIMIC-CXR: A De-identified Publicly Available Database of Chest Radiographs with Free-Text Reports (2019)</summary>

- MIMIC-CXR: A De-identified Publicly Available Database of Chest Radiographs with Free-Text Reports. [[paper]](https://www.nature.com/articles/s41597-019-0322-0)
  - Alistair E. W. Johnson, Tom J. Pollard, Seth J. Berkowitz, Nathaniel B. Greenbaum, Matthew P. Lungren, Chih-ying Deng, Roger G. Mark, Steven Horng.
  - Modality: Chest Radiographs
  - <details><summary>Digest</summary>
    This paper introduces **MIMIC-CXR**, a large-scale, de-identified dataset comprising 377,110 chest X-ray images associated with 227,827 imaging studies from 64,588 patients at the Beth Israel Deaconess Medical Center between 2011 and 2016. Each study includes free-text radiology reports, providing a rich resource for developing, evaluating, and comparing machine learning algorithms in medical imaging. The dataset is part of the MIMIC family and is freely accessible, promoting advancements in automated image interpretation and facilitating reproducible research in the medical imaging community.
  </details>
</details>

### VQA
<details>
<summary>A Dataset of Clinically Generated Visual Questions and Answers About Radiology Images (2018)</summary>

- A Dataset of Clinically Generated Visual Questions and Answers About Radiology Images. [[paper]](https://www.nature.com/articles/sdata2018251)
  - Jason J. Lau, Soumya Gayen, Asma Ben Abacha, Dina Demner-Fushman.
  - Modality: CT, x-ray, T2 weighted MRI.
  - <details><summary>Digest</summary>
    This paper introduces a dataset comprising 15,292 clinically generated visual questions and answers (VQA) related to radiology images. The dataset includes a variety of question types, such as modality, plane, abnormality, and attribute, each associated with corresponding radiology images and expert-provided answers. This resource aims to facilitate the development and evaluation of VQA systems in the medical domain, promoting advancements in automated image interpretation and clinical decision support.
  </details>
</details>

### Captioning
<details>
<summary>BIOMEDICA: An Open Biomedical Image-Caption Archive, Dataset, and Vision-Language Models Derived from Scientific Literature (2025)</summary>

- BIOMEDICA: An Open Biomedical Image-Caption Archive, Dataset, and Vision-Language Models Derived from Scientific Literature. [[paper]](https://arxiv.org/abs/2501.07171)
  - Alejandro Lozano, Min Woo Sun, James Burgess, Liangyu Chen, Jeffrey J. Nirschl, Jeffrey Gu, Ivan Lopez, Josiah Aklilu, Austin Wolfgang Katzer, Collin Chiu, Anita Rau, Xiaohan Wang, Yuhui Zhang, Alfred Seunghoon Song, Robert Tibshirani, Serena Yeung-Levy.
  - Modality: pathology, MRI, Chest Xray, Dermatoscope, BreastUltrasound, Fluresence Microscopy, Electron Microscopy, Light Microscopy, Microscopy, Laparoscopic Surgery.
  - <details><summary>Digest</summary>
    This paper introduces **BIOMEDICA**, a comprehensive, open-source framework designed to extract, annotate, and serialize the entire PubMed Central Open Access subset into an accessible dataset. The resulting archive comprises over 24 million unique image-text pairs from more than 6 million articles, encompassing a wide range of biomedical knowledge. The authors also present **BMCA-CLIP**, a suite of CLIP-style models pre-trained on the BIOMEDICA dataset via streaming, eliminating the need for extensive local storage. These models demonstrate state-of-the-art performance across 40 tasks in various biomedical fields, including pathology, radiology, ophthalmology, dermatology, and more, excelling in zero-shot classification and image-text retrieval tasks. The BIOMEDICA framework and models aim to advance research in biomedical vision-language applications by providing a diverse and extensive dataset for training and evaluation.
  </details>
</details>


### Diagnosis
### Img-txt Retrieval
### Embedding Extractor

## Challenges and Future Direction
