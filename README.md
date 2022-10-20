ECPEC (Emotion-Cause Pair Extraction in Conversations)
=========

## Overview
>In this repo, we put the data and code of a new task named emotion-cause pair extraction in conversations. The target of this ECPEC task is to extract all the possible emotion-cause pairs in conversations. 
> We build a new dataset entitled ConvECPE on the basis of IEMOCAP dataset. Besides, we also developed a two-step baseline model for ECPEC task. In the first step, all the possible emotion and cause utterances are jointly extracted. After that,
> the extracted emotion and cause utterances are paired into emotion-cause pairs for final binary classification. This is the code for JointEC framework. It consists of four documents including source dataset, Joint-Xatt(step1 model), Joint-GCN(step1 model) and Joint-EC(step 2 model).

Detailed description of this project will come soon...

<img src="https://github.com/Maxwe11y/JointEC/blob/main/model_p2_v2-1.png" width = 55% height = 55% div align=center />

<!-- <figure class="half">
  <img src="https://github.com/Maxwe11y/JointEC/blob/main/model_step_1_10-1.png" width = 45% height = 45% div align=left />
  <img src="https://github.com/Maxwe11y/JointEC/blob/main/model_p2_v2-1.png" width = 45% height = 45% div align=right />
</figure> -->
  
## ConvECPE Dataset

## Uasge
In order to implement the proposed two-step framework, you have to download the pre-trained GloVe vectors(glove.6B.100d.txt is the most commonly used vectors in this project).
The downloaded GloVe vectors should be placed in the dir of both step 1 and step 2 models(Joint-EC, Joint-GCN, Joint-Xatt).

👉 Check out [GloVe Embeddings](https://nlp.stanford.edu/data/glove.6B.zip) before you run the **code**.

Detailed description of this project will come soon...
