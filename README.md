# Multimodal-Image-Text-Match-Classifier

## Introduction

We implement an Image-Text Match (ITM) Classifier, optimising for a binary prediction for positive caption and image pairs from the Flickr8k dataset. This project serves as an introductory exploration of multimodal models and fusion techniques, as such, no significant effort is made to maximise model performance. With significant focus being targeted at multimodal models within the SOTA model landscape, this work is motivated by an initial investigation into the architectural considerations/methods applied for such models.

## Dataset

We utilise a subset of the Flickr8k dataset. The training and test datasets were formed by sampling one positive and one random caption for each image, resulting in a 1:1 positive to negative pair ratio. No efforts were made to construct 'hard negative' samples; this would form a basis for future work, however, it falls outside the primarily architectural focus of this study. Positive samples are labelled with 1 and negative samples with 0.

We utilise an ~70:15:15 Train to Test to Validation dataset size; however, as no hyperparameter optimisation was performed, we do not use the validation set. We ensure no data leakage by keeping Training and Test set samples mutually exclusive at the image level so the model only sees test samples once, at inference.

## Method
### Model Architecture

The model consists of a Transformer encoder text embedding head, ResNet image embedding head, and an MLP classification head. We use pre-trained text and image embedding heads (Hugging Face's all-MiniLM-L6-v2 and PyTorch's resnet18 respectively) and freeze their weights such that all parameter updates are done within the MLP. The use of pre-trained encoders is due to the small size data subsets used here (training set of 2000 samples).

### Modality Fusion

A late fusion approach is taken, where the embeddings from each head are normalised using L2-Norm separately, then concatenated per sample. This ensures that upon concatenation, no singular modality dominates the embedding space. This ensures that we avoid any single modality dominating the feature space. While late fusion limits the ability to learn deep cross-modal interactions, the simplicity of the task and the use of frozen pretrained encoders make this approach well-suited to the scope of this study.

### Training

An Adam optimiser is utilised with a training batch size of 50 (resulting in 40 batches per epoch); as mentioned, only the MLP parameters were trainable with all other model parameters frozen at training and inference. An epoch number of 5 was used; we do so as a baseline with the possibility of implementation of early stopping for future work. We optimise a cross-entropy loss that utilises the final model logits of size (B,2). This is the standard approach for binary classification.

## Evaluation

As we implemented a 50:50 positive:negative pair ratio in both the test and train sets, test set accuracy was chosen as the primary evaluation metric. As mentioned earlier, we ensured no data leakage, being careful to ensure the model only ever sees the test set samples once, at inference time. To investigate the impact of multimodal data on this task, we perform an ablation study, removing both the text and image encoder heads independently. This was done by replacing one head’s embeddings with zero values of the same shape. This was then repeated with the other head swapped out for zero encodings. This method was deemed suitable here to accommodate the scope of study; however, it is noted that the use of zero embeddings may introduce OOD samples per study which could slightly lower performance for each study. We deemed this an insignificant impact, as although it is unlikely for all zero embeddings to occur naturally, we are primarily interested in the scale of performance difference, not specifically single embedding robustness of the model.

## Results

The complete model produced an inference accuracy of 73.5%, with the text and image embedding-only studies producing accuracies of 48.9% and 50.0% respectively. The model performs well, considering no fine-tuning was conducted and the absence of 'hard negative' pairs during training. Additionally, we see that both modalities significantly improve the model's performance, with each study showing essentially random guessing. This allows us to conclude that the late fusion method was successful in learning interactions between the embeddings, as the model is not learning a 'shortcut' from any one modality.
