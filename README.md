# Biomarker-Guided CNNs for
Precision Detection of MCI in Alzheimer’s Disease
- **Dementia**  is currently the seventh leading cause of death, disability and dependency among older people worldwide
- Alzheimer’s disease (AD)  roughly contributes to 70% of the total cases

## AD consequences
- Progressive **cognitive decline** which lead to a **loss of autonomy**
- **Influence**: patient, family members and caregivers

## What this project is about
1. **Region of interest (ROI)** based deep learning approach: a deep learning model that can process the hippocampus MRI as input instead of the whole brain volume
2. **Ternary classification problem**: instead of simply classifying AD vs Cognitevely normal patients, another class, namely MCI or Early-Stage is added to the mix, which makes the AD detection problem extremely challenging
3. **XAI**: Implementation of a XAI technique, namely GradCAM, to check on what the model focuses during its predictions

## The chosen ROI
- One of the first physical symptoms of AD is known as "brain atrophy", which is basically a shrinkage of the brain volume
- The **hippocampus** is one of the first brain areas to exhibit such phenomena

## Proposed pipeline
