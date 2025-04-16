import nibabel as ni
from glob import glob 

import os

f1 = ni.load('C:/Users/giaco/Documents/oasis/oasis_3/mri/OAS30001_MR_d2430/sub-OAS30001_ses-d2430_acq-TSE_T2w.nii.gz')

print(f1.shape)