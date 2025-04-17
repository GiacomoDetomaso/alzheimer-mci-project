from genericpath import isdir
import os
import pandas as pd
import random
import shutil

from glob import glob

class DatasetHelper:

    @staticmethod
    def undersample_majority_class(subject_experiments:pd.Series, diagnosis:pd.Series, final_size:int):        
        # TODO raise error if class is not series
        
        # Get majority class
        majority_class = diagnosis.value_counts().idxmax()   
        
        # Consider only majority class instances
        majority_labels = diagnosis[diagnosis == majority_class]
        other_labels = diagnosis[diagnosis != majority_class]
        resampled = majority_labels.sample(n=final_size, random_state=42)
        
        # Build the new dataset with the majority class resampled
        new_diagnosis = pd.concat([resampled, other_labels]).sort_values()
        
        return new_diagnosis.index
    
    @staticmethod
    def move_folders(subject_experiments:list, base_dir:str, dst_dir:str):
        if not os.path.isdir(dst_dir):
            os.mkdir(dst_dir)
            
        for subject in subject_experiments:
            src_dir = os.path.join(base_dir, subject)
            dst_copy = os.path.join(dst_dir, subject)
            
            shutil.copytree(src_dir, dst_copy)
        
    @staticmethod
    def get_dataset(images:str, labels:list):
        image_files = sorted(glob(images))

        return [
            {'image': image, 'label': label}
            for image, label in zip(image_files, labels)
        ]