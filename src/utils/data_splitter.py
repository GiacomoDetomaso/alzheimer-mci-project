from random import Random
import pandas as pd
import random

from sklearn.model_selection import train_test_split

class DataSplitterTrainTest:
    def __init__(self, subject_experiments:pd.Series, diagnosis:pd.Series):
        self.subject_experiments = subject_experiments
        self.diagnosis = diagnosis
        
        # Basic splitting
        #self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
        #    subject_experiments, 
        #    diagnosis, 
        #    stratify=diagnosis,
        #    random_state=42,
        #    test_size=0.25
        #)
    
    def set_transformations(self, transformations_se:pd.Series, transformations_d:pd.Series):
        self.subject_experiments = pd.concat([self.subject_experiments, transformations_se], ignore_index=True)
        self.diagnosis = pd.concat([self.diagnosis, transformations_d], ignore_index=True)
    
    def __move_files(self):
        pass

    def __check_subjects_distribution(X_train, X_test, ):
        pass
    
    def get_basic_train_set():
        pass

    def get_test_set(self):
        for t in self.X_train:
            subject = t.split('_')[0]
            
            return         
