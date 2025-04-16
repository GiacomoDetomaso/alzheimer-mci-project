import pandas as pd

from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler

class DataSplitterTrainTest:
    def __init__(self, subject_experiments:pd.Series, diagnosis:pd.Series):
        self.subject_experiments = subject_experiments
        self.diagnosis = diagnosis

    def set_transformations(self, transformations_se:pd.Series, transformations_d:pd.Series):
        self.subject_experiments = pd.concat([self.subject_experiments, transformations_se], ignore_index=True)
        self.diagnosis = pd.concat([self.diagnosis, transformations_d], ignore_index=True)
    
    def __move_files(self):
        pass

    def __rand_undersample_majority_class(self):
        pass

    def split(self):
        # TODO exclude lables
        X_train, X_test, y_train, y_test = train_test_split(
            self.subject_experiments, 
            self.diagnosis, 
            stratify=self.diagnosis,
            random_state=42,
            test_size=0.25
        )

        return X_train, X_test, y_train, y_test

