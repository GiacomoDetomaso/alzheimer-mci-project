import pandas as pd

from sklearn.model_selection import train_test_split


class DataSplitterTrainTest:
    """
        A class that execute the classical train test split with a twist.
        A subject can't be found in both train and test. This class takes
        care of this constraint.
    """
    def __init__(self, subject_experiments:pd.Series, diagnosis:pd.Series):
        """
            ## Args
                - subject_experiments (pd.Series): a series with the experiment to split (MR session)
                - diagnosis (pd.Series): the diagnosis labels associated to `subject_experiments`
        """
        self.subject_experiments = subject_experiments
        self.diagnosis = diagnosis
        

    def get_train_test_set_idx(self) -> tuple:
        """
            Perform the data split.

            ## Returns
                A tuple consisting of
                    - train idx: indeces of the training instances
                    - test idx: indices of the test instances 
                
                The returned index are extracted from the `subject_experiments` and
                `diagnosis` series.
        """
        # Get the data split first and then correct it
        X_train, X_test, y_train, y_test = train_test_split(
            self.subject_experiments,
            self.diagnosis,
            stratify=self.diagnosis,
            random_state=42,
            test_size=0.3
        )
        # Get series made with the experiment's subjects names (index wont change)
        mapping_lambda = lambda s: s[0]
        test_subjects = X_test.str.split('_').map(mapping_lambda).to_frame()
        train_subjects = X_train.str.split('_').map(mapping_lambda).to_list()

        test_subjects['flag'] = test_subjects.map(
            # Create a boolean column which is True only if 
            # the subject x is in the train set too
            lambda x: x in train_subjects
        )

        # Get the index of the instances (inside the test set) to put in the train set
        train_set_instances_idxs = test_subjects[test_subjects['flag']].index

        # From now on it is possible to operate on the y only since 
        # at the end the index will be returned which is the same independently
        # by the fact the "concat" are performed on the X or y

        # Add test subjects that appear in the train set back to it
        y_train = pd.concat([y_train, y_test.loc[train_set_instances_idxs]])

        # After the transfer of these subjects, they can be removed from the test set
        y_test = y_test.drop(index=train_set_instances_idxs)


        # We can return the index since scikit preserved the series structure
        return  y_train.index, y_test.index
    