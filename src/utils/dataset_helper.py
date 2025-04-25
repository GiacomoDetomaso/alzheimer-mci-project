import os
import pandas as pd
import shutil

class DatasetHelper:
    """
        A static class that provides helper functions to help building
        the torch dataset that will be fed to DL models.
    """
    @staticmethod
    def undersample_majority_class(diagnosis:pd.Series, final_size:int) -> pd.Index:
        """
            A method to undersample the majority class instances. Since 
            the whole project deals with images, a simple random undersample
            is performed.

            ## Args
                - diagnosis (pd.Series): the series of class labels (one for each MR session)
                - final_size (int): the final size of the sample
            
            ## Returns
                The index of `diagnosis`, which indicates the instances to keep
        """        
        # TODO raise error if class is not series
        
        # Get majority class
        majority_class = diagnosis.value_counts().idxmax()   
        
        # Consider only majority class instances
        majority_labels = diagnosis[diagnosis == majority_class]
        other_labels = diagnosis[diagnosis != majority_class]
        resampled = majority_labels.sample(n=final_size, random_state=42)
        
        # Build the new dataset with the majority class resampled
        # it is safe to do this since the index from "majority_labels"
        # and "other_labels" can't overlap
        new_diagnosis = pd.concat([resampled, other_labels]).sort_values()
        
        return new_diagnosis.index
    
    @staticmethod
    def move_folders(subject_experiments:list, base_dir:str, dst_dir:str):
        """
            A method to copy to move a folder from a place to another. 
            The moved folders are copied in the destination path.

            ## Args
                - subject_experiments (list): the list of MR sessions
                - base_dir (str): the directory that contains the folders to move,
                where the name of the latter ones is defined by `subject_experiments`
                - dst_dir (str): the directory on which the folders from `base_dir` will be transferred
        """
        if not os.path.isdir(dst_dir):
            os.mkdir(dst_dir)
        
        if len(os.listdir(dst_dir)) == 0:     
            for subject in subject_experiments:
                src_dir = os.path.join(base_dir, subject)
                dst_copy = os.path.join(dst_dir, subject)

                shutil.copytree(src_dir, dst_copy)
        else:
            print("Destination directory is not empty!")

    @staticmethod
    def augment_dataframe(df:pd.DataFrame, id_col_name:str, suffix_list:list) -> list:
        """
            Adds artificial instances to the input dataframe

            ## Args
                - df (pd.DataFrame): the dataframe to augment
                - id_col_name (str): the id of the dataframe (the column should be string)
                - suffix_list (list): a list of suffix. For every exsisting instance N new versions
                                      of the same ones will be created according to `len(suffix_list)` A new version of 
                                      an exsisting instance will consist of a new id value in `id_col_name`m with 
                                      all the other attributes inherited by the original version of the 
                                      instance 

            ## Return
                The augmented dataframe:
                    - Each new instance is identified by the modified `id_col_name` value
                    on which it is added the suffix from `suffix_list`
        """
        dfs = []

        for suffix in suffix_list:
            df_aug = df.copy(deep=True)
            df_aug[id_col_name + '_' + 'original'] = df_aug[id_col_name]
            df_aug[id_col_name] = df_aug[id_col_name].map(lambda s: s + '_' + suffix)

            dfs.append(df_aug)

        return dfs
    
    @staticmethod
    def extract_augmentation_column(col:pd.Series) -> pd.Series:
        """
            It extracts from the input series the last substring obtained
            by splitting at '_'

            ## Args
                col (pd.Series): the series to process

            ## Returns
                A series whose values are  the last substrings obtained from `col` by splitting at '_'
        """
        return   col \
                    .str \
                    .split('_') \
                    .map(lambda x: x[-1] if not x[-1].startswith('d') else 'normal')
    
        
    @staticmethod
    def get_dataset_dict(images_path:list, experiments:list, labels:list) -> list[dict]:
        """
            It returns a dictionary dataset with the format required.

            ## Args
                - images_path (list): the list of paths to add to the dataset under the 'image' key
                - experiments (list): the list of MR session to add to the dataset under the 'experiment' key
                - labels (list): the list of labels to add to the dataset under the 'label' key

            ## Returns
                The dataset to feed to torch-like models
        """
        return [
            {'image': image, 'experiment': experiment, 'label': label, 'path': image}
            for image, experiment, label in zip(images_path, experiments, labels)
        ]
    
    
    @staticmethod
    def get_image_paths_by_experiments(base_dir:str, relative_file_path: str, experiments:list)->list:
        """
            This function returns a list of paths of images that corresponds to the input
            experiments.

            ## Args
                - base_dir (str): the path of the directory where to search the images
                - relative_file_path (str): the path of the image relative to `base_dir`
                - experiments (list): a list of MR sessions

            ## Returns 
                A list of path of images related to the input experiments
        """
        return [
            os.path.join(base_dir, exp, relative_file_path) 
            for exp in os.listdir(base_dir) 
            if exp in experiments
        ]
