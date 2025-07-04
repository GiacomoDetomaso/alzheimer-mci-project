import src.preprocessing.data_pre_processor as dp
import src.utils.dataset_helper as dh
import os
import importlib

importlib.reload(dp)

preprocessor = dp.DataPreProcessor
operations = dp.PreprocessingOperations
helper = dh.DatasetHelper

default_operations = lambda size: operations.STD_PREPROCESS + [
        operations.CROP, 
        operations.get_pad(size[0]),
        operations.get_resize(size[1]), 
    ]

def get_datasets(
        base_dir:str, 
        experiments:list, 
        labels:list,
        left_file_name:str,
        right_file_name:str
    ) -> dict:
    # Remove the suffix from the experiment related to data augmentation,
    # in order to correctly find the path of the images
    experiments_fixed = [
        s.removesuffix('_' + s.split('_')[-1]) 
            if s.split('_')[-1] in ['flip', 'rot45', 'rot30', 'rot90', 'rot60'] 
            else s  
        for s in experiments 
    ]

    # Get the paths of the images related to the input experiments 
    left_paths = helper.get_image_paths_by_experiments(
        base_dir=base_dir, 
        relative_file_path=os.path.join('mri', left_file_name),
        experiments=experiments_fixed
    )

    right_paths = helper.get_image_paths_by_experiments(
        base_dir=base_dir, 
        relative_file_path=os.path.join('mri', right_file_name),
        experiments=experiments_fixed
    )

    left_dataset = helper.get_dataset_dict_preprocessing(sorted(left_paths), experiments, labels)
    right_dataset = helper.get_dataset_dict_preprocessing(sorted(right_paths), experiments, labels)

    return {'left': left_dataset, 'right': right_dataset}


def test_preprocessing(single_dataset:dict, operations:list=[], ):
    return preprocessor.get_first(single_dataset, operations)


def max_size_after_crop(train_data:dict, val_data:dict, test_data:dict):
    print('Training set')
    x_left_train, y_left_train, z_left_train = preprocessor.get_crop_max_shape(train_data['left'])
    x_right_train, y_right_train, z_right_train = preprocessor.get_crop_max_shape(train_data['right'])

    print('Validation set')
    x_left_val, y_left_val, z_left_val = preprocessor.get_crop_max_shape(val_data['left'])
    x_right_val, y_right_val, z_right_val = preprocessor.get_crop_max_shape(val_data['right'])

    print('Test set')
    x_left_test, y_left_test, z_left_test = preprocessor.get_crop_max_shape(test_data['left'])
    x_right_test, y_right_test, z_right_test = preprocessor.get_crop_max_shape(test_data['right'])

    max_x = max([x_left_train, x_right_train, x_left_test, x_right_test, x_left_val, x_right_val])
    max_y = max([y_left_train, y_right_train, y_left_test, y_right_test, y_left_val, y_right_val])
    max_z = max([z_left_train, z_right_train, z_left_test, z_right_test, z_left_val, z_right_val])

    return max(max_x, max_y, max_z) 


def execute_pre_processing(
        save_dir_name:str, 
        datasets:dict, 
        left_file_name:str,
        right_file_name:str,
        operations:list=[]
    ):

    preprocessor.process(
            save_dir_name=os.path.join('..', 'data', save_dir_name), 
            file_name=left_file_name, 
            dataset_dict=datasets['left'],
            description='Processing left hippocampus', 
            pre_process_operations=operations,
    )

    preprocessor.process(
        save_dir_name=os.path.join('..', 'data', save_dir_name), 
        file_name=right_file_name, 
        dataset_dict=datasets['right'], 
        description='Processing right hippocampus',
        pre_process_operations=operations,
    )