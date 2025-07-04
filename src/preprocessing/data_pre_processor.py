"""
    A class that incorporates the logic of pre processing 
    the dataset of this project prior to the training loop.
"""
import nibabel as nib
import numpy as np
import os

from torch import long

from monai.data import Dataset, DataLoader
from monai.utils import first
from tqdm.auto import tqdm
from monai.transforms import (
	CropForegroundd,
	EnsureTyped,
	EnsureChannelFirstd,
	LoadImaged,
    Flipd,
    Rotated,
	Resized,
	ScaleIntensityd,
    HistogramNormalized,
    NormalizeIntensityd,
    RandScaleIntensityd,
    RandShiftIntensityd,
	Spacingd,
	SpatialPadd,
    ToTensord,
    Compose
)

class PreprocessingOperations: 
    """
        A static class that exposes the pre processing operations allowed 
    """
    STD_PREPROCESS = [
            EnsureChannelFirstd(
                keys='image'
            ),
            EnsureTyped(
                keys='image', 
                dtype=long
            ),
            Spacingd(
                keys='image',
                pixdim=(1.0, 1.0, 1.0),
                mode='bilinear',
                align_corners=True,
                scale_extent=True
            ),
            ScaleIntensityd(
                keys='image', 
                channel_wise=True
            ),
    ]
        
    CROP = CropForegroundd(
			keys='image',
			source_key='image',
			select_fn=(lambda x: x > 0),
			allow_smaller=True
	)

    FLIP = Flipd(
        keys='image',
    )

    CONTRAST = [
        HistogramNormalized(keys='image'),
        #NormalizeIntensityd(keys='image', nonzero=True, channel_wise=True),
	    RandScaleIntensityd(keys='image', factors=0.1, prob=1.0),
	    RandShiftIntensityd(keys='image', offsets=0.1, prob=1.0),
    ]

    CAST_TENSOR = ToTensord(keys='image')

    @staticmethod
    def get_loader(reader:str) -> LoadImaged:
        """
            Provides a loader to read the nifti images. This operations
            must be the first one to be declared when creating the pre-
            processing pipeline, with the monai `Compose` interface.

            ## Args
                - reader (str): the name of the library used to load nifti image.
                                See the same parameter name in monai.transform

            ## Returns
                The reader object 
        """
        return LoadImaged(
            keys='image', 
            reader=reader, 
            image_only=True
        )

    @staticmethod
    def get_rotation(degree:float) -> Rotated:
        """
            Instantiate the rotation object. The angle of the rotation
            is decided by the input parameter.

            ## Args
                - degree (float): the degree of the rotations
            
            ## Returns
                The `Rotated` object
        """
        return Rotated(
            keys='image',
            angle=(degree * np.pi) / 180,
            padding_mode='zeros',
            align_corners=True
        )
    
    @staticmethod
    def get_pad(spatial_size:int) -> SpatialPadd:
        """
            Instantiate the padding object, specifying a spatial size
            that all the images will follow after padding.

            ## Args: 
                - spatial_size (int): the same attribute of `monai.transform.SpatialPadd`. 
                                      In this case the same spatial size is applied to every
                                      dimension of the volumetric images processed by the object
                                      instantiated by this function
            ## Returns
                The `Spatiapadd` object with the input specified spatial size
        """
        return SpatialPadd(
            keys='image', 
            spatial_size=(spatial_size, spatial_size, spatial_size), 
            mode='minimum'
        )
    
    
    @staticmethod
    def get_resize(spatial_size:int):
        """
            Instantiate the `Resized` object.

             ## Args: 
                - spatial_size (int): the same attribute of `monai.transform.Resized`. 
    
            ## Returns
                The `Resized` object with the input specified spatial size
        """
        return Resized(
            keys='image',
            spatial_size=spatial_size,
            size_mode='longest',
            mode='trilinear',
            align_corners=True,
            anti_aliasing=True
        )

    
class DataPreProcessor: 
    """
        A static class that encapsulate the main operations
        performed during the pre processing phase. Every method 
        of `DataPreProcessor` works on a dictionary dataset built 
        with the following keys and constraints:

        ## Rules
            - The 'image' key holds the path of the image to process. After the data is processed
            this key is substituted with the actual image `metatensor`
            - The 'experiment' key holds the the MR session related to the processed image
            - The 'label' key is the target to predict
            - The 'path' key is a duplication of 'image', used to not lose the path information
            after the image is processed
    """
    @staticmethod
    def get_crop_max_shape(dataset_dict:list[dict]) -> tuple:
        """
            Given an input dataset of images to crop, the function
            determines the maximum size of each dimension after all the images
            have been cropped. This evaluation is useful to determine the spatial
            size of eache image in padding and resize operations.

            ## Args
                - dataset_dict (list[dict]): the dictionary dataset
            
            ## Returns 
                A tuple composed by the maximum size of the x, y and z dimension respectevely
        """
        x_lst = []
        y_lst = []
        z_lst = []

        # Concat the operations to execute
        operations = Compose(
            [PreprocessingOperations.get_loader('NibabelReader')] +
            PreprocessingOperations.STD_PREPROCESS + 
            [PreprocessingOperations.CROP] + 
            [PreprocessingOperations.CAST_TENSOR]
        )
        
        # Create the dataset and the loader
        dataset = Dataset(dataset_dict, transform=operations)
        loader = DataLoader(dataset, batch_size=1)

        # Define a progress bar
        progress_bar = tqdm(range(len(loader)), leave=True, desc='Image processing')

        for b in loader:
            shape = b['image'].shape
            
            x_lst.append(shape[2])
            y_lst.append(shape[3])
            z_lst.append(shape[4])

            progress_bar.update(1)

        return (max(x_lst), max(y_lst), max(z_lst))

    @staticmethod
    def get_first(dataset_dict:dict, operations:list=[]) -> tuple:  
        """
            It returns the first preprocessed item in the input dataset.

            ## Args
                - dataset_dict (list[dict]): the dictionary dataset
                - operations (list): a list of preprocessing operations to apply to `dataset_dict`. 
                The list should be built using the exposed operations of `PreprocessingOperations`
            
            ## Returns 
                A tuple composed by
                    - The original first image of the dataset
                    - The preprocessed first image of the dataset
                    - The experiment id related to the first image of the dataset
        """
        # Treat every object as a torch tensor
        operations = Compose(
            [PreprocessingOperations.get_loader('NibabelReader')] + 
            operations + 
            [PreprocessingOperations.CAST_TENSOR]
        )

        # Create the dataset and the loader
        dataset = Dataset(dataset_dict, transform=operations)
        loader = DataLoader(dataset, batch_size=1)

        # Define return
        preproc = nib.Nifti1Image(first(loader)['image'][0, 0, :, :, :].numpy(), np.eye(4))
        orig = nib.load(first(loader)['path'][0])
        experiment = first(loader)['experiment'][0]

        return orig, preproc, experiment 

    @staticmethod
    def process(
        save_dir_name:str, 
        file_name:str,
        dataset_dict:dict,
        description:str,
        pre_process_operations:list=[],
    ) -> None:
        """
            Execute the preprocessing for every instance of the dataset,
            saving the data in the specified input folder.

            ## Args
                - save_dir (str): the name of the directory on which the output is saved
                - file_name (str): the name of the file that will be saved in `save_dir`
                - dataset_dict (list[dict]): the dictionary dataset
                - description (str): a brief description
                - pre_process_operations (list): a list of preprocessing operations to 
                apply to `dataset_dict`.  
        """
        # Create the saving directory if not already created
        if not os.path.isdir(save_dir_name):
          os.mkdir(save_dir_name)

        # Treat every object as a torch tensor
        pre_process_operations = Compose(
            [PreprocessingOperations.get_loader('NibabelReader')] +
            pre_process_operations + 
            [PreprocessingOperations.CAST_TENSOR]
        )
            
        # Create the dataset and the loader
        dataset = Dataset(dataset_dict, transform=Compose(pre_process_operations))
        loader = DataLoader(dataset, batch_size=1)
        
        # Define a progress bar
        progress_bar = tqdm(range(len(loader)), leave=True, desc=description)
        
        for b in loader:
            # Take first value since there is a single batch
            image = b['image'][0]
            experiment = b['experiment'][0]

            volume = image[0, :, :, :].numpy()

            # Define the path where to save the file using the format
            # subject_experiment_label/files.nii.gz
            subject_dir = os.path.join(save_dir_name, experiment)
            save_path = os.path.join(subject_dir, file_name)

            # Create the subject dir if not already created
            if not os.path.isdir(subject_dir):
              os.mkdir(subject_dir)

            # Save data to a nifti file
            nib.save(nib.Nifti1Image(volume, np.eye(4)), save_path)

            progress_bar.update(1)