import os
import shutil
import requests

from requests.auth import HTTPBasicAuth
from zipfile import ZipFile

class DataManagerMRI:
    def __init__(self, freesurfer_output_dir:str):
        self.output_dir = freesurfer_output_dir
        self.PROJECT_ID = 'OASIS3'
        self.BASE_URL = 'https://www.nitrc.org/ir/data/archive/projects/'
        
        if not os.path.isdir(freesurfer_output_dir):
            os.mkdir(freesurfer_output_dir)

    def __build_download_url(self, subjects, experiments, assessors):
        args = [
            "/subjects/", 
            subjects, 
            "/experiments/", 
            experiments, 
            "/assessors/",
            assessors, 
            "/files?format=zip"
        ]

        args = ''.join(args)

        return self.BASE_URL + self.PROJECT_ID + args
    
    def __extraction_post_process(
            self,
            fs_id:str, 
            experiment_label:str, 
            files_to_keep:list[str]=[]
        ):

        source = os.path.join(self.output_dir, fs_id, 'out', 'resources', 'DATA', 'files', experiment_label, 'mri')
        dest = os.path.join(self.output_dir, experiment_label)

        # Overwrite the folder if it already exsist
        if os.path.isdir(dest):
            shutil.rmtree(dest)

        # Create the destination folder
        os.mkdir(dest)
        
        # Move the mri folder inside the new folder
        shutil.move(source, dest)

        # Remove all the content previously stored in download_path/freesurfer_id
        shutil.rmtree(os.path.join(self.output_dir, fs_id))

        # Remove all the files whose name do not match with files_to_keep list
        if len(files_to_keep) > 0:
            mri_folder = os.path.join(dest, 'mri')

            for file in os.listdir(mri_folder):
                if file not in files_to_keep:
                    object_to_remove_path = os.path.join(mri_folder, file)
                    
                    # The mri folder can contain both files and directories
                    if os.path.isfile(object_to_remove_path):
                        os.remove(object_to_remove_path)
                    elif os.path.isdir(object_to_remove_path):
                        shutil.rmtree(object_to_remove_path)
                    
                    print(f'- Removed: {object_to_remove_path}')

        # Lastly the zip file can be removed
        os.remove(os.path.join(self.output_dir, experiment_label + '.zip'))


    def download(self, subjects_file_path:str, username:str, password:str, files_to_keep:list=[]):
        fs_id_list = []

        # Obtain the list of subjects to download
        with open(subjects_file_path, 'r') as f: 
            # Use [:-1] to avoid /n at the end of each freesufer id
            fs_id_list = [line[:-1] for line in f.readlines()]

        if len(fs_id_list) > 0:
            print("Session started correctly")

            with requests.Session() as session:
                session.auth = HTTPBasicAuth(username, password)
                
                for fs_id in fs_id_list:
                    fs_id_splitted = fs_id.split('_')

                    subject_id = fs_id_splitted[0]
                    time_id = fs_id_splitted[2]
                    experiment_label = f'{subject_id}_MR_{time_id}'

                    print(f"- Starting download for: {fs_id}")

                    download_url = self.__build_download_url(subject_id, experiment_label, fs_id)
                    download_response = session.get(download_url)

                    if download_response.status_code == 200:
                        output_path = os.path.join(self.output_dir, experiment_label)

                        # The get request is successful and the zip file bytes can be written
                        with open(f'{output_path}.zip', 'wb') as f:
                            f.write(download_response.content)

                        print(f'- ✅ File {output_path}.zip downloaded') 

                        # Extract data inside the zip file
                        with ZipFile(f'{output_path}.zip', 'r') as zf:
                            zf.extractall(path=self.output_dir) 
                        
                        print("- Zip file extracted. Starting post-processing.")

                        # Apply the post processing function in order to extract the mri folder
                        # and the files wanted by the user located inside it
                        self.__extraction_post_process(fs_id, experiment_label, files_to_keep)
                    else:
                        print(download_url)  
        else:
            raise Exception("""There must be at list one subject freesurfer id in 
                               the input csv specified with output_dir paramater. Got None.""")

    def clean_freesurfers(self, base_empty_dir, mri_dir):
        """
            # TODO doc
        """
        # Create an empty dir
        if not os.path.isdir(base_empty_dir):
            os.mkdir(path=base_empty_dir)

        for d in os.listdir(self.output_dir):
            # The folder to move
            source_path = os.path.join(self.output_dir, d)

            mri_path = os.path.join(source_path, mri_dir)

            # len == 1 for empty folder since there's always a file called .xdebug_mris_calc
            if len(os.listdir(mri_path)) == 1:
                destination_path = os.path.join(base_empty_dir, d)
                shutil.move(src=source_path, dst=destination_path)

                print(f"Empty: {mri_path}")    