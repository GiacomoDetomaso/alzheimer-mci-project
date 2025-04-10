import os

from abc import ABC, abstractmethod

class OasisDataManagerInterface(ABC):
    def __init__(self, output_dir, project_id='OASIS3'):
        """
            Manager of OASIS Interface

            ## Args:
                - output_dir (str): the path of the directory on which the fill will be downloaded
                - project_id (str): default OASIS3 (OASIS4 not available at the moment)
        """
        self.output_dir = output_dir
        self.project_id = project_id
        self.base_url = 'https://www.nitrc.org/ir/data/archive/projects/'

        if not os.path.isdir(self.output_dir):
            os.mkdir(self.output_dir)

    @abstractmethod
    def download(self, subjects_file_path:str, username:str, password:str):
        pass

    @abstractmethod
    def clean(self, base_empty_dir):
        pass