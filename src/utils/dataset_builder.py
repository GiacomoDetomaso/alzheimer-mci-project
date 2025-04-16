from glob import glob
import os

class MonaiDataSetBuilder:
    def __init__(self, scans_dir_path:str, labels:list):
        self.scans_dir_path = scans_dir_path
        self.labels = labels

    def build(self):
        image_files = glob(self.scans_dir_path)

        return [{'image': img, 'label': label} for img, label in zip(self.scans_dir_path, self.labels)]
