"""
    This module incorporates the logic of plots that 
    are displayed in the notebooks.
"""
import matplotlib.pyplot as plt
import pandas as pd
import nibabel as nib
import os
import numpy as np

from nilearn import plotting

def bar_plot(s: pd.Series, title:str, labels=None):
    """
        Plots fancier bars.

        ## Args
            - s (pd.Series): the numeric series to plot
            - title (str): the title of the plot

        ## Returns
            Displays the plot
    """
    if labels is None:
        labels = []

    s.plot(kind='bar',figsize=(7, 5))
    plt.title(title)

    for i in range(len(s)):
        # Placing text at half the bar height
        plt.text(i, s.values[i] + 15, s.values[i], ha='center')

    if len(labels) != 0:
        plt.xticks([False, True], labels, rotation=90)


def processed_example_comparison_plot(
        original: nib.nifti1.Nifti1Image, 
        preprocessed: nib.nifti1.Nifti1Image, 
        subject:str):
    """
        A function that can be used to compare two volumetric images.
        Its original purpose is to compare the preprocessed version of an image
        with the original one.

        ## Args
            - original (nibabel.nifti1.Nifti1Image): the original image
            - preprocessed (nibabel.nifti1.Nifti1Image): preprocess version of `original`
            - subject (str): the id of the subject represented in these images
        
        ## Returns
            Displays the plot
    """
    if len(original.shape) == 4:
        original = nib.Nifti1Image(original.get_fdata()[0, :, :, :], np.identity(4))
    
    if len(preprocessed.shape) == 4:
        preprocessed = nib.Nifti1Image(preprocessed.get_fdata()[0, :, :, :], np.identity(4))

    plotting.plot_img(original, cmap='gray', title=f'{subject} original');
    plotting.plot_img(preprocessed, cmap='gray', title=f'{subject} preprocessed');


def plot_img_by_experiment_id(dir:str, id:str, file_name_left:str, file_name_right:str):
    """
        A function that plots the left and right hippocampus of a subject's experiment,
        given its id (the MR session)

        ## Args
            - dir (str): the directories that contains the images
            - id (str): the id of the MR session to display

        ## Returns
            Displays the plot
    """
    img_left = nib.load(os.path.join(dir, id, file_name_left))
    img_right = nib.load(os.path.join(dir, id, file_name_right))

    if len(img_left.shape) == 4:
        img_left = nib.Nifti1Image(img_left.get_fdata()[0, :, :, :], np.identity(4))
    
    if len(img_right.shape) == 4:
        img_right = nib.Nifti1Image(img_right.get_fdata()[0, :, :, :], np.identity(4))

    plotting.plot_img(img_left, cmap='gray', title=f'{id} Left');
    plotting.plot_img(img_right, cmap='gray', title=f'{id} Right');