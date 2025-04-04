# Download freesurfers of OASIS-3 T1w MRI

The script downaloads the freesurfer's MRI folder of the OASIS-3 dataset, which contains various segmentation of the T1w scans. Moreover the script let you specify which file you want to keep, discarding the unspecified ones (improving disk space).

The proposed script is a modified version of the script found in the download_freesurfer folder, of the following project: https://github.com/NrgXnat/oasis-scripts/tree/master/download_freesurfer

## Dependencies

- curl (already installed with most distro)
- zip

## Required inputs 

- <input_file.csv> - A Unix formatted, comma-separated file containing a column for freesurfer_id (e.g. OAS30001_Freesurfer53_d0129)
- <directory_name> - A directory path (relative or absolute) to save the Freesurfer files
- <nitrc_ir_username> - Your NITRC IR username used for accessing OASIS data on nitrc.org/ir 
(you will be prompted for your password before downloading)
- <files_to_keep> - A list of files to keep inside the downloaded mri folder (all the others are discarded). The files shoud be specified using the following format: "f1.mgz f2.mgz f3.mgz...".         
    - Note: If the parameter is not specified all files will be kept

## Usage

- Open the terminal and reach the folder where the script is located;
- The folder should contain:
    - The download_oasis_freesurfer_mri.sh
    - A unix formatted csv filed with the id of the freesurfers to download
- Call the script using the command `./download_oasis_freesurfer.sh p1 p2 p3 p4`
- $p_i$ represent script parameters 

## Download folder structer

The folder selected to download data will contain various subfolders. Example:
- OAS30001_MR_d0129
    - mri
        - files

### Example 1: download full MRI folder

`./download_oasis_freesurfer.sh subjects.csv download_folder nitrc_useraname`

After being prompted to input nitrc password the download of the freesurfers will start. Downloaded files will be located in the "download_folder".

### Example 1: download certain files from MRI folder

`./download_oasis_freesurfer.sh subjects.csv download_folder nitrc_useraname "posterior_Right-Hippocampus.mgz posterior_Left-Hippocampus.mgz"`

The MRI folder will contain only these hippocampus mgz files.
