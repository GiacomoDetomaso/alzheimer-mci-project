import nibabel as ni
from glob import glob 

import os

x_list = []
y_list = []
z_list = []

dirs = glob(os.path.join('data', 'freesurfers', '*', '*', '*Hippocampus.mgz'))

for d in dirs:
    print(d)

    file = ni.load(d)

    x_list.append(file.shape[0])
    y_list.append(file.shape[1])
    z_list.append(file.shape[2])

print(f'x min: {min(x_list)}, x max: {max(x_list)}')
print(f'y min: {min(y_list)}, y max: {max(y_list)}')
print(f'z min: {min(z_list)}, z_max: {max(z_list)}')