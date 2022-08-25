
import os

for file in os.listdir('.'):
    new_file = file.replace('lr1', 'lra')
    new_file = new_file.replace('lr2', 'lrf')
    new_file = new_file.split('_')
    new_file = '_'.join(new_file[:3]  + new_file[4:])
    os.system(f'mv {file} {new_file}')

