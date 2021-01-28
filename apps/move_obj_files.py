import os
import shutil
path = '/media/ubuntu/SAMSUNG/watertight_obj'
new_path = '/home/ubuntu/Desktop/PIFu/data/GEO/OBJ'
file_list = os.listdir(path)
for i, file_name in enumerate(file_list):
    os.remove(new_path + f'/{file_name[:-9]}/{file_name}')
    shutil.copyfile(path + f'/{file_name}', new_path + f'/{file_name[:-9]}/{file_name}')