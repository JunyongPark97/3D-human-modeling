import argparse
import glob
import os

import natsort as natsort


def changeFolderName(dir_path, dir_name):
    file_list_raw = os.listdir(dir_path)
    file_list = natsort.natsorted(file_list_raw, reverse=False)
    if dir_path[-1] == "/":
        dir_path = dir_path[:-1]

    for i in range(len(file_list)):
        first = (i+1) % 10
        second = int(((i+1) / 10) % 10)
        third = int((((i+1)/10)/10)%10)
        os.rename(dir_path + f'/{os.path.splitext(os.path.basename(file_list[i]))[0]}', dir_path + f'/{dir_name}_{third}{second}{first}_OBJ')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--dir_path', type=str, default='/home/ubuntu/Desktop/1/')
    parser.add_argument('-n', '--dir_name', type=str, default='test_posed')
    args = parser.parse_args()

    changeFolderName(args.dir_path, args.dir_name)