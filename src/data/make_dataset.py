# -*- coding: utf-8 -*-
import glob
import logging
import os
import random
import re
import tarfile


def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    command = "obabel -ict *.ct -osdf -m"  # command to trasform the .ct file int .sdf file
    folders = []

    # static folder name for data
    input_folder = os.getcwd().replace("src/data", "data/raw/")
    output_folder = os.getcwd().replace("src/data", "data/interim/")

    # iter through zip file, unzipped and trasformed
    for file in glob.glob(input_folder + "*.tar.gz"):
        print(file + "  Decompressed")
        folders.append(file.replace("raw", "interim").replace(".tar.gz", "/"))
        with tarfile.open(file) as tar:
            tar.extractall(path=output_folder)
    for folder in folders:
        os.chdir(folder)
        os.system(command)

    # alkane dataset haven't been yet splitted into 10-CV so we do it

    filepath_to_10_cv = 'dataset_boiling_point_names.txt'

    mol_dict = {}
    with open(filepath_to_10_cv) as fp:
        line = fp.readline()
        line = re.findall("[,0-9a-zA-Z()_-]*.ct [-0-9]*[.0-9]*", line)
        line = line[0].split(' ')
        mol_dict[line[0]] = line[1]
        while line:
            line = fp.readline()
            if line:
                line = re.findall("[,0-9a-zA-Z()_-]*.ct [-0-9]*[.0-9]*", line)
                line = line[0].split(' ')
                mol_dict[line[0]] = line[1]

    keys = list(mol_dict.keys())
    random.shuffle(keys)
    shuffled_mol_dict = {}
    for key in keys:
        shuffled_mol_dict[key] = mol_dict[key]

    from collections import deque
    mol_file_name = deque(shuffled_mol_dict.keys())
    for i in range(10):
        mol_file_name = list(mol_file_name)
        f = open('trainset_' + str(i) + '.ds', 'w')
        for nameF in mol_file_name[:135]:
            # read a single line
            f.write(nameF + ' ' + shuffled_mol_dict[nameF] + '\n')
        # close the pointer to that file
        f.close()
        f = open('testset_' + str(i) + '.ds', 'w')
        for nameF in mol_file_name[135:]:
            # read a single line
            f.write(nameF + ' ' + shuffled_mol_dict[nameF] + '\n')
        # close the pointer to that file
        f.close()
        mol_file_name = deque(mol_file_name)
        mol_file_name.rotate(15)

        # %%


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
