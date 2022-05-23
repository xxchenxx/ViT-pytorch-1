import os
import re
from pdb import set_trace
from os.path import join


def main():
    data_root = os.path.expanduser("~/dataset/pets/train")

    name2dir = {}

    for dir in os.listdir(data_root):
        catergory_name = re.match(r'([a-zA-Z_]*)_[0-9]*.jpg', os.listdir(join(data_root, dir))[0])
        name2dir[catergory_name[1]] = dir

    with open("split/Pet37/Pet37_train.txt", "r") as f:
        lines = f.readlines()

    lines_new = []
    for line in lines:
        catergory_name = re.match(r'([a-zA-Z_]*)_[0-9]*.jpg [0-9]*', line)
        catergory_name = catergory_name[1]
        lines_new.append(line.replace(catergory_name, join("train", name2dir[catergory_name], catergory_name)))

    with open("split/Pet37/Pet37_train_new.txt", "w") as f:
        f.writelines(lines_new)


    with open("split/Pet37/Pet37_val.txt", "r") as f:
        lines = f.readlines()

    lines_new = []
    for line in lines:
        catergory_name = re.match(r'([a-zA-Z_]*)_[0-9]*.jpg [0-9]*', line)
        catergory_name = catergory_name[1]
        lines_new.append(line.replace(catergory_name, join("train", name2dir[catergory_name], catergory_name)))

    with open("split/Pet37/Pet37_val_new.txt", "w") as f:
        f.writelines(lines_new)


    with open("split/Pet37/Pet37_test.txt", "r") as f:
        lines = f.readlines()

    lines_new = []
    for line in lines:
        catergory_name = re.match(r'([a-zA-Z_]*)_[0-9]*.jpg [0-9]*', line)
        catergory_name = catergory_name[1]
        lines_new.append(line.replace(catergory_name, join("val", name2dir[catergory_name], catergory_name)))

    with open("split/Pet37/Pet37_test_new.txt", "w") as f:
        f.writelines(lines_new)


if __name__ == "__main__":
    main()
