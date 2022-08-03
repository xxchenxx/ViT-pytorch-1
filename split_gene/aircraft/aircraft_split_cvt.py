import os
from os.path import join
import numpy as np


def read(root, subs):
  dirs = os.listdir(join(root, subs[0]))
  dirs = sorted(dirs)
  dir2class = {d: i for i, d in enumerate(dirs)}

  lines_all = []
  for sub in subs:
    dirs = os.listdir(join(root, sub))
    dirs = sorted(dirs)
    lines = []
    for d in dirs:
      images = os.listdir(join(root, sub, d))
      for image in images:
        lines.append("{} {}\n".format(join(sub, d, image), dir2class[d]))
    lines_all.append(lines)

  return lines_all


def read_aircraft_train_val(root="/home/ziyu/dataset/food101"):
  train, val = read(root=root, subs=["train", "val"])

  dir = "gene_result/food101"
  os.system("mkdir -p {}".format(dir))

  for sub, lines in zip(["train", "val"], [train, val]):
    with open("{}/{}.txt".format(dir, sub), "w") as f:
      f.writelines(lines)


def read_food101_train_val(root="/home/ziyu/dataset/food101"):
  train, val = read(root=root, subs=["train", "val"])

  dir = "gene_result/food101"
  os.system("mkdir -p {}".format(dir))

  for sub, lines in zip(["train", "val"], [train, val]):
    with open("{}/{}.txt".format(dir, sub), "w") as f:
      f.writelines(lines)


def read_cub200_train_val(root="/home/ziyu/dataset/cub200"):
  train, val = read(root=root, subs=["train", "val"])

  dir = "gene_result/cub200"
  os.system("mkdir -p {}".format(dir))

  for sub, lines in zip(["train", "val"], [train, val]):
    with open("{}/{}.txt".format(dir, sub), "w") as f:
      f.writelines(lines)


def read_cub200_train_val(root="/home/ziyu/dataset/cub200"):
  train, val = read(root=root, subs=["train", "val"])

  dir = "gene_result/cub200"
  os.system("mkdir -p {}".format(dir))

  for sub, lines in zip(["train", "val"], [train, val]):
    with open("{}/{}.txt".format(dir, sub), "w") as f:
      f.writelines(lines)


if __name__ == "__main__":
  # read_aircraft_train_val()
  # read_food101_train_val()
  read_cub200_train_val()