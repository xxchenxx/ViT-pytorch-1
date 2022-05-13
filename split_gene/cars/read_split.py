import os
from os.path import join
import numpy as np


def read_stanford_car(root="/home/grads/j/jiangziyu/dataset/stanford_car", subs=["train","val"]):
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


def read_stanford_car_train_val(root="/home/grads/j/jiangziyu/dataset/stanford_car"):
  train, val = read_stanford_car(root=root)

  for sub, lines in zip(["train", "val"], [train, val]):
    with open("split/stanford_car/{}.txt".format(sub), "w") as f:
      f.writelines(lines)


if __name__ == "__main__":
  read_stanford_car_train_val()

