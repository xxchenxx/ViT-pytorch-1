from pdb import set_trace

read_files = ["images_variant_train.txt", "images_variant_val.txt", "images_variant_test.txt"]
write_files = ["train.txt", "val.txt", "test.txt"]

label_cnt = 0
target_dict = {}

for read_file, write_file in zip(read_files, write_files):
    with open(read_file) as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        line = line.strip("\n")
        # set_trace()
        name = line[:7]
        target = line[8:]

        if target not in target_dict:
            target_dict[target] = label_cnt
            label_cnt += 1

        new_line = "{}.jpg {}\n".format(name, target_dict[target])
        new_lines.append(new_line)

    with open(write_file, "w") as f:
        f.writelines(new_lines)
