from pdb import set_trace

read_files = ["train_100.txt", "test.txt"]

label_cnt = 0
target_dicts = [{}, {}]

for read_file, target_dict in zip(read_files, target_dicts):
    with open(read_file) as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        line = line.strip("\n")
        # set_trace()
        label = line.split()[-1]
        label_name = line[:-(len(label) + 2)].split("/")[1]

        if label_name not in target_dict:
            target_dict[label_name] = label
        else:
            assert target_dict[label_name] == label

set_trace()