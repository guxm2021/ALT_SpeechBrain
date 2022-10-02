import os


def prepare_text():
    # prepare text corpora for valid split
    valid_path = "data/valid.txt"
    
    # valid
    with open(os.path.join("data/dali/valid.txt"), "r") as f:
        lines = f.readlines()
        lines = [line.split('\n')[0] for line in lines if len(line) > 0]
        print(f"load from data/dali/valid.txt, read {len(lines)} lines")
    
    with open(valid_path, "w") as f:
        for line in lines:
            f.write(line)
            f.write('\n')
    
    # prepare text corpora for train splits
    train_path = "data/train.txt"

    # train
    train_lines = []
    for path in ["data/dsing/train30.txt", "data/dali/train.txt"]:
        with open(path, "r") as f:
            lines = f.readlines()
            lines = [line.split('\n')[0] for line in lines if len(line) > 0]
            train_lines.extend(lines)
            print(f"load from {path}, read {len(lines)} lines")

    with open(train_path, "w") as f:
        for line in train_lines:
            f.write(line)
            f.write('\n')


if __name__ == "__main__":
    prepare_text()
