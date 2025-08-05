def read_case_list(txt_path):
    with open(txt_path, 'r') as f:
        return [line.strip() for line in f.readlines()]
