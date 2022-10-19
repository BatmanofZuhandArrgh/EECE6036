from parse_data_file import dataline_to_img

with open('./dataset/train_img.txt', 'r') as f:
    str_data = f.readlines()

with open('./dataset/train_lab.txt', 'r') as f:
    str_label = f.readlines()

idx = 4
print(str_label[idx])
dataline_to_img(str_data[idx], str_label[idx], show = True)