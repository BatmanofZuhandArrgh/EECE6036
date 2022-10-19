from tqdm import tqdm
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import random
'''
import struct

TRAIN_IMG_PATH = './Assignment2/dataset/train-images-idx3-ubyte'
TRAIN_LAB_PATH = './Assignment2/dataset/train-labels-idx1-ubyte'
TEST_IMG_PATH = './Assignment2/dataset/t10k-images-idx3-ubyte'
TEST_LAB_PATH = './Assignment2/dataset/t10k-labels-idx1-ubyte'

def get_img_data(path):
    with open(path,'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        nrows, ncols = struct.unpack(">II", f.read(8))
        data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        data = data.reshape((size, nrows, ncols))
    return data

def get_lab_data(path):
    with open(path,'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        data = data.reshape((size,)) # (Optional)
    return data    

if __name__ == "__main__":
    data = get_img_data(TRAIN_IMG_PATH)
    print(data.shape)

    labels = get_lab_data(TRAIN_LAB_PATH)
    print(labels.shape)
    
    print(labels[0])
    plt.imshow(data[0,:,:], cmap='gray')
    plt.show()

'''
IMG_TXT_PATH = "./dataset/MNISTnumImages5000_balanced.txt"
LAB_TXT_PATH = "./dataset/MNISTnumLabels5000_balanced.txt"

def dataline_to_img(dataline, label, save = False, show = False):
    cur_img = dataline.split('\n')[0].split('\t')
    cur_img = np.array([int(float(x)*255) for x in cur_img])
    cur_img = np.reshape(cur_img, (28,28))
    cur_img = cur_img.T  

    if save:
        cv2.imwrite(f'./dataset/images/{label}_sample.png', cur_img)
    if show:
        plt.imshow(cur_img)
        plt.show()
    
    return cur_img

def str_to_array(str_data):
    array = []
    for str_dataline in str_data:
        cur_array = str_dataline.split('\n')[0].split('\t')
        array.append([float(x) for x in cur_array])

    return np.array(array)

def read_img_from_txt(txt_path):
    with open(txt_path, 'r') as f:
        str_data = f.readlines()

    return str_to_array(str_data)

def read_lab_from_txt(txt_path):
    with open(txt_path, 'r') as f:
        str_label = f.readlines()
        
    return [int(x.split('\n')[0]) for x in str_label]

if __name__ == "__main__":
    os.makedirs('./dataset', exist_ok=True)
    #Processing
    with open(IMG_TXT_PATH, 'r') as f:
        str_data = f.readlines()
    
    with open(LAB_TXT_PATH, 'r') as f:
        str_label = f.readlines()

    labelline = [x.split('\n')[0] for x in str_label]

    data = []
    labels = [str(x) for x in range(0, 10)]
    label_dict = {}
    for l in labels:
        label_dict[l] = ''

    #Separate out images for 0,1,7,9 and all other number
    for i, dataline in enumerate(tqdm(str_data)):
        cur_label = labelline[i]
        label_dict[cur_label] += dataline

    for l in tqdm(labels):
        with open(f'./dataset/{l}_img.txt', 'w') as f:
            f.write(label_dict[l])

    #Get train 01 set
    with open('./dataset/0_img.txt', 'r') as f:
        data_0 = f.readlines()

    with open('./dataset/1_img.txt', 'r') as f:
        data_1 = f.readlines()
    
    #Get train set
    train_data = data_0[:400] + data_1[:400]
    train_label = ['0' for x in range(400)] + ["1" for x in range(400)]
    random.Random(4).shuffle(train_data)
    random.Random(4).shuffle(train_label)

    with open(f'./dataset/train_img.txt', 'w') as f:
        f.write(''.join(train_data))
    with open(f'./dataset/train_lab.txt', 'w') as f:
        f.write('\n'.join(train_label))
    
    #Get test set
    test_data = data_0[400:] + data_1[400:]
    test_label = ['0' for x in range(100)] + ["1" for x in range(100)]
    random.Random(4).shuffle(test_data)
    random.Random(4).shuffle(test_label)
    with open(f'./dataset/test_img.txt', 'w') as f:
        f.write(''.join(test_data))
    with open(f'./dataset/test_lab.txt', 'w') as f:
        f.write('\n'.join(test_label))

    rest_nums = [str(x) for x in range(2, 10)]
    rest_data = []
    rest_label = []
    for num in rest_nums:
        with open(f'./dataset/{num}_img.txt', 'r') as f:
            rest_data.extend(f.readlines()[:100])
            rest_label.extend([num for x in range(100)])

    with open(f'./dataset/chal_img.txt', 'w') as f:
        f.write(''.join(rest_data))
    with open(f'./dataset/chal_lab.txt', 'w') as f:
        f.write('\n'.join(rest_label))

