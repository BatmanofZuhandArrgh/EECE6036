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
    
    test_imgs = []
    test_labels = []
    train_imgs = []
    train_labels = []

    for num in range(0, 10):
        data_path = f'./dataset/{str(num)}_img.txt'
        with open(data_path, 'r') as f:
            str_data = f.readlines()

        train_imgs.extend(str_data[:400])
        train_labels.extend([str(num) for x in range(400)])

        test_imgs.extend(str_data[400:])
        test_labels.extend([str(num) for x in range(100)])

    # with open(f'./dataset/train_4000imgs.txt', 'w') as f:
    #     f.write(''.join(train_imgs))
    # with open(f'./dataset/train_4000labs.txt', 'w') as f:
    #     f.write('\n'.join(train_labels))

    # with open(f'./dataset/test_1000imgs.txt', 'w') as f:
    #     f.write(''.join(test_imgs))
    # with open(f'./dataset/test_1000labs.txt', 'w') as f:
    #     f.write('\n'.join(test_labels))

    testing_sample = read_img_from_txt('./dataset/test_1000imgs.txt')
    print(testing_sample.shape)