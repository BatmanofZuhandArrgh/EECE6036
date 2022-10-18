from tqdm import tqdm
import numpy as np
import cv2
import matplotlib.pyplot as plt
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

if __name__ == "__main__":
    #Processing
    with open(IMG_TXT_PATH, 'r') as f:
        str_data = f.readlines()
    
    with open(LAB_TXT_PATH, 'r') as f:
        str_label = f.readlines()

    labelline = [x.split('\n')[0] for x in str_label]

    data = []
    for i, dataline in tqdm(enumerate(str_data)):
        cur_img = dataline.split('\n')[0].split('\t')
        cur_img = np.array([int(float(x)*255) for x in cur_img])
        cur_img = np.reshape(cur_img, (28,28))
        cur_img = cur_img.T
        
        cv2.imwrite(f'./dataset/images/{i}_{labelline[i]}.png', cur_img)

