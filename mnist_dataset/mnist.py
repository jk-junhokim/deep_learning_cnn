# coding: utf-8
try:
    import urllib.request
except ImportError:
    raise ImportError('You should use Python 3.x')
import os.path
import gzip
import pickle
import os
import numpy as np


url_base = 'http://yann.lecun.com/exdb/mnist/'
key_file = {
    'train_img':'train-images-idx3-ubyte.gz',
    'train_label':'train-labels-idx1-ubyte.gz',
    'test_img':'t10k-images-idx3-ubyte.gz',
    'test_label':'t10k-labels-idx1-ubyte.gz'
}

# this sets the dataset directory
dataset_dir = os.path.dirname(os.path.abspath(__file__))
save_file = dataset_dir + "/mnist.pkl" # pickle file

train_num = 60000 # train data. to build accurate model
test_num = 10000 # test data. to test real world accuracy
img_dim = (1, 28, 28) # how the data info is inputed (as 3d matrices)
img_size = 784 # going to span out the 3d matrix into 1d array


def _download(file_name):
    file_path = dataset_dir + "/" + file_name
    
    if os.path.exists(file_path):
        return

    print("Downloading " + file_name + " ... ")
    urllib.request.urlretrieve(url_base + file_name, file_path)
    print("Done")
    
def download_mnist(): # download image info as originally given
    for v in key_file.values():
       _download(v)
        
def _load_label(file_name):
    file_path = dataset_dir + "/" + file_name
    
    print("Converting " + file_name + " to NumPy Array ...")
    with gzip.open(file_path, 'rb') as f:
            labels = np.frombuffer(f.read(), np.uint8, offset=8)
    print("Done")
    
    return labels

def _load_img(file_name):
    file_path = dataset_dir + "/" + file_name
    
    print("Converting " + file_name + " to NumPy Array ...")    
    with gzip.open(file_path, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
    data = data.reshape(-1, img_size)
    # used numpy reshape function to change 3d matrix
    # into 1d 1x784 array
    print("Done")
    
    return data
    
def _convert_numpy():
    dataset = {}
    # the key_file data is already saved as dictionary above
    dataset['train_img'] =  _load_img(key_file['train_img'])
    dataset['train_label'] = _load_label(key_file['train_label'])    
    dataset['test_img'] = _load_img(key_file['test_img'])
    dataset['test_label'] = _load_label(key_file['test_label'])

    # all the image data is now converted into a numpy type
    # 1x784 array per image
    
    return dataset

def init_mnist():
    download_mnist()
    dataset = _convert_numpy() # now we have what we want (1x784)
    print("Creating pickle file ...")
    with open(save_file, 'wb') as f: # save to pickle file for convenience
        pickle.dump(dataset, f, -1)
    print("Done!")

# one hot encoding is only for the label
# the numpy type image data set is a 1x784 array
# with 1~255 int type values per index (this is just how an image is formatted)
def _change_one_hot_label(X): # X is the number of answer labels
    T = np.zeros((X.size, 10)) # per answer label we created a 10 empty space array to save the one hot encoded value
    for idx, row in enumerate(T):
        row[X[idx]] = 1
        
    return T
    

def load_mnist(normalize=True, flatten=True, one_hot_label=False):
    """MNIST 데이터셋 읽기
    
    Parameters
    ----------
    normalize : re-numbers all image pixel values to 0.0~1.0
    one_hot_label : 
        If one_hot_label is True, the funtion return one-hot encoded array i.e. [0,0,1,0,0,0,0,0,0,0]
    flatten : changes image matrix into 1d 
    
    Returns
    -------
    (train image, train label), (test image, test label)
    """
    if not os.path.exists(save_file):
        init_mnist()
        
    with open(save_file, 'rb') as f:
        dataset = pickle.load(f)
    
    if normalize:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].astype(np.float32)
            dataset[key] /= 255.0
            
    if one_hot_label:
        dataset['train_label'] = _change_one_hot_label(dataset['train_label'])
        dataset['test_label'] = _change_one_hot_label(dataset['test_label'])    
    
    if not flatten:
         for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].reshape(-1, 1, 28, 28)

    return (dataset['train_img'], dataset['train_label']), (dataset['test_img'], dataset['test_label']) 


if __name__ == '__main__':
    init_mnist()