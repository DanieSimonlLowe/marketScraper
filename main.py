import kagglehub
import tensorflow
# train encoder to do input dimension. (remove last layer and replace with 32 dim)
# two basic matrix mult on enbedding (as window so large don't need to do attention on this or anything)
# 384 * 40 + 40 * 32
# two graph convolutions on max of above
# 32 * 32 * 2 * 2
# Group this once (use graph theory create groups)
# 32 * 32 * 2 * 2
# Degroup this once
# 32 * 32 * 2 * 2
# then final layer down to good size on max of the above
# 32 * 16 + 16 * 1

path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")

print("Path to dataset files:", path)

from numpy import genfromtxt


def read_to_numpy_array(file_path):
    '''
    Reads a file and transforms it into a numpy array.
    :param file_path: The path of the file
    :return: A numpy array
    '''
    return genfromtxt(file_path, delimiter=',')