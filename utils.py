import numpy as np
import pickle
from torch.utils.data import Dataset
import json



#####################CIFAR utils
# def pickle_to_dict(file):
#     with open(file, 'rb') as fo:
#         unpickled = pickle.load(fo, encoding='bytes')
#     return unpickled



# def reconstruct_image(image_data):
#     image = []
#     for i in range(0,image_data.shape[0],1024):
#         channel = image_data[i:i+1024]
#         channel = channel.reshape(32,32)
#         image.append(channel.astype(np.float32))
#     return image



# class img_dataset(Dataset):
#     def __init__(self, img, label):
#         self.x = img
#         self.y = label
#     def __len__(self):
#         return len(self.x)
#     def __getitem__(self,index):
#         return self.x[index], self.y[index]
###########################################


def load_json(path):
    with open(path) as file:
        loaded = json.load(file)
    return loaded

class MCCD():
    def __init__(self):
        pass
    
    def get_item(self,index):
        return self.dataset[index]
    def get_dataset(self,subset):
        self.dataset = load_json(f"D:\\Datasets\\BCCD\\dataset2-master\\dataset2-master\\images\\TRAIN\\{subset}.json")
        return self.dataset
        

class CVPR():
    def __init__(self):
        pass
    
    def get_item(self,index):
        return self.dataset[index]
    def get_dataset(self,subset):
        self.dataset = load_json(f"D:\\Datasets\\CVPR\\indoorCVPR_09\\Images\\{subset}.json")
        return self.dataset
    
class AHE():
    def __init__(self):
        pass
    
    def get_item(self,index):
        return self.dataset[index]
    def get_dataset(self,subset):
        self.dataset = load_json(f"D:\\Datasets\\AHE\\{subset}.json")
        return self.dataset


def collate(example):
    # transforms a list of dictionaries to a dictionary of lists, sharing the same keys
    temp_dict = dict()
    for item in example:
        for key in item:
            if key not in temp_dict.keys():
                temp_dict[key] = []
            temp_dict[key].append(item[key])
        
    example = temp_dict
    return example