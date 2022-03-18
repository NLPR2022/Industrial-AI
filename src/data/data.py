import os.path

import torch, pickle
from torch.utils.data import DataLoader, Dataset

'''

'''

category_dataset_pkl_file = '.cache/CategoryDataset.pkl'

class CategoryDataset(Dataset):
    def __init__(self, sentence_list, label_list, transform, category_manager):
        self.sentence_list = []
        self.label_list = []
        for sentence in sentence_list:
            self.sentence_list.append(transform([sentence]))
        for label in label_list:
            self.label_list.append(category_manager.code_to_one_hot('%03d' % int(label)))

    def __len__(self):
        return len(self.sentence_list)

    def __getitem__(self, idx):
        return self.sentence_list[idx], torch.FloatTensor(self.label_list[idx])

    @property
    def transform(self):
        return self.transform

    @staticmethod
    def newCategoryDataset(sentence_list, label_list, transform, category_manager):
        if os.path.exists(category_dataset_pkl_file):
            print('load saved category dataset')
            with open(category_dataset_pkl_file, 'rb') as f:
                category_dataset = pickle.load(f)
                return category_dataset
        if os.path.exists('.cache') == False:
            os.mkdir('.cache')
        category_dataset = CategoryDataset(sentence_list, label_list, transform, category_manager)
        with open(category_dataset_pkl_file, 'wb') as f:
            print('save category dataset')
            pickle.dump(category_dataset, f)
        return category_dataset

class Index:
    id_idx = 0
    digit1_idx = 1
    digit2_idx = 2
    digit3_idx = 3
    text_obj_idx = 4
    text_mthd_idx = 5
    text_deal_idx = 6

def read_txt_file(filename):
    sentence_list = []
    label_list = []

    f = open(filename)
    f.readline()
    lines = f.readlines()

    for line in lines:
        words = line.split('|')
        sentence_list.append(words[Index.text_obj_idx]+' '+words[Index.text_mthd_idx]+' '+words[Index.text_deal_idx])
        label_list.append(words[Index.digit3_idx])

    return sentence_list, label_list

def read_raw_txt_file(filename):
    f = open(filename)
    f.readline()
    lines = f.readlines()

    return lines

def get_category_dataloader(batch_size, category_manager, transform, train_portion=0.7, shuffle=True, filename='data/1. 실습용자료.txt'): #, num_workers = 2):
    sentence_list, label_list = read_txt_file(filename)
    category_dataset = CategoryDataset.newCategoryDataset(sentence_list, label_list, transform, category_manager)

    dataset_size = len(category_dataset)
    train_size = (int)(train_portion * dataset_size)
    train_set, val_set = torch.utils.data.random_split(category_dataset, [train_size, dataset_size - train_size])

    trainDataLoader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle)#, num_workers=num_workers)
    validDataLoader = DataLoader(val_set, batch_size=batch_size, shuffle=shuffle)#, num_workers=num_workers)

    return trainDataLoader, validDataLoader

def rawdata_to_sentence(rawdata):
    words = rawdata.split('|')
    return words[Index.text_obj_idx] + ' ' + words[Index.text_mthd_idx] + ' ' + words[Index.text_deal_idx]

def fill_answer(sentence, big_category, mid_category, small_category):
    sentence_list = sentence.split('|')
    sentence_list.insert(1, small_category)
    sentence_list.insert(1, mid_category)
    sentence_list.insert(1, big_category)
    sentence.join('|')
    return sentence

if __name__ == '__main__':
    tr_dataloader, val_dataloader = get_category_dataloader(10)
    print('hello')
