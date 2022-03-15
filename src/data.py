import os.path

import torch
from torch.utils.data import DataLoader, Dataset
import gluonnlp as nlp

class CategoryDataset(Dataset):
    def __init__(self, id_list, digit1_list, digit2_list, digit3_list, text_obj_list, text_mthd_list, text_deal_list, bert_tokenizer):
        self.id_list = id_list
        self.digit1_list = digit1_list
        self.digit2_list = digit2_list
        self.digit3_list = digit3_list
        # self.word_bag = bert_tokenizer(text_obj_list[0])
        self.text_obj_list = text_obj_list
        self.text_mthd_list = text_mthd_list
        self.text_deal_list = text_deal_list
        # self.bert_tokenizer = bert_tokenizer

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, idx):
        id = self.id_list[idx]
        digit1 = self.digit1_list[idx]
        digit2 = self.digit2_list[idx]
        digit3 = self.digit3_list[idx]
        text_obj = self.text_obj_list[idx]
        text_mthd = self.text_mthd_list[idx]
        text_deal = self.text_deal_list[idx]

        return id, digit1, digit2, digit3, text_obj, text_mthd, text_deal

class Index:
    id_idx = 0
    digit1_idx = 1
    digit2_idx = 2
    digit3_idx = 3
    text_obj_idx = 4
    text_mthd_idx = 5
    text_deal_idx = 6

def read_txt_file(filename):
    print(os.path.curdir)
    # id_list = []
    # digit1_list = []
    # digit2_list = []
    # digit3_list = []
    sentence_list = []
    label_list = []
    # text_obj_list = []
    # text_mthd_list = []
    # text_deal_list = []
    f = open(filename)
    print(f.readline())
    lines = f.readlines()
    for line in lines:
        words = line.split('|')
        # id_list.append(words[Index.id_idx])
        # digit1_list.append(words[Index.digit1_idx])
        # digit2_list.append(words[Index.digit2_idx])
        # digit3_list.append(words[Index.digit3_idx])
        sentence_list.append(words[Index.text_obj_idx]+' '+words[Index.text_mthd_idx]+' '+words[Index.text_deal_idx])
        label_list.append(words[Index.digit3_idx])
        # text_obj_list.append(words[Index.text_obj_idx])
        # text_mthd_list.append(words[Index.text_mthd_idx])
        # text_deal_list.append(words[Index.text_deal_idx])
    return sentence_list, label_list


def get_category_dataloader(batch_size, train_portion=0.7, shuffle=True, bert_tokenizer=None, filename='data/1. 실습용자료.txt'):
    id_list, digit1_list, digit2_list, digit3_list, text_obj_list, text_mthd_list, text_deal_list = read_txt_file(filename)
    category_dataset = CategoryDataset(id_list, digit1_list, digit2_list, digit3_list, text_obj_list, text_mthd_list, text_deal_list, bert_tokenizer)
    dataset_size = len(category_dataset)
    train_size = (int)(train_portion * dataset_size)
    train_set, val_set = torch.utils.data.random_split(category_dataset, [train_size, dataset_size - train_size])
    trainDataLoader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle)
    validDataLoader = DataLoader(val_set, batch_size=batch_size, shuffle=shuffle)
    return trainDataLoader, validDataLoader


if __name__ == '__main__':
    tr_dataloader, val_dataloader = get_category_dataloader(10)
    print('hello')
