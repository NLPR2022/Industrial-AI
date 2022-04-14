import os.path

import gluonnlp as nlp
from kobert import get_tokenizer
from kobert import get_pytorch_kobert_model

import torch, pickle
from torch.utils.data import DataLoader, Dataset

from src.config import *
from src.preprocess.category_manager import *

'''
데이터는 아래와 같이 구성되었다.
1. 대분류
2. 중분류
3. 소분류
4. 사업 대상, 단어1로 부르겠다.
5. 사업 방법, 단어2
6. 사업 취급 품목, 단어3
'''

class CategoryDataset(Dataset):
    def __init__(self, sentence_list, label_list, transform, category_manager):
        ''' Category Dataset

        :param sentence_list: 문장을 담은 list
        :param label_list: 문장의 소분류 label을 담은 list
        :param transform: tokenize하고 embedding하는 transform
        :param category_manager: 카테고리의 정보를 담은 category_manager
        '''
        self.sentence_list = []
        self.label_list = []
        for sentence in sentence_list:
            self.sentence_list.append(transform([sentence]))
        for label in label_list:
            if label != '':
                self.label_list.append(category_manager.code_to_one_hot('%03d' % int(label)))
            else:
                self.label_list.append([0])

    def __len__(self):
        return len(self.sentence_list)

    def __getitem__(self, idx):
        return self.sentence_list[idx], torch.FloatTensor(self.label_list[idx])

    @staticmethod
    def newCategoryDataset(sentence_list, label_list, transform, category_manager, pkl_tag):
        ''' 새로운 CategoryDataset을 만드는 함수

        :param sentence_list: 전처리를 거친 문장의 리스트
        :param label_list: 라벨의 리스트
        :param transform: tokenize -> embedding을 하는 transform
        :param category_manager: 카테고리 정보를 담는 category manager
        :param pkl_tag: pkl 파일 뒤에 붙일 태그, 어떤 데이터인지를 의미한다.
        :return: category dataset
        '''
        category_dataset_pkl_file = f'.cache/CategoryDataset-{pkl_tag}.pkl'

        category_manager = CategoryManager.new_category_manager(category_file)
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

class INDEX:
    '''
    한국표준산업분류 xlsx 파일에 있는 column idx
    '''
    ID_IDX = 0 # data id의 column
    BIG_IDX = 1 # 대분류의 column
    MID_IDX = 2 # 중분류의 column
    SMALL_IDX = 3 # 소분류의 column
    TEXT_OBJ_IDX = 4 # 사업 대상, 단어1
    TEXT_MTHD_IDX = 5 # 사업 방법, 단어2
    TEXT_DEAL_IDX = 6 # 사업 취급 품목, 단어3

def rawdata_to_sentence(rawdata):
    ''' raw 문장으로 전처리 한 문장 만드는 것

    :param rawdata: txt파일에서 읽어온 그대로의 문장
    :return: 단어1 + 단어2 + 단어3로 만든 문장
    '''
    words = rawdata.split('|')
    return words[INDEX.TEXT_OBJ_IDX] + ' ' + words[INDEX.TEXT_MTHD_IDX] + ' ' + words[INDEX.TEXT_DEAL_IDX], words[INDEX.SMALL_IDX]

def read_txt_file(filename):
    ''' txt 파일을 읽어서 단어1 + 단어2 + 단어3로 문장으로 만들고, 소분류를 label로 만드는 함수

    :param filename: txt 파일
    :return: 문장 리스트, 라벨 리스트
    '''
    sentence_list = []
    label_list = []
    lines = read_raw_txt_file(filename)

    for line in lines:
        s, l = rawdata_to_sentence(line)
        sentence_list.append(s)
        label_list.append(l)

    return sentence_list, label_list

def read_raw_txt_file(filename):
    ''' txt 파일의 각 줄의 문장을 list로 표현한 것

    :param filename: txt 파일
    :return: 모든 줄의 raw 문장 list
    '''
    f = open(filename)
    f.readline()
    lines = f.readlines()

    return lines

def get_category_dataloader(batch_size, category_manager, transform, train_portion, filename, pkl_tag, shuffle=True):
    ''' category dataloader 얻는 함수

    :param batch_size: batch size
    :param category_manager: category 정보를 담은 category_manager
    :param transform: tokenizer -> embedding 하는 transform
    :param train_portion: trainset의 비율 (0~1)
    :param filename: file이름
    :param pkl_tag: pkl 파일 뒤에 붙는 tag
    :param shuffle: shuffle 유무
    :return: trainDataLoader, validDataLoader
    '''
    sentence_list, label_list = read_txt_file(filename)
    category_dataset = CategoryDataset.newCategoryDataset(sentence_list, label_list, transform, category_manager, pkl_tag)

    dataset_size = len(category_dataset)
    train_size = (int)(train_portion * dataset_size)
    train_set, val_set = torch.utils.data.random_split(category_dataset, [train_size, dataset_size - train_size])

    trainDataLoader = DataLoader(category_dataset, batch_size=batch_size, shuffle=shuffle) if len(train_set) != 0 else None
    validDataLoader = DataLoader(category_dataset, batch_size=batch_size, shuffle=shuffle) if len(val_set) != 0 else None

    return trainDataLoader, validDataLoader

def fill_answer_sentence(sentence, big_category, mid_category, small_category):
    ''' 빈칸에 추론한 답을 채워서 문장을 만드는 함수

    :param sentence: 빈칸으로 된 문장
    :param big_category: 추론한 대분류
    :param mid_category: 추론한 중분류
    :param small_category: 추론한 소분류
    :return: 추론한 값들로 다시 구성한 문장
    '''
    sentence_list = sentence.split('|')
    sentence_list.insert(1, small_category)
    sentence_list.insert(1, mid_category)
    sentence_list.insert(1, big_category)
    sentence.join('|')
    return sentence


def get_bert_tokenizer(max_len):
    ''' pretrained된 bert model, tokenize -> embedding 하는 transform

    :param max_len: (int) 문장의 최대 길이
    :return: pretrained된 bert model, transform
    '''
    bertmodel, vocab = get_pytorch_kobert_model()
    tokenizer = get_tokenizer()
    bert_tokenizer = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)
    transform = nlp.data.BERTSentenceTransform(bert_tokenizer, max_seq_length=max_len, pad=True, pair=False)
    return bertmodel, transform

if __name__ == '__main__':
    bert, transform = get_bert_tokenizer(max_len)
    category_manager = CategoryManager.new_category_manager(category_file)
    train_dataloader, valid_dataloader = get_category_dataloader(32, category_manager, transform,train_portion=train_portion, shuffle=True,filename='preprocess/2. 모델개발용자료.txt', pkl_tag='unlabeled')
