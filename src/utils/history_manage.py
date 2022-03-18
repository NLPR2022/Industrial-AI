import os

from matplotlib import pyplot as plt
import csv

class ValueHistory():
    def __init__(self):
        '''
            acc나 loss를 csv나 graph로 기록하기 위해 만든 클래스
            여러 리스트를 dictionary 형태로 저장해두었다가 graph나 csv로 표현하는 클래스
        '''
        self.value_dictionary = {}

    def add_history(self, name, value):
        ''' name을 이름으로 갖는 리스트에 value를 추가

        :param name: (string) 리스트의 이름
        :param value: (ang) 추가할 값
        :return: None
        '''
        if name in self.value_dictionary:
            if type(value) is int or type(value) is float:
                self.value_dictionary[name].append(value)
            else:
                self.value_dictionary[name] = self.value_dictionary[name] + value

        else:
            if type(value) is int or type(value) is float:
                self.value_dictionary[name] = [value]
            else:
                self.value_dictionary[name] = value

    def draw_list_by_name(self, name):
        ''' 리스트 이름을 입력하면 matplot으로 graph를 그린다.

        :param name: (string) 리스트의 이름
        :return: None
        '''
        self._draw_list(self.value_dictionary[name])

    def _draw_list(self, loss_list, fig_size = (10, 5), title ="",
                   label = "", xlabel ="", ylabel = ""):
        ''' list를 넣으면 graph를 그리는 함수

        :param loss_list: (list) graph로 그릴 list
        :param fig_size: (tuple) graph의 크기
        :param title: (string) graph의 title
        :param label: (string) graph의 a
        :param xlabel: (string) graph의 x축 title
        :param ylabel: (string) graph의 y축 title
        :return:
        '''
        plt.figure(figsize=fig_size)
        plt.title = title
        plt.plot(loss_list, label=label)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.show()

    def save_csv_all_history(self, filename, path = ""):
        '''history에 저장된 data를 csv로 저장하는 함수

        :param filename: 파일의 이름
        :param path: 저장할 path
        :return: None
        '''
        try:
            if path == "":
                path = os.path.curdir
            else:
                if os.path.exists(path) == False:
                    os.mkdir(path)
            if ".csv" not in filename:
                filename = filename + ".csv"
            filepath = path + "\\" + filename
            longest_len = 0
            for k, v in self.value_dictionary.items():
                longest_len = len(v) if len(v) > longest_len else longest_len
            with open(filepath, 'w') as csvfile:
                w = csv.writer(csvfile, lineterminator='\n')
                w.writerow(self.value_dictionary.keys())
                for i in range(longest_len):
                    save_index_list = []
                    for k, v in self.value_dictionary.items():
                        if len(v) <= i:
                            save_index_list.append('-')
                        else:
                            save_index_list.append(v[i])
                    w.writerow(save_index_list)
        except:
            print('fail to save history')

    def load_csv_history(self, filepath):
        raise NotImplementedError

def test():
    history= ValueHistory()
    history.add_history("a", 1)
    history.add_history("a", [1,2,3])
    history.draw_list_by_name("a")
    history.add_history("b", 1)
    history.add_history("b", [1, 2, 3])
    history.draw_list_by_name("b")
    history.save_csv_all_history("test")
# test()