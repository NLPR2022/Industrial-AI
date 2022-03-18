import os

from matplotlib import pyplot as plt
import csv

class ValueHistory():
    def __init__(self):
        self.value_dictionary = {}

    def add_history(self, name, value):
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
        self._draw_list(self.value_dictionary[name])

    def _draw_list(self, loss_list, fig_size = (10, 5), title ="",
                   label = "", xlabel ="", ylabel = ""):
        plt.figure(figsize=fig_size)
        plt.title = title
        plt.plot(loss_list, label=label)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.show()

    def save_csv_all_history(self, filename, path = ""):
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