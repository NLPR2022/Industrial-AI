import os.path
import pickle

import pandas

class CATEGORY_COLUMN:
    BIG_CODE = 0
    BIG_NAME = 1
    MID_CODE = 2
    MID_NAME = 3
    SMALL_CODE = 4
    SMALL_NAME = 5
    SUB_CODE = 6
    SUB_NAME = 7
    SUBSUB_CODE = 8
    SUBSUB_NAME = 9

class CategoryManager():
    def __init__(self, category_manager_file):
        self.category_name = {} # key : code, value : name
        self.small_to_big_dictionary = {}
        self.small_to_mid_dictionary = {}
        self.mid_to_big_dictionary = {}
        self.num_mid_in_big = {}
        self.num_small_in_mid = {}
        self.build_category(category_manager_file)

    def build_category(self, category_manager_file):
        excel_data = pandas.read_excel(category_manager_file)

        big_category = None
        mid_category = None
        small_category = None
        sub_category = None
        subsub_category = None

        big_category_name = None
        mid_category_name = None
        small_category_name = None
        sub_category_name = None
        subsub_category_name = None

        for row_idx, row_data in enumerate(excel_data.iterrows()):
            if row_idx < 2:
                continue
            if str(row_data[1][CATEGORY_COLUMN.BIG_CODE]) != 'nan':
                big_category = str(row_data[1][CATEGORY_COLUMN.BIG_CODE])
                big_category_name = str(row_data[1][CATEGORY_COLUMN.BIG_NAME])
                self.category_name[big_category] = big_category_name
                self.num_mid_in_big[big_category] = 0

            if str(row_data[1][CATEGORY_COLUMN.MID_CODE]) != 'nan':
                mid_category = str(row_data[1][CATEGORY_COLUMN.MID_CODE])
                mid_category_name = str(row_data[1][CATEGORY_COLUMN.MID_NAME])
                self.category_name[mid_category] = mid_category_name
                self.num_mid_in_big[big_category] = self.num_mid_in_big[big_category] + 1
                self.num_small_in_mid[mid_category] = 0

            if str(row_data[1][CATEGORY_COLUMN.SMALL_CODE]) != 'nan':
                small_category = str(row_data[1][CATEGORY_COLUMN.SMALL_CODE])
                small_category_name = str(row_data[1][CATEGORY_COLUMN.SMALL_NAME])
                self.category_name[small_category] = small_category_name
                self.num_small_in_mid[mid_category] = self.num_small_in_mid[mid_category] + 1

            if str(row_data[1][CATEGORY_COLUMN.SUB_CODE]) != 'nan':
                sub_category = str(row_data[1][CATEGORY_COLUMN.SUB_CODE])
                sub_category_name = str(row_data[1][CATEGORY_COLUMN.SUB_NAME])
                self.category_name[sub_category] = sub_category_name

            if str(row_data[1][CATEGORY_COLUMN.SUBSUB_CODE]) != 'nan':
                subsub_category = str(row_data[1][CATEGORY_COLUMN.SUBSUB_CODE])
                subsub_category_name = str(row_data[1][CATEGORY_COLUMN.SUBSUB_NAME])
                self.category_name[subsub_category] = subsub_category_name

            self.small_to_big_dictionary[small_category] = big_category
            self.small_to_mid_dictionary[small_category] = mid_category
            self.mid_to_big_dictionary[mid_category] = big_category

        self.save_category(category_manager_file)

    @staticmethod
    def new_category_manager(category_manager_file, refresh=False):
        if refresh == True:
            return CategoryManager()
        if os.path.exists(category_manager_file) == True:
            with open(category_manager_file, 'rb') as f:
                unpickler = pickle.Unpickler(f)
                return unpickler.load()
        else:
            return CategoryManager()

    def save_category(self, category_manager_file):
        with open(category_manager_file, 'wb') as f:
            pickle.dump(self, f)

    def get_name_by_code(self, code):
        return self.category_name[code]

    def get_mid_category(self, small_category):
        return self.small_to_mid_dictionary[small_category]

    def get_big_category(self, category_code):
        if category_code in self.mid_to_big_dictionary:
            return self.mid_to_big_dictionary[category_code]
        if category_code in self.small_to_big_dictionary:
            return self.small_to_big_dictionary[category_code]
        raise Exception(f"code {category_code} is not exist.")

if __name__ == '__main__':
    category_manager = CategoryManager.new_category_manager('etc/category.pkl')