import os.path

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
    def __init__(self, category_file):
        self.category_name_dict = {} # key : code, value : name
        self.small_to_big_dict = {}
        self.small_to_mid_dict = {}
        self.mid_to_big_dict = {}
        self.num_mid_per_big_dict = {}
        self.num_small_per_mid_dict = {}
        self.code_to_id_dict = {}
        self.id_to_code_dict = {}
        self.build_category(category_file)

    def build_category(self, category_file):
        excel_data = pandas.read_excel(category_file)

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

        small_category_count = 0

        for row_idx, row_data in enumerate(excel_data.iterrows()):
            if row_idx < 2:
                continue
            if str(row_data[1][CATEGORY_COLUMN.BIG_CODE]) != 'nan':
                big_category = str(row_data[1][CATEGORY_COLUMN.BIG_CODE])
                big_category_name = str(row_data[1][CATEGORY_COLUMN.BIG_NAME])
                self.category_name_dict[big_category] = big_category_name
                self.num_mid_per_big_dict[big_category] = 0

            if str(row_data[1][CATEGORY_COLUMN.MID_CODE]) != 'nan':
                mid_category = str(row_data[1][CATEGORY_COLUMN.MID_CODE])
                mid_category_name = str(row_data[1][CATEGORY_COLUMN.MID_NAME])
                self.category_name_dict[mid_category] = mid_category_name
                self.num_mid_per_big_dict[big_category] = self.num_mid_per_big_dict[big_category] + 1
                self.num_small_per_mid_dict[mid_category] = 0

            if str(row_data[1][CATEGORY_COLUMN.SMALL_CODE]) != 'nan':
                small_category = str(row_data[1][CATEGORY_COLUMN.SMALL_CODE])
                small_category_name = str(row_data[1][CATEGORY_COLUMN.SMALL_NAME])
                self.category_name_dict[small_category] = small_category_name
                self.num_small_per_mid_dict[mid_category] = self.num_small_per_mid_dict[mid_category] + 1
                self.code_to_id_dict[small_category] = small_category_count
                self.id_to_code_dict[small_category_count] = small_category
                small_category_count = small_category_count + 1

            if str(row_data[1][CATEGORY_COLUMN.SUB_CODE]) != 'nan':
                sub_category = str(row_data[1][CATEGORY_COLUMN.SUB_CODE])
                sub_category_name = str(row_data[1][CATEGORY_COLUMN.SUB_NAME])
                self.category_name_dict[sub_category] = sub_category_name

            if str(row_data[1][CATEGORY_COLUMN.SUBSUB_CODE]) != 'nan':
                subsub_category = str(row_data[1][CATEGORY_COLUMN.SUBSUB_CODE])
                subsub_category_name = str(row_data[1][CATEGORY_COLUMN.SUBSUB_NAME])
                self.category_name_dict[subsub_category] = subsub_category_name

            self.small_to_big_dict[small_category] = big_category
            self.small_to_mid_dict[small_category] = mid_category
            self.mid_to_big_dict[mid_category] = big_category

    @staticmethod
    def new_category_manager(category_file):
        return CategoryManager(category_file)

    def get_name_by_code(self, code):
        return self.category_name_dict[code]

    def get_mid_category(self, small_category):
        return self.small_to_mid_dict[small_category]

    def get_big_category(self, category_code):
        if category_code in self.mid_to_big_dict:
            return self.mid_to_big_dict[category_code]
        if category_code in self.small_to_big_dict:
            return self.small_to_big_dict[category_code]
        raise Exception(f"code {category_code} is not exist.")

    @property
    def small_category_num(self):
        return len(self.small_to_big_dict)

    @property
    def mid_category_num(self):
        return len(self.mid_to_big_dict)

    @property
    def big_category_num(self):
        return len(self.num_mid_per_big_dict)

    def id_to_code(self, id):
        return self.id_to_code_dict[id]

    def code_to_code_id(self, code):
        return self.code_to_id_dict[code]

    def code_to_one_hot(self,code):
        code_id = self.code_to_code_id(code)
        one_hot_vector = [0 for _ in range(self.small_category_num)]
        one_hot_vector[code_id] = 1
        return one_hot_vector

if __name__ == '__main__':
    category_manager = CategoryManager.new_category_manager('data/한국표준산업분류(10차)_국문.xlsx')

    print('hi')