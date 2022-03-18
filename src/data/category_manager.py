import os.path

import pandas

'''
category_manager

산업분류의 구성:
    산업분류는 code(0111)와 name(곡물 및 기타 식량작물 재배업)으로 구성되어 있다.
    프로그램에서 사용하는 code는 소분류의 code이다.
    소분류의 코드는 continuous하지 않다. 따라서 232개의 소분류 코드를 0~231로 인코딩 해줘야한다.
    
category manager의 역할: 
    한국표준산업분류 엑셀을 읽어서 정리해놓음.
    다음과 같은 정보를 저장한다.
    - 카테고리 이름과 코드
    - 각 하위 분류가 속하는 상위 분류
    - 
'''
class CATEGORY_COLUMN:
    '''
        한국표준산업분류(10차)_국문.xlsx'의 컬럼명
    '''
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
        ''' '한국표준산업분류(10차)_국문.xlsx'를 읽어서 해당 파일의 정보를 정리해놓은 클래스

        :param category_file: (string) '한국표준산업분류(10차)_국문.xlsx'의 경로
        '''
        self.category_name_dict = {} # 소분류, 중분류, 대분류에 있는 모든 code와 name 저장, key : code, value : name
        self.small_to_big_dict = {} # 소분류가 속하는 대분류를 알려줌
        self.small_to_mid_dict = {} # 소분류가 속하는 중분류를 알려줌
        self.mid_to_big_dict = {} # 중분류가 속하는 대분류를 알려줌
        self.num_mid_per_big_dict = {} # 대분류에 있는 중분류의 개수를 알려줌
        self.num_small_per_mid_dict = {} # 중분류에 있는 소분류의 개수를 알려줌
        self.code_to_id_dict = {} # code를 id로 바꿔줌
        self.id_to_code_dict = {} # id를 code로 바꿔줌
        self._build_category(category_file) # 위 자료구조를 채우는 함수

    def _build_category(self, category_file):
        ''' category 정보를 저장

        :param category_file: (string) '한국표준산업분류(10차)_국문.xlsx'의 경로
        :return: None
        '''
        excel_data = pandas.read_excel(category_file)

        # 현재 읽고 있는 줄의 카테고리 code 저장
        big_category = None
        mid_category = None
        small_category = None
        sub_category = None
        subsub_category = None

        # 현재 읽고 있는 줄의 카테고리 name 저장
        big_category_name = None
        mid_category_name = None
        small_category_name = None
        sub_category_name = None
        subsub_category_name = None


        '''
        소분류 카테고리 continuous하게 encoding:
            소분류 카테고리 코드를 id로 encoding 해주기 위한 변수
        '''
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
        '''category manager 생성

        :param category_file: (string) '한국표준산업분류(10차)_국문.xlsx'의 경로
        :return: Category의 정보를 담은 category manager
        '''
        return CategoryManager(category_file)

    def get_name_by_code(self, code):
        ''' code로 name을 얻는 함수

        :param code: (string) code
        :return: name
        '''
        return self.category_name_dict[code]

    def get_mid_category(self, small_category):
        ''' 소분류로 중분류 얻는 함수

        :param small_category: (string) 소분류
        :return: 중분류
        '''
        return self.small_to_mid_dict[small_category]

    def get_big_category(self, category_code):
        ''' 소분류나 중분류 code로 대분류 얻는 함수

        :param category_code: (string) 소분류 코드
        :return: None
        '''
        if category_code in self.mid_to_big_dict:
            return self.mid_to_big_dict[category_code]
        if category_code in self.small_to_big_dict:
            return self.small_to_big_dict[category_code]
        raise Exception(f"code {category_code} is not exist.")

    @property
    def small_category_num(self):
        ''' 소분류의 개수

        :return: 소분류의 개수
        '''
        return len(self.small_to_big_dict)

    @property
    def mid_category_num(self):
        ''' 중분류의 개수

        :return: 중분류의 개수
        '''
        return len(self.mid_to_big_dict)

    @property
    def big_category_num(self):
        ''' 대분류의 개수

        :return: 대분류의 개수
        '''
        return len(self.num_mid_per_big_dict)

    def id_to_code(self, id):
        ''' id를 소분류 code로 바꿔줌

        :param id: (int) id
        :return: (string) 소분류 코드
        '''
        return self.id_to_code_dict[id]

    def code_to_code_id(self, code):
        ''' 소분류 code를 id로 바꿔줌

        :param code: (string) 소분류 code
        :return: (int) id
        '''
        return self.code_to_id_dict[code]

    def code_to_one_hot(self,code):
        ''' 소분류 code를 1-hot id로 바꿔줌

        :param code: (string) 소분류 code
        :return: (int) 1-hot id
        '''
        code_id = self.code_to_code_id(code) # code -> id
        one_hot_vector = [0 for _ in range(self.small_category_num)] # id -> 1-hot id
        one_hot_vector[code_id] = 1
        return one_hot_vector

if __name__ == '__main__':
    category_manager = CategoryManager.new_category_manager('data/한국표준산업분류(10차)_국문.xlsx')

    print('hi')