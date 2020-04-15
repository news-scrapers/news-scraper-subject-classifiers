#tar -cJf filename.tar.xz /path/to/folder_or_file ...
#https://blog.mimacom.com/text-classification/

import pandas as pd
import re
import numpy as np
import pickle



class TagProcessor:
    def __init__(self):
        self.df = None
        self.repetitions_threads_hole=50
        self.main_tags = []
    


    def extract_main_tags(self):
        print("extracting main tags using dataframe")
        list_of_cols = self.df.tags.tolist()
        self.total = self.df.shape[0]
        self.gruped_tags = {}
        count = 0 
        for tags in list_of_cols:
            for tag in tags:
                if tag in self.gruped_tags:
                    self.gruped_tags[tag] = self.gruped_tags[tag] + 1
                else:
                    self.gruped_tags[tag] = 1
            count = count + 1
            print("\r", count, self.total)
        return self.gruped_tags

    def depurate_dictionary(self):
        self.list_of_common_tags = []
        for tag in self.gruped_tags.keys():
            if (self.gruped_tags[tag]>self.repetitions_threads_hole):
                self.list_of_common_tags.append(tag)
        print(self.list_of_common_tags)

    
    def add_main_tags_to_news(self):
        print("adding main tags to dataframe")

        self.df['common_tags'] = np.empty((len(self.df), 0)).tolist()
        
        self.df_clean_tags = pd.DataFrame(columns = self.df.columns)

        for index, row in self.df.iterrows():
            common_tags_in_row = []
            for tag in row.tags:
                if tag in self.gruped_tags and self.gruped_tags[tag]>self.repetitions_threads_hole:
                    common_tags_in_row.append(tag)
            if len(common_tags_in_row) == 0:
                self.df.drop([index])
                print("removing row")
            else:
                row.common_tags = common_tags_in_row
                print(common_tags_in_row)
                self.df_clean_tags=self.df_clean_tags.append(row)

            print("\r",index, self.total, self.df_clean_tags.shape[0])


    def import_df(self):
        print("importing data")
        filename = "../data/json_news_tagged_bundle/clean_data.json"
        self.df = pd.read_json(filename)
    

    def saver_results_df(self):
        print("saving new tags")
        self.df_clean_tags.to_json("../data/json_news_tagged_bundle/clean_data-unified-tags.json")
        

if __name__== "__main__":
    cleaner = TagProcessor()
    cleaner.import_df()
    print(cleaner.df)
    cleaner.extract_main_tags()
    cleaner.add_main_tags_to_news()
    cleaner.saver_results_df()
    print(cleaner.df)
