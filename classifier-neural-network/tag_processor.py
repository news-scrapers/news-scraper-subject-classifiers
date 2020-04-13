#tar -cJf filename.tar.xz /path/to/folder_or_file ...
#https://blog.mimacom.com/text-classification/

import pandas as pd
import re
import numpy as np
import pickle



class TagProcessor:
    def __init__(self):
        self.df = None
        self.main_tags = []
    


    def extract_main_tags(self):
        list_of_cols = self.df.tags.tolist()
        total = self.df.shape[0]
        gruped_tags = {}
        count = 0 
        for tags in list_of_cols:
            for tag in tags:
                if tag in gruped_tags:
                    gruped_tags[tag] = gruped_tags[tag] + 1
                else:
                    gruped_tags[tag] = 1
            count = count + 1
            print(count, total)

        print(gruped_tags)

    
    def save_main_tags(self):
        pass

    def import_df(self):
        filename = "../data/json_news_tagged_bundle/clean_data.json"
        self.df = pd.read_json(filename)
    

    def saver_results_df(self):
        self.df.to_json("../data/json_news_tagged_bundle/clean_data-unified-tags.json")
        

if __name__== "__main__":
    cleaner = TagProcessor()
    cleaner.import_df()
    cleaner.extract_main_tags()
