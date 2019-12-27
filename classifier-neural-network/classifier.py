import pandas as pd
import glob

path = "../data/json_news_tagged"
pathoutbundle = "../data/json_news_tagged_bundle/"

all_files = glob.glob(path + "/*.json")

li = []

for filename in all_files:
    print(filename)
    df = pd.read_json(filename, orient='columns')
    li.append(df)

frame = pd.concat(li, axis=0, ignore_index=True)