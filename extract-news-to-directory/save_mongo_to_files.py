from pymongo import MongoClient
from random import randint
import datetime
import json
from bson import json_util
from bson.json_util import dumps
from bson.json_util import dumps, CANONICAL_JSON_OPTIONS

pathout = "../data/json_news_tagged/"
pathoutbundle = "../data/json_news_tagged_bundle/"

def createjson(item):
  out = dict()
  out["id"] = item["id"]
  out["content"] = item["content"]
  out["date"] = item["date"].strftime("%m/%d/%Y, %H:%M:%S")
  out["tags"] = item["tags"]
  out["newspaper"] = item["newspaper"]
  return out

def save_news_into_jsons(news):
  for item in news:
    #item = createjson(item)
    #print(item)
    filename = pathout + item["id"] + "--" + item["newspaper"] + ".json"
    
    with open(filename, 'w',encoding="utf-8") as outfile:
      json.dump(item, outfile, ensure_ascii=False)


def save_news_into_json_bundle(news):
  filename = pathoutbundle + "large-bundle.json"
  out = []
  for item in news:
    item = createjson(item)
    out = out + [item]
    
  with open(filename, 'w',encoding="utf-8") as outfile:
    json.dump(out, outfile, ensure_ascii=False)

def main():
    client = MongoClient(port=27017)
    db = client["news-scraped-with-tags"]

    #query = {"date": {
    #    "$gte":   datetime.datetime.strptime("2018-11-01", '%Y-%m-%d'),
    #    "$lt":     datetime.datetime.strptime("2019-12-31", '%Y-%m-%d')}
    #}

    query = {"tags":{"$ne" : None}}
    print(query)

    count = db["NewsContentScraped"].find(query)

    #save_news_into_jsons( count)
    save_news_into_json_bundle(count)

if __name__ == "__main__":
    main()



#tar -cJf bundle.tar.xz bundle.json