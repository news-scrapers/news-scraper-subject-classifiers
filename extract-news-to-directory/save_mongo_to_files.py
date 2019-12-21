from pymongo import MongoClient
from random import randint


def main():
    client = MongoClient(port=27017)
    db=client["news-scraped-with-tags"]
    count = db["NewsContentScraped"].find({})
    print(count[1])

  
if __name__== "__main__":
  main()


