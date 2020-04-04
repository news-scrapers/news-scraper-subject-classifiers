const MongoClient = require('mongodb').MongoClient;
require('dotenv').config()
const url = process.env["database_url"];
const fs = require("fs");

const pathout = "../data/json_news_tagged/"
const pathoutbundle = "../data/json_news_tagged_bundle/"

MongoClient.connect(url, function (err, db) {
  if (err) throw err;
  var dbo = db.db("news-scraped-with-tags");
  var query = {
    "tags": { "$ne": null }, 
    "date": {
      "$gte": new Date("2013-05-01"),
      "$lt": new Date("2021-05-01")
    },
   'tags.4': { "$exists": true } }



  dbo.collection("NewsContentScraped").find(query).toArray(function (err, result) {
  if (err) throw err;
  console.log(url)
  let data = JSON.stringify(result);
  fs.writeFileSync(pathoutbundle + "large-bundle-corona.json", data);

  db.close();
});
});