cd .. 
cd data/json_news_tagged_bundle
#gunzip -ck large-bundle-corona.json > reviews-180k-bundle-after-corona.json.gz
zip reviews-400k-bundle-after-corona.zip large-bundle-corona.json 
zip reviews-400k-bundle-after-corona-clean_data.zip clean_data.json 
