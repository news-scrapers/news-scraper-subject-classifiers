cd classifier-neural-network
pipenv run python3 cleaner.py
pipenv run python3 tag_processor.py

cd ..
bash compress_clean_data.sh
