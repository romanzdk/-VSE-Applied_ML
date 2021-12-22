# Czech table tennis games prediction

Project created as a semestral work for *Practical applications of machine learning* subject at university.
The target was to get data, prepare it and create machine learning model.

## Data
I scraped entire EloST.cz database as of 12/2021. Used BeautifulSoup4. All in all it is about 2GB of csv data (~6M rows, 21 columns).

## Stack
- requests
- beautifulsoup
- pandas
- pyspark
- matplotlib
- seaborn

Using pyspark instead of pandas because of the dataset size. 