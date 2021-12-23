# Czech table tennis games prediction

Project created as a semestral work for *Practical applications of machine learning* subject at university.
The target was to get data, prepare it and create machine learning model.

## Process
1. Data gathering
2. Data union
3. Exploratory data analysis
4. Feature engineering
5. Modeling

## Data
I scraped entire EloST.cz database as of 12/2021. Used BeautifulSoup4. All in all it is about 2GB of csv data (~6M rows, 21 columns).

## Results
|model|      acc |       f1 | precision |   recall |      auc |
|----:|---------:|---------:|----------:|---------:|---------:|
|  LR | 0.779064 | 0.789827 |  0.794493 | 0.785215 | 0.778690 |
| GBT | 0.765141 | 0.777575 |  0.785662 | 0.769653 | 0.764818 |
|  RF | 0.742364 | 0.759276 |  0.777601 | 0.741795 | 0.742424 |
|  NB | 0.681694 | 0.749676 |  0.912191 | 0.636311 | 0.726736 |
|  DT | 0.739318 | 0.733386 |  0.686170 | 0.787581 | 0.743283 |

## Issues
I had some issues with Java/Spark on my machine, that's why you can see some warnings in the notebooks. If you know what could be the cause of it, please let me know.

## Improve
- hyperparameter tuning
- neural network