# Czech table tennis games prediction

Project created as a semestral work for *Practical applications of machine learning* subject at university.
The target was to get data, prepare it and create machine learning model.

For me personally it was pretty interesting task with some new things - e.g. distributed Keras NN with Elephas.

Throughout the project I am using:
- `beautifulsoup` + `requests` for data gathering
- `matplotlib` + `seaborn` + `pandas` for data exploration
- `pyspark` + `pandas` for data manipulation
- `pyspark` + `keras` + `elephas` for modeling

## Process
1. Data gathering
2. Data union
3. Exploratory data analysis
4. Feature engineering
5. Modeling

## Data
I scraped entire EloST.cz database as of 12/2021. Used BeautifulSoup4. All in all it is about 2GB of csv data (~6M rows, 21 columns).

**Raw** data sample:

|            Hráč |                 Oddíl |                 Družstvo |               Soupeř |                               Oddíl soupeř |                          Družstvo soupeř | Výsledek | Datum zápasu |                   Soutěž |   Elo hráče |   Elo (min) |   Elo (max) | Elo soupeře | Elo (min) soupeře | Elo (max) soupeře | Rok narození | Max elo | Elo nejlepšího poraženého soupeře |                       ID Hráč |                 ID Soupeř | Rok narození soupeř |
|----------------:|----------------------:|-------------------------:|---------------------:|-------------------------------------------:|-----------------------------------------:|---------:|-------------:|-------------------------:|------------:|------------:|------------:|------------:|------------------:|------------------:|-------------:|--------:|----------------------------------:|------------------------------:|--------------------------:|--------------------:|
|   Tomáš Tregler |      HB Ostrov , z.s. |         HB Ostrov H.Brod | Oleksandr Tymofieiev |                          SKST Hodonín z.s. |                        SKST PLUS Hodonín |      3:0 |    1.10.2021 |           Extraliga mužů | 2447-> 2447 | 2419-> 2419 | 2473-> 2473 | 2167-> 2167 |       2143-> 2143 |       2200-> 2200 |         1990 |  2458.0 |                            2446.0 |   /st/hrac/tregler_tomas_1990 | tymofieiev_oleksandr_1996 |                1996 |
|   Tomáš Tregler | TT Club Ostrava, z.s. |         TTC Ostrava 2016 |    Michal Lebeda ml. |               TJ Sokol PP Hradec Králové 2 | TJ Sokol PP Hradec Králové 2 Lunatour.cz |      3:0 |     2.4.2017 |           Extraliga mužů | 2392-> 2392 | 2364-> 2364 | 2411-> 2411 | 2079-> 2079 |       2060-> 2060 |       2100-> 2100 |         1990 |  2458.0 |                            2446.0 |   /st/hrac/tregler_tomas_1990 |     lebeda_michal_ml_1993 |                1993 |
|   Tomáš Tregler |      HB Ostrov , z.s. | STEN marketing HB Ostrov |         Peter Šereda |                         DTJ Hradec Králové |                       DTJ Hradec Králové |      3:0 |    13.5.2015 |           Extraliga mužů | 2405-> 2409 | 2382-> 2387 | 2427-> 2430 | 2330-> 2326 |       2308-> 2305 |       2347-> 2343 |         1990 |  2458.0 |                            2446.0 |   /st/hrac/tregler_tomas_1990 |         sereda_peter_1984 |                1984 |
| David Reitšpies |             SKST Cheb |                SKST Cheb |            Jan Pisár | Stavební fakulta SK Kotlářka El Niňo Praha |                   SF SKK El Niňo Praha B |      3:0 |    4.12.2021 |  Český pohár II.st. muži | 2419-> 2419 | 2378-> 2378 | 2486-> 2486 | 2047-> 2047 |       2035-> 2035 |       2059-> 2059 |         1996 |  2419.0 |                            2458.0 | /st/hrac/reitspies_david_1996 |            pisar_jan_2004 |                2004 |
| David Reitšpies |             SKST Cheb |                SKST Cheb |       Martin Olejník |              Klub stolního tenisu KT Praha |                                 KT Praha |      3:0 |    24.9.2021 |           Extraliga mužů | 2368-> 2376 | 2305-> 2316 | 2447-> 2453 | 2345-> 2342 |       2326-> 2323 |       2361-> 2357 |         1996 |  2419.0 |                            2458.0 | /st/hrac/reitspies_david_1996 |       olejnik_martin_1972 |                1972 |



Model inputs:

| Rok narození | Elo hráče před | Hráč je žena | Max elo | Elo nejlepšího poraženého soupeře | Rok narození soupeř | Elo soupeře před | Soupeř je žena | Max elo_soupeř | Elo nejlepšího poraženého soupeře_soupeř | Den v týdnu | Víkend | obdobi | Hráč vyhrál |
|-------------:|---------------:|-------------:|--------:|----------------------------------:|--------------------:|-----------------:|---------------:|---------------:|-----------------------------------------:|------------:|-------:|-------:|------------:|
|       1990.0 |         2454.0 |          0.0 |  2458.0 |                            2446.0 |              1997.0 |           2377.0 |            0.0 |         2404.0 |                                   2425.0 |         1.0 |    0.0 |    4.0 |         1.0 |
|       1990.0 |         2453.0 |          0.0 |  2458.0 |                            2446.0 |              1996.0 |           2241.0 |            0.0 |         2264.0 |                                   2339.0 |         2.0 |    0.0 |    4.0 |         1.0 |
|       1990.0 |         2458.0 |          0.0 |  2458.0 |                            2446.0 |              1996.0 |           2419.0 |            0.0 |         2419.0 |                                   2458.0 |         1.0 |    0.0 |    4.0 |         0.0 |
|       1990.0 |         2452.0 |          0.0 |  2458.0 |                            2446.0 |              1998.0 |           2315.0 |            0.0 |         2324.0 |                                   2419.0 |         2.0 |    0.0 |    4.0 |         1.0 |
|       1990.0 |         2454.0 |          0.0 |  2458.0 |                            2446.0 |              2002.0 |           2318.0 |            0.0 |         2327.0 |                                   2454.0 |         2.0 |    0.0 |    4.0 |         0.0 |

## Results
|model|      acc |       f1 | precision |   recall |      auc |
|----:|---------:|---------:|----------:|---------:|---------:|
|  LR | 0.779064 | 0.789827 |  0.794493 | 0.785215 | 0.778690 |
| GBT | 0.765141 | 0.777575 |  0.785662 | 0.769653 | 0.764818 |
|  RF | 0.742364 | 0.759276 |  0.777601 | 0.741795 | 0.742424 |
|  NB | 0.681694 | 0.749676 |  0.912191 | 0.636311 | 0.726736 |
|  DT | 0.739318 | 0.733386 |  0.686170 | 0.787581 | 0.743283 |
|  NN | 0.000000 |

## Issues
I had some issues with Java/Spark on my machine, that's why you can see some warnings in the notebooks. If you know what could be the cause of it, please let me know.

## Improve
- hyperparameter tuning
- neural networks
- additional data
