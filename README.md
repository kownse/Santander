# [Santander Value Prediction (Leak Hunt) Challenge](https://www.kaggle.com/c/santander-value-prediction-challenge)
top 7% solution for Santander Value Prediction Challenge

The original task of this competition is to predict the value of transactions for each potential customer.
But the data provided is not a regular structure data  has no column names.
In the first month of this competition, kagglers were trying GBTs but no one got RMSLE less than 1.0.
Then [Giba](https://www.kaggle.com/titericz) who's rank 1st in kaggle global found that the data is a time series in both 2 dimensions,
and this turns the competition into a leak hunt for the repeatable series hidden in the data.

## Hunt for leaky Series
The straight way to find more leak was to extend Giba's series by brute force compute. 
[Jiazhen Xi](https://www.kaggle.com/johnfarrell/giba-s-property-extended-extended-result) did a good job in this method.
But it was obvious that one can not get good rank with only one series.
So we tried many thing else.

## T-SNE and Directions
I learnt to use tsne to find the series hidden from the forum but the results have no directions which means they are sets no series.

I developed a algorithm to get the directions using the observation found in the data that the series should be visually like a inverted
triangle shape in which the later columns and rows are shift from the previous ones.
I found 16 valid series with right directions in this method. And it was a better result from the forum because those series do not have 
enough length to give in prediction.

## Gready Search
The next question is to find the right combination from the 16 series to train a GBT.
I used gready search that from the begining select only one series and train the model, then select the best series into the set,
and repeat this procedure until there is no more series to be add which can decrease the RMSLE.
Finnaly, I add a second pass which try to remove one series from the selections to get smaller RMSLE.

I didn't do ensemble nor stack because I've got only 10 days for this competition.

## Results
It was lucky that I've learnt and tried many new things in this competition.
But the kaggle platform made some mistake which take late submission into the private leaderboard. 
This caused a lot complaint but my rank didn't drop too much from that.
