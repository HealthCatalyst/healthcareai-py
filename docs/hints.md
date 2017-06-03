# Hints and Tips

## Gathering the data

If you have interesting data in a CSV file or even across serveral
databases on a single server, you are in good shape. While it's easiest
to pull data into healthcareai via a single table, one can also use joins
to gather data from separate tables or databases.

What's most important is the following:

- You have a column you're excited about predicting and some data that might be relevant
- If you're predicting a binary outcome (ie, 0 or 1), you have to [convert the column to be Y or N](https://msdn.microsoft.com/en-us/library/hh213574.aspx).

## Feature Engineering

It's almost always helpful to do some [feature engineering](https://en.wikipedia.org/wiki/Feature_engineering) before
creating a model. Here are some practical examples of that:

- If you think the thing your predicting might have a seasonal pattern, you could [convert](http://stackoverflow.com/a/25149272/5636012) a date-time column into columns representing DayOfWeek, DayOfMonth, WeekOfYear, etc.
- If you have rows with both a latitude and longitude, it may be beneficial to [add a zip code column](https://www.zipcodeapi.com/)

## Model building tips

- Start small. You can often get a good idea of model performance by starting with 10k rows instead of 1M.
- Don't throw out rows with missing values. We'll help you experiment with [imputation](https://en.wikipedia.org/wiki/Imputation_(statistics)), which may improve the model's performance.
- Focus on new features. Rather than finding more rows of the same columns, finding or engineering better columns (ie, features) will give better results.
