# Making Predictions

Healthcareai provides a few options when you want to get predictions from a trained model. Specifically these predictions come from an instance of TrainedSupervisedModel.

Please note that you will likely only need one of these prediction output types.

Each prediction output format is detailed below.

## Predictions Only

By passing the `.make_predictions(prediction_dataframe)` method a raw prediction dataframe you'll get back a dataframe containing the grain id and predicted values.

```python
# Make some predictions
predictions = trained_model.make_predictions(prediction_dataframe)
print(predictions.head())
```

## Important Factors

By passing the `.make_factors(prediction_dataframe)` method a raw prediction dataframe you'll get back a dataframe containing the grain id and top predictive factors.

```python
# Get the important factors
factors = trained_model.make_factors(prediction_dataframe)
print(factors.head())
```

## Predictions + Factors

By passing the `.make_predictions_with_k_factors(prediction_dataframe)` method a raw prediction dataframe you'll get back a dataframe containing the grain id and predicted values, and top factors.

```python
# Get predictions + factors
predictions_with_factors_df = trained_model.make_predictions_with_k_factors(prediction_dataframe)
print(predictions_with_factors_df.head())
```

## Original Dataframe + Predictions + Factors

By passing the `.make_original_with_predictions_and_factors(prediction_dataframe)` method a raw prediction dataframe you'll get back a dataframe containing all the original data, the predicted values, and top factors.

```python
# Get original dataframe + predictions + factors
original_plus_predictions_and_factors = trained_model.make_original_with_predictions_and_factors(
    prediction_dataframe)
print(original_plus_predictions_and_factors.head())
```







