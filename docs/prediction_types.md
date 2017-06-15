# Choosing a Prediction Output Type

Healthcareai provides a few options when you want to get predictions from a trained model. Specifically these predictions come from an instance of TrainedSupervisedModel.

Please note that you will likely only need one of these prediction output types.

## Database Setup

Each prediction type has a different set of columns and types. You will need to set up your database tables to receive these with appropriate data types.

An easy way to understand each of the prediction types is to inspect the `.dtypes` property of each returned dataframe. For example: `print(predictions.dtypes)`.

## Prediction Types

Each prediction output format is detailed below.

### Predictions Only

By passing the `.make_predictions(prediction_dataframe)` method a raw prediction dataframe you'll get back a dataframe containing the grain id and predicted values.

```python
# Make some predictions
predictions = trained_model.make_predictions(prediction_dataframe)
print(predictions.head())
```

### Important Factors

By passing the `.make_factors(prediction_dataframe)` method a raw prediction dataframe you'll get back a dataframe containing the grain id and top predictive factors.

```python
# Get the important factors
factors = trained_model.make_factors(prediction_dataframe)
print(factors.head())
```

### Predictions + Factors

By passing the `.make_predictions_with_k_factors(prediction_dataframe)` method a raw prediction dataframe you'll get back a dataframe containing the grain id and predicted values, and top factors.

```python
# Get predictions + factors
predictions_with_factors_df = trained_model.make_predictions_with_k_factors(prediction_dataframe)
print(predictions_with_factors_df.head())
```

### Original Dataframe + Predictions + Factors

By passing the `.make_original_with_predictions_and_factors(prediction_dataframe)` method a raw prediction dataframe you'll get back a dataframe containing all the original data, the predicted values, and top factors.

```python
# Get original dataframe + predictions + factors
original_plus_predictions_and_factors = trained_model.make_original_with_predictions_and_factors(
    prediction_dataframe)
print(original_plus_predictions_and_factors.head())
```

### Health Catalyst EDW Format

Many of our users operate on and in the Health Catalyst ecosystem, and most have standardized on a table format that others may find useful. Please note that if you do intend to use this specific format there is an easier and more robust way to save this to your databaes outlined in the [Health Catalyst EDW Instructions](catalyst_edw_instructions.md).

By passing the `.create_catalyst_dataframe(prediction_dataframe)` method a raw prediction dataframe you'll get back a dataframe containing all the original data, the predicted values, and top factors.

```python
## Health Catalyst EDW specific instructions. Uncomment to use.
# This output is a Health Catalyst EDW specific dataframe that includes grain lumn, the prediction and factors
catalyst_dataframe = trained_model.create_catalyst_dataframe(ediction_dataframe)
print(catalyst_dataframe.head())
```

