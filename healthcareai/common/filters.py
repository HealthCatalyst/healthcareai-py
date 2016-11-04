import pandas as pd

def remove_datetime_columns(df):
    # TODO: make this work with col names shorter than three letters
    cols = [c for c in df.columns if c[-3:] != 'DTS']
    return df[cols]