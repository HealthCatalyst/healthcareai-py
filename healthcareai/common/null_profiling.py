"""Null Profiling Utilities."""
import pandas as pd
from tabulate import tabulate


def calculate_numeric_column_null_percentages(df):
    """Calculate null percentages in numeric columns."""
    numeric_df = df.select_dtypes(exclude=[object, 'category']).copy()

    return calculate_column_null_percentages(numeric_df)


def calculate_column_null_percentages(df):
    """Calculate null percentages in all columns."""
    nulls = df.isnull().sum()

    return nulls / len(df)


def print_numeric_null_percentages(df, null_header='Percent Null'):
    """Print a nice table of null percentages by column."""
    results = pd.DataFrame(calculate_numeric_column_null_percentages(df))
    table = tabulate(results, headers=['Column', null_header],
                     tablefmt='fancy_grid')
    print(table)
