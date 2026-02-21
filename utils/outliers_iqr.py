import pandas as pd
import numpy as np

class OutlierIQR:

    def __init__(self, df):
        self.df = df

    def remove_outliers_iqr(self, columns=None, factor=1.5):

        """
        Remove outliers using IQR method.

        Parameters:
        df : pandas DataFrame
        columns : list of columns to check (default = all numeric columns)
        factor : IQR multiplier (default = 1.5)

        Returns:
        Cleaned DataFrame without outliers
        """

        df_clean = self.df.copy()

        if columns is None:
            columns = df_clean.select_dtypes(include=np.number).columns

        for col in columns:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR

            df_clean = df_clean[
                (df_clean[col] >= lower_bound) &
                (df_clean[col] <= upper_bound)
            ]

        # Return the DataFrame without outliers
        return df_clean