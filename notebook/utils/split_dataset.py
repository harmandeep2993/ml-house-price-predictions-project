from sklearn.model_selection import train_test_split


def split_dataset(df, target_column):
    """
    Split dataset into train and test sets.

    Parameters
    ----------
    df : pandas DataFrame
        Full dataset including features and target.

    target_column : str
        Name of the target column.

    test_size : float, default=0.2
        Proportion of dataset for testing.

    random_state : int, default=42
        Seed for reproducibility.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """

    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=20
    )

    print("X_train:", X_train.shape)
    print("X_test :", X_test.shape)
    print("y_train:", y_train.shape)
    print("y_test :", y_test.shape)

    return X_train, X_test, y_train, y_test
