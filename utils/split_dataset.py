from sklearn.model_selection import train_test_split

def split(self, test_size: float = 0.2, random_state: int = 42):
    """
    Split dataset into train and test sets.
    """

    X = self.df.drop(columns=[self.target])
    y = self.df[self.target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    print("X_train:", X_train.shape)
    print("X_test :", X_test.shape)
    print("y_train:", y_train.shape)
    print("y_test :", y_test.shape)

    return X_train, X_test, y_train, y_test