from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
from typing import Union
import numpy.typing as npt

def model_evaluation(
    ytrain: Union[npt.ArrayLike, list],
    ytrainpred: Union[npt.ArrayLike, list],
    ytest: Union[npt.ArrayLike, list],
    ytestpred: Union[npt.ArrayLike, list],
    step: str
) -> dict:
    """
    Evaluate regression model performance.

    Parameters
    ----------
    y_train : array-like
        Actual training target values.

    y_train_pred : array-like
        Predicted training target values.

    y_test : array-like
        Actual test target values.

    y_test_pred : array-like
        Predicted test target values.

    step : str
        Name of the model or experiment step.

    Returns
    -------
    dict
        Dictionary containing R2, MAE, and RMSE for train and test sets.
    """
    
    # R2
    r2_train = r2_score(ytrain, ytrainpred)
    r2_test = r2_score(ytest, ytestpred)

    # MAE
    mae_train = mean_absolute_error(ytrain, ytrainpred)
    mae_test = mean_absolute_error(ytest, ytestpred)

    # RMSE
    rmse_train = np.sqrt(mean_squared_error(ytrain, ytrainpred))
    rmse_test = np.sqrt(mean_squared_error(ytest, ytestpred))

    print(f"Model Evaluation - {step}")
    print("\nR2 Score")
    print(f"Train: {r2_train:.4f}")
    print(f"Test : {r2_test:.4f}")

    print("\nMAE")
    print(f"Train: {mae_train:.4f}")
    print(f"Test : {mae_test:.4f}")

    print("\nRMSE")
    print(f"Train: {rmse_train:.4f}")
    print(f"Test : {rmse_test:.4f}")