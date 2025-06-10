import copy
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import root_mean_squared_error, r2_score, precision_recall_curve, auc
from imblearn.over_sampling import SMOTE  



def cross_validation_regressor(model, X_train, y_train, X_test, y_test, k=10, random_state=2406):
    """
    Perform k-fold cross-validation for regression models.
    
    This function evaluates a regression model using k-fold cross-validation,
    calculating RMSE and R² scores for each fold. It then trains a final model
    on the full training set and evaluates it on the test set.
    
    Parameters
    ----------
    model : sklearn estimator
        A scikit-learn compatible regression model (must implement fit and predict methods).
    X_train : array-like of shape (n_samples, n_features)
        Training feature matrix.
    y_train : array-like of shape (n_samples,)
        Training target values.
    X_test : array-like of shape (n_test_samples, n_features)
        Test feature matrix.
    y_test : array-like of shape (n_test_samples,)
        Test target values.
    k : int, default=10
        Number of folds for cross-validation.
    random_state : int, default=2406
        Random state for reproducible results.
    
    Returns
    -------
    tuple of (list, list, float)
        - train_scores : list of float
            RMSE scores on training folds for each CV fold.
        - val_scores : list of float
            RMSE scores on validation folds for each CV fold.
        - test_score : float
            RMSE score on the final test set.
    """
    
    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
    train_scores = []
    val_scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), 1):
        # Split data
        X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
        y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]
        
        # Create a copy of the model
        model_fold = copy.deepcopy(model)
        
        # Train model
        model_fold.fit(X_train_fold, y_train_fold)
        
        # Predict and calculate RMSE
        y_train_pred = model_fold.predict(X_train_fold)
        y_val_pred = model_fold.predict(X_val_fold)
        

        rmse_train = root_mean_squared_error(y_train_fold, y_train_pred)
        rmse_val = root_mean_squared_error(y_val_fold, y_val_pred)
        
        r2_val = r2_score(y_val_fold, y_val_pred)

        train_scores.append(rmse_train)
        val_scores.append(rmse_val)

        print(f"Fold {fold}: RMSE_train = {rmse_train}; RMSE_val = {rmse_val}; R2_val = {r2_val}")
    
    final_model = copy.deepcopy(model)
    final_model.fit(X_train, y_train)
    y_test_pred = final_model.predict(X_test)
    test_score = root_mean_squared_error(y_test, y_test_pred)

    return train_scores, val_scores, test_score


def cross_validation_classifier(model, X_train, y_train, X_test, y_test, k=10, random_state=2406):
    """
    Perform stratified k-fold cross-validation for binary classification models with SMOTE oversampling.
    
    This function evaluates a binary classification model using stratified k-fold cross-validation.
    It applies SMOTE oversampling to handle class imbalance in each training fold and calculates
    precision-recall AUC scores. A final model is trained on the original training set and 
    evaluated on the test set.
    
    Parameters
    ----------
    model : sklearn estimator
        A scikit-learn compatible classification model (must implement fit and predict_proba methods).
    X_train : array-like of shape (n_samples, n_features)
        Training feature matrix.
    y_train : array-like of shape (n_samples,)
        Training target labels (binary: 0 or 1).
    X_test : array-like of shape (n_test_samples, n_features)
        Test feature matrix.
    y_test : array-like of shape (n_test_samples,)
        Test target labels (binary: 0 or 1).
    k : int, default=10
        Number of folds for cross-validation.
    random_state : int, default=2406
        Random state for reproducible results (used for both StratifiedKFold and SMOTE).
    
    Returns
    -------
    tuple of (list, list, float)
        - train_scores : list of float
            Precision-recall AUC scores on oversampled training folds for each CV fold.
        - val_scores : list of float
            Precision-recall AUC scores on validation folds for each CV fold.
        - auc_score_test : float
            Precision-recall AUC score on the final test set.
    """
    
    kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
    train_scores = []
    val_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train), 1):
        # Split data
        X_train_fold = X_train[train_idx]
        X_val_fold = X_train[val_idx]
        y_train_fold = y_train[train_idx]
        y_val_fold = y_train[val_idx]
        
        # Apply SMOTE or RandomOverSampler
        oversampler = SMOTE(random_state=random_state)
        X_train_fold_res, y_train_fold_res = oversampler.fit_resample(X_train_fold, y_train_fold)

        # Create a copy of the model
        model_fold = copy.deepcopy(model)
        
        # Train model
        model_fold.fit(X_train_fold_res, y_train_fold_res)

        # Predict
        y_train_pred = model_fold.predict_proba(X_train_fold_res)[:, 1]
        y_val_pred = model_fold.predict_proba(X_val_fold)[:, 1]


        # Calculate AUC (adjust as needed)
        precision, recall, thresholds = precision_recall_curve(y_train_fold_res, y_train_pred)
        auc_score_train = auc(recall, precision)
        precision, recall, thresholds = precision_recall_curve(y_val_fold, y_val_pred)
        auc_score_val = auc(recall, precision)


        train_scores.append(auc_score_train)
        val_scores.append(auc_score_val)

        print(f"Fold {fold}: AUC_score_train = {auc_score_train}; AUC_score_val = {auc_score_val}")
    
    final_model = copy.deepcopy(model)
    final_model.fit(X_train, y_train)
    y_test_pred = np.array(final_model.predict_proba(X_test)[:, 1])
    precision, recall, thresholds = precision_recall_curve(y_test, y_test_pred)
    auc_score_test = auc(recall, precision)

    return train_scores, val_scores, auc_score_test
