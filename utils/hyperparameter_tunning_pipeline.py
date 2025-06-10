import numpy as np
from sklearn.svm import SVR, SVC
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
import xgboost as xgb
from skopt import gp_minimize
from skopt.space import Real, Integer
from utils.cross_validation_pipeline import cross_validation_regressor, cross_validation_classifier


def optimize_random_forest_regressor(X_train, y_train, X_test, y_test, n_calls=30, verbose=True):
    """
    Optimize Random Forest Regressor
    """
    def objective(params):
        n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features = params
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=42,
            n_jobs=-1
        )
        train_score, val_score,_ = cross_validation_regressor(model, X_train, y_train, X_test, y_test)
        generalization_gap = np.abs(np.mean(train_score) - np.mean(val_score))
        tracking_metrics = np.mean(val_score) + 0.3*generalization_gap
        return tracking_metrics
    
    search_space = [
        Integer(50, 300, name='n_estimators'),
        Integer(3, 20, name='max_depth'),
        Integer(2, 20, name='min_samples_split'),
        Integer(1, 10, name='min_samples_leaf'),
        Real(0.1, 1.0, name='max_features')
    ]
    
    result = gp_minimize(objective, search_space, n_calls=n_calls, random_state=0, verbose=verbose)
    
    result.best_params = {
        'n_estimators': result.x[0],
        'max_depth': result.x[1],
        'min_samples_split': result.x[2],
        'min_samples_leaf': result.x[3],
        'max_features': result.x[4],
        'random_state': 42,
        'n_jobs': -1
    }
    result.best_model_class = RandomForestRegressor
    
    return result

def optimize_svr_regressor(X_train, y_train, X_test, y_test, n_calls=30, verbose=True):
    """
    Optimize Support Vector Regression
    """
    def objective(params):
        C, epsilon, gamma = params
        model = SVR(C=C, epsilon=epsilon, gamma=gamma, kernel='rbf')
        train_score, val_score,_ = cross_validation_regressor(model, X_train, y_train, X_test, y_test)
        generalization_gap = np.abs(np.mean(train_score) - np.mean(val_score))
        tracking_metrics = np.mean(val_score) + 0.3*generalization_gap
        return tracking_metrics
    
    search_space = [
        Real(0.1, 100, name='C'),
        Real(1e-6, 1e-1, name='epsilon'),
        Real(1e-6, 1e-1, name='gamma')
    ]
    
    result = gp_minimize(objective, search_space, n_calls=n_calls, random_state=0, verbose=verbose)
    
    result.best_params = {
        'C': result.x[0],
        'epsilon': result.x[1], 
        'gamma': result.x[2],
        'kernel': 'rbf'
    }
    result.best_model_class = SVR
    
    return result

def optimize_xgboost_regressor(X_train, y_train, X_test, y_test, n_calls=30, verbose=True):
    """
    Optimize XGBoost Regressor
    """
    def objective(params):
        n_estimators, max_depth, learning_rate, subsample, colsample_bytree, reg_alpha, reg_lambda = params
        model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )
        train_score, val_score,_ = cross_validation_regressor(model, X_train, y_train, X_test, y_test)
        generalization_gap = np.abs(np.mean(train_score) - np.mean(val_score))
        tracking_metrics = np.mean(val_score) + 0.3*generalization_gap
        return tracking_metrics
    
    search_space = [
        Integer(50, 300, name='n_estimators'),
        Integer(3, 10, name='max_depth'),
        Real(0.01, 0.3, name='learning_rate'),
        Real(0.6, 1.0, name='subsample'),
        Real(0.6, 1.0, name='colsample_bytree'),
        Real(0, 10, name='reg_alpha'),
        Real(0, 10, name='reg_lambda')
    ]
    
    result = gp_minimize(objective, search_space, n_calls=n_calls, random_state=0, verbose=verbose)
    
    result.best_params = {
        'n_estimators': result.x[0],
        'max_depth': result.x[1],
        'learning_rate': result.x[2],
        'subsample': result.x[3],
        'colsample_bytree': result.x[4],
        'reg_alpha': result.x[5],
        'reg_lambda': result.x[6],
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': 0
    }
    result.best_model_class = xgb.XGBRegressor
    
    return result

def optimize_lightgbm_regressor(X_train, y_train, X_test, y_test, n_calls=30, verbose=True):
    """
    Optimize LightGBM Regressor using Bayesian Optimization
    """
    def objective(params):
        n_estimators, max_depth, num_leaves, min_child_samples, learning_rate, subsample, colsample_bytree = params
        model = LGBMRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            num_leaves=num_leaves,
            min_child_samples=min_child_samples,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=42,
            n_jobs=-1
        )
        train_score, val_score,_ = cross_validation_regressor(model, X_train, y_train, X_test, y_test)
        generalization_gap = np.abs(np.mean(train_score) - np.mean(val_score))
        tracking_metrics = np.mean(val_score) + 0.3 * generalization_gap
        return tracking_metrics

    search_space = [
        Integer(50, 300, name='n_estimators'),
        Integer(3, 15, name='max_depth'),
        Integer(50, 100, name='num_leaves'),
        Integer(5, 20, name='min_child_samples'),
        Real(1e-3, 0.3, prior='log-uniform', name='learning_rate'),
        Real(0.6, 1.0, name='subsample'),
        Real(0.6, 1.0, name='colsample_bytree')
    ]

    result = gp_minimize(objective, search_space, n_calls=n_calls, random_state=0, verbose=verbose)

    result.best_params = {
        'n_estimators': result.x[0],
        'max_depth': result.x[1],
        'num_leaves': result.x[2],
        'min_child_samples': result.x[3],
        'learning_rate': result.x[4],
        'subsample': result.x[5],
        'colsample_bytree': result.x[6],
        'random_state': 42,
        'n_jobs': -1
    }
    result.best_model_class = LGBMRegressor

    return result




def optimize_random_forest_classifier(X_train, y_train, X_test, y_test, n_calls=30, verbose=True):
    """
    Optimize Random Forest Classifier using Bayesian Optimization
    """
    def objective(params):
        n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features = params
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=42,
            n_jobs=-1
        )
        train_score, val_score, _ = cross_validation_classifier(model, X_train, y_train, X_test, y_test)
        generalization_gap = np.abs(np.mean(train_score) - np.mean(val_score))
        tracking_metrics = np.mean(val_score) - 0.3 * generalization_gap 
        return -tracking_metrics 

    search_space = [
        Integer(50, 300, name='n_estimators'),
        Integer(3, 20, name='max_depth'),
        Integer(2, 20, name='min_samples_split'),
        Integer(1, 10, name='min_samples_leaf'),
        Real(0.1, 1.0, name='max_features')
    ]
    
    result = gp_minimize(objective, search_space, n_calls=n_calls, random_state=0, verbose=verbose)

    result.best_params = {
        'n_estimators': result.x[0],
        'max_depth': result.x[1],
        'min_samples_split': result.x[2],
        'min_samples_leaf': result.x[3],
        'max_features': result.x[4],
        'random_state': 42,
        'n_jobs': -1
    }
    result.best_model_class = RandomForestClassifier

    return result

def optimize_svc_classifier(X_train, y_train, X_test, y_test, n_calls=30, verbose=True):
    """
    Optimize Support Vector Classifier (SVC) with fixed RBF kernel using Bayesian Optimization.
    """
    def objective(params):
        C, gamma = params
        model = SVC(probability=True,
            C=C,
            gamma=gamma,
            kernel='rbf',
            random_state=42
        )
        train_score, val_score, _ = cross_validation_classifier(model, X_train, y_train, X_test, y_test)
        generalization_gap = np.abs(np.mean(train_score) - np.mean(val_score))
        tracking_metrics = np.mean(val_score) - 0.3 * generalization_gap
        return -tracking_metrics

    search_space = [
        Real(1e-3, 100.0, prior='log-uniform', name='C'),
        Real(1e-4, 10.0, prior='log-uniform', name='gamma')
    ]

    result = gp_minimize(objective, search_space, n_calls=n_calls, random_state=0, verbose=verbose)

    result.best_params = {
        'C': result.x[0],
        'gamma': result.x[1],
        'kernel': 'rbf',
        'random_state': 42
    }
    result.best_model_class = SVC

    return result


def optimize_lightgbm_classifier(X_train, y_train, X_test, y_test, n_calls=30, verbose=True):
    """
    Optimize LightGBM Classifier using Bayesian Optimization
    """
    def objective(params):
        n_estimators, max_depth, num_leaves, min_child_samples, learning_rate, subsample, colsample_bytree = params
        model = LGBMClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            num_leaves=num_leaves,
            min_child_samples=min_child_samples,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=42,
            n_jobs=-1
        )
        train_score, val_score, _ = cross_validation_classifier(model, X_train, y_train, X_test, y_test)
        generalization_gap = np.abs(np.mean(train_score) - np.mean(val_score))
        tracking_metrics = np.mean(val_score) - 0.3 * generalization_gap
        return -tracking_metrics

    search_space = [
        Integer(50, 500, name='n_estimators'),
        Integer(3, 15, name='max_depth'),
        Integer(10, 100, name='num_leaves'),
        Integer(5, 50, name='min_child_samples'),
        Real(1e-3, 0.3, prior='log-uniform', name='learning_rate'),
        Real(0.5, 1.0, name='subsample'),
        Real(0.5, 1.0, name='colsample_bytree')
    ]

    result = gp_minimize(objective, search_space, n_calls=n_calls, random_state=0, verbose=verbose)

    result.best_params = {
        'n_estimators': result.x[0],
        'max_depth': result.x[1],
        'num_leaves': result.x[2],
        'min_child_samples': result.x[3],
        'learning_rate': result.x[4],
        'subsample': result.x[5],
        'colsample_bytree': result.x[6],
        'random_state': 42,
        'n_jobs': -1
    }
    result.best_model_class = LGBMClassifier

    return result

def optimize_xgb_classifier(X_train, y_train, X_test, y_test, n_calls=30, verbose=True):
    """
    Optimize XGBoost Classifier using Bayesian Optimization
    """
    def objective(params):
        n_estimators, max_depth, learning_rate, subsample, colsample_bytree, reg_alpha, reg_lambda = params
        model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42,
            n_jobs=-1
        )
        train_score, val_score, _ = cross_validation_classifier(model, X_train, y_train, X_test, y_test)
        generalization_gap = np.abs(np.mean(train_score) - np.mean(val_score))
        tracking_metrics = np.mean(val_score) - 0.3 * generalization_gap
        return -tracking_metrics

    search_space = [
        Integer(50, 300, name='n_estimators'),
        Integer(3, 10, name='max_depth'),
        Real(0.01, 0.3, name='learning_rate'),
        Real(0.6, 1.0, name='subsample'),
        Real(0.6, 1.0, name='colsample_bytree'),
        Real(0, 10, name='reg_alpha'),
        Real(0, 10, name='reg_lambda')
    ]

    result = gp_minimize(objective, search_space, n_calls=n_calls, random_state=0, verbose=verbose)

    result.best_params = {
        'n_estimators': result.x[0],
        'max_depth': result.x[1],
        'learning_rate': result.x[2],
        'subsample': result.x[3],
        'colsample_bytree': result.x[4],
        'reg_alpha': result.x[5],
        'reg_lambda': result.x[6],
        'use_label_encoder': False,
        'eval_metric': 'logloss',
        'random_state': 42,
        'n_jobs': -1
    }
    result.best_model_class = xgb.XGBClassifier

    return result