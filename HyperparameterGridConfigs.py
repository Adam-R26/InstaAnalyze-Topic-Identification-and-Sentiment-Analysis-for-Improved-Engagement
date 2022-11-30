import numpy as np

class HyperparameterGridConfigs:
    def get_rf_hyperparam_grid(self):
        max_features_range = ['sqrt', 'log2']
        n_estimators_range = np.arange(100,600, 100)
        param_grid = dict(max_features=max_features_range, n_estimators=n_estimators_range)
        return param_grid

    def get_svm_hyperparam_grid(self):
        kernel = ['linear', 'rbf']
        C = [1]
        param_grid = dict(kernel=kernel, C=C)
        return param_grid
    
    