import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from joblib import dump, load

class XGBoostModel:
    def __init__(self, param_dist=None):
        self.xgb_clf = xgb.XGBClassifier()
        
        self.param_dist = param_dist or {
            "n_estimators": [250, 350, 450],
            "max_depth": range(15, 20, 1),
            "gamma": [0.05, 0.06],
            "eta": [0.1, 0.2, 0.3]
        }
        self.grid = None

    def train(self, X_train, y_train):
        
        cv_log = StratifiedKFold(n_splits=5)
        self.grid = GridSearchCV(self.xgb_clf, self.param_dist, cv=cv_log, refit=True, scoring='roc_auc', verbose=2)
        self.grid.fit(X_train, y_train)
        print("Best hyperparameters:", self.grid.best_params_)
    
    def save_model(self, filename):
        
        if self.grid is not None:
            dump(self.grid, filename)
        else:
            raise Exception("Model not trained yet. Train the model before saving.")
    
    def load_model(self, filename):
       
        self.grid = load(filename)
    
    def predict(self, X_test):
       
        if self.grid is not None:
            return self.grid.predict(X_test)
        else:
            raise Exception("Model not trained yet. Train or load the model before predicting.")
