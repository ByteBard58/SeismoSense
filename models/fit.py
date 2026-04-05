from sklearn.ensemble import BaggingClassifier
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split,RandomizedSearchCV,learning_curve
from sklearn.base import BaseEstimator

# Models + SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

import numpy as np
import pandas as pd
import joblib
import time

def load_data(path="dataset/earthquake_data.csv") -> np.ndarray:
  df = pd.read_csv(path)

  x = df.iloc[:,:-1].to_numpy()
  y_unenc = df.iloc[:,-1]

  labelenc = LabelEncoder()
  y = labelenc.fit_transform(y_unenc)

  original_features = labelenc.classes_
  encoded = range(len(original_features))
  mapped = dict(zip(original_features,encoded)) # Saving encoded names

  x_train,x_test,y_train,y_test = train_test_split(x,y,
  test_size=0.2,random_state=4,shuffle=True,stratify=y
  )
  
  return x_train,x_test,y_train,y_test

def get_column_names(path = "dataset/earthquake_data.csv") -> np.ndarray:
  df = pd.read_csv(path)
  column_names = df.columns
  column_names = column_names.to_numpy()
  return column_names

def model() -> BaseEstimator:
  x_train,x_test,y_train,y_test = load_data()
  xgb_model = xgb.XGBClassifier(random_state=91)
  rf_model = RandomForestClassifier(random_state=92)
  svc_model = SVC(random_state=3,)
  knn_model = KNeighborsClassifier(p=2,metric="minkowski")
  dtree = DecisionTreeClassifier(criterion="entropy",max_depth=4)
  bagging_model = BaggingClassifier(estimator=dtree,random_state=192)

  pipe = Pipeline([
    ("imputation", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
    ("smote", SMOTE(random_state=84)),
    ("model", xgb_model)
  ])

  param_grid = [
    { # XGBoost
      "model":[xgb_model],"model__n_estimators":[500,700,1000,1200],
      "model__learning_rate":[0.01,0.1], "model__max_depth":np.arange(4,11,2)
    },
    { # Random Forest
      "model": [rf_model],"model__n_estimators":np.arange(600,1100,100),
      "model__max_depth": np.arange(5,11,2)
    },
    { # SVC
      "model":[svc_model],"model__kernel":["rbf"],"model__C":[0.01,0.1,1,10,100],
      "model__gamma":[0.01,0.1,1,10,100]
    },
    { # KNN
      "model":[knn_model],"model__n_neighbors":np.arange(3,6,1)
    },
    { # Bagging Classifier with Decision tree estimator
      "model":[bagging_model],"model__n_estimators":[500,700,1000]
    }
  ]

  rscv = RandomizedSearchCV(
    estimator=pipe,param_distributions=param_grid,n_iter=12,
    n_jobs=-1,cv=10,random_state=53,refit=True
  )

  print("Starting Model Training....")
  print("It will take some time. Please sit tight....")
  t1 = time.time()
  rscv.fit(x_train,y_train)
  print(f"✅ Training Completed")
  t2 = time.time()
  minutes, seconds = np.divmod((t2-t1),60)
  print(f"Time Elapsed: {minutes} Minutes {seconds:.2f} Seconds")
  estimator = rscv.best_estimator_
  score = rscv.best_score_
  config = rscv.best_params_
  print(f"Best model: {type(rscv.best_estimator_.named_steps['model']).__name__}")

  y_true = y_test
  y_pred = rscv.predict(x_test)
  print(classification_report(y_true=y_true,y_pred=y_pred))

  return estimator

def dump(estimator,column_names) -> None:
  joblib.dump(estimator,"models/estimator.pkl")    # Dumping the estimator into a .pkl file
  joblib.dump(column_names,"models/names.pkl")     # Dumping the column names into a .pkl file
  print("✅ .pkl files dumped successfully")

def main() -> None:
  estimator = model()
  column_names = get_column_names(path = "dataset/earthquake_data.csv")
  dump(estimator=estimator,column_names=column_names)

if __name__ == "__main__":
  main()
