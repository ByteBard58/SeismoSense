import marimo

__generated_with = "0.18.1"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # SeismoSense

    The SeismoSense is a ML classifier project which predicts the alret level of an earthquake based on user-given inputs. The project serves through a Flask web app.

    In order to develop the perfect ML model for the project, we need to do lots of research and we also need to write the code somewhere else before putting it in the final script. This notebook is solely created for that purpose. It is used as a playground to test different hyperparameter settings as well as preprocessing approaches.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Importing the libraries
    """)
    return


@app.cell
def _():
    from sklearn.ensemble import BaggingClassifier
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import classification_report
    from sklearn.model_selection import train_test_split, RandomizedSearchCV, learning_curve
    from sklearn.ensemble import RandomForestClassifier
    # Models + SMOTE
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    import xgboost as xgb
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os
    return (
        BaggingClassifier,
        DecisionTreeClassifier,
        KNeighborsClassifier,
        LabelEncoder,
        Pipeline,
        RandomForestClassifier,
        RandomizedSearchCV,
        SMOTE,
        SVC,
        SimpleImputer,
        StandardScaler,
        classification_report,
        learning_curve,
        np,
        pd,
        plt,
        sns,
        train_test_split,
        xgb,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Reading the dataset
    """)
    return


@app.cell
def _(pd):
    df = pd.read_csv("dataset/earthquake_data.csv")
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Printing out various information related to the data
    """)
    return


@app.cell
def _(df):
    df.head(5)
    return


@app.cell
def _(df):
    df.loc[df.alert == "orange", :].head(5)
    return


@app.cell
def _(df):
    df["alert"].value_counts()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Plotting correlational heatmap of the features
    """)
    return


@app.cell
def _(df, plt, sns):
    cor = df.iloc[:,:-1].corr()

    plt.figure(figsize=(9,6))
    sns.heatmap(cor,cmap="icefire",annot=True,fmt=".2f")
    plt.title("Heatmap of feature correlation (Pearson)",fontdict={"fontsize":16})
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Key Takeaways
    - **Strong Positive Correlation**: The strongest positive correlation is observed between cdi and mmi ($\text{r} = 0.68$).

    - **Strong Negative Correlation**: The strongest negative correlation is between depth and mmi ($\text{r} = -0.57$), suggesting that as the depth increases, the Modified Mercalli Intensity tends to decrease.

    - **Moderate Correlations**: Features like magnitude show moderate positive correlations with cdi ($\text{r} = 0.33$), mmi ($\text{r} = 0.30$), and sig ($\text{r} = 0.24$).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Saving the column names
    """)
    return


@app.cell
def _(df):
    column_names = df.columns
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Feature and target column seperation
    """)
    return


@app.cell
def _(df):
    x = df.iloc[:,:-1].to_numpy()
    y_unenc = df.iloc[:,-1]
    return x, y_unenc


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Using LabelEncoder() to encode the class labels into integers
    """)
    return


@app.cell
def _(LabelEncoder, y_unenc):
    labelenc = LabelEncoder()
    y = labelenc.fit_transform(y_unenc)
    return labelenc, y


@app.cell
def _(labelenc):
    original_features = labelenc.classes_
    encoded = range(len(original_features))
    mapped = dict(zip(original_features,encoded))
    print(mapped)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Train-test split
    """)
    return


@app.cell
def _(train_test_split, x, y):
    x_train,x_test,y_train,y_test = train_test_split(x,y,
     test_size=0.2,random_state=4,shuffle=True,stratify=y
    )
    return x_test, x_train, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Pipeline
    """)
    return


@app.cell
def _(
    BaggingClassifier,
    DecisionTreeClassifier,
    KNeighborsClassifier,
    Pipeline,
    RandomForestClassifier,
    SMOTE,
    SVC,
    SimpleImputer,
    StandardScaler,
    xgb,
):
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
    return bagging_model, knn_model, pipe, rf_model, svc_model, xgb_model


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Performing Randomized Search CV
    """)
    return


@app.cell
def _(
    RandomizedSearchCV,
    bagging_model,
    knn_model,
    np,
    pipe,
    rf_model,
    svc_model,
    x_train,
    xgb_model,
    y_train,
):
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
      n_jobs=-1,cv=10,random_state=53
    )

    rscv.fit(x_train,y_train)
    estimator = rscv.best_estimator_
    score = rscv.best_score_
    config = rscv.best_params_

    print(f"Best configuration:\n{config}")
    print(f"Best score = {score}")
    return estimator, rscv


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Printing the classification report
    """)
    return


@app.cell
def _(classification_report, rscv, x_test, y_test):
    y_true = y_test
    y_pred = rscv.predict(x_test)
    print(classification_report(y_true=y_true,y_pred=y_pred))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Plotting the Learning Curve
    """)
    return


@app.cell
def _(estimator, learning_curve, np, x_train, y_train):
    train_size,train_scr,val_scr = learning_curve(estimator,x_train,y_train,
     train_sizes=np.linspace(0.1,1.0,10),cv=10,random_state=65,shuffle=True,n_jobs=-1
    )

    train_mean = np.mean(train_scr,axis=1)
    val_mean = np.mean(val_scr,axis=1)
    train_std = np.std(train_scr,axis=1)
    val_std = np.std(val_scr,axis=1)
    return train_mean, train_size, train_std, val_mean, val_std


@app.cell
def _(plt, train_mean, train_size, train_std, val_mean, val_std):
    plt.figure(figsize=(10,7))
    plt.plot(train_size,train_mean,color="red",label="Training Accuracy",marker="s")
    plt.fill_between(
      train_size,train_mean + train_std, train_mean - train_std, color="red",alpha=0.3
    )

    plt.plot(train_size,val_mean,color="orange",label="Validation Accuracy",marker="v")
    plt.fill_between(
      train_size,val_mean + val_std, val_mean - val_std, color="orange",alpha=0.3
    )

    plt.title("Learning Curve",fontdict={"fontsize":16})
    plt.xlabel("Train Size",fontdict={"fontsize":14})
    plt.ylabel("Accuracy",fontdict={"fontsize":14})
    plt.ylim(0.5,1.2)
    plt.legend()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Key Takeaways
    - **Training Accuracy is Perfect (1.0)**: The training accuracy is consistently $\mathbf{1.0}$ (or $\mathbf{100\%}$) across all training set sizes, suggesting the model has very high capacity and is overfitting the training data.

    - **Validation Accuracy Improves Steadily**: The validation accuracy starts around $\mathbf{0.70}$ for small training sizes and gradually increases to about $\mathbf{0.92}$ as the training set size grows.

    - **Moderate Variance/Bias**: The somewhat large gap between the Training Accuracy and the Validation Accuracy suggests the model suffers from moderate variance. Adding more data (increasing training size) helps reduce this gap slightly, but the training accuracy remains at $1.0$.
    """)
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
