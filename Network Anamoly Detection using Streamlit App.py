import streamlit as st
import warnings
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import optuna

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, f1_score

warnings.filterwarnings('ignore')

st.title("Machine Learning Model Training and Evaluation")

# File upload
train_file = st.file_uploader("Upload Train Data CSV", type=["csv"])
test_file = st.file_uploader("Upload Test Data CSV", type=["csv"])

if train_file and test_file:
    train = pd.read_csv(train_file)
    test = pd.read_csv(test_file)
    
    # Display data information
    st.subheader("Train Data Information")
    st.write(train.info())
    st.write(train.describe())
    st.write(train.head())
    st.write(train.describe(include='object'))
    st.write(f"Shape of Train Data: {train.shape}")
    st.write(train.isnull().sum())

    # Missing value percentage
    total = train.shape[0]
    missing_columns = [col for col in train.columns if train[col].isnull().sum() > 0]
    missing_info = {col: (train[col].isnull().sum(), round((train[col].isnull().sum() / total) * 100, 3)) for col in missing_columns}
    st.write("Missing Value Information:", missing_info)

    st.write(f"Number of duplicate rows: {train.duplicated().sum()}")

    # Data Visualization
    st.subheader("Data Visualization")
    st.write(sns.countplot(x=train['protocol_type']))
    st.pyplot()
    st.write(sns.countplot(x=train['service']))
    st.pyplot()
    st.write(train['protocol_type'].value_counts())
    st.write(sns.countplot(x=train['class']))
    st.pyplot()
    
    # Label Encoding
    def le(df):
        for col in df.columns:
            if df[col].dtype == 'object':
                label_encoder = LabelEncoder()
                df[col] = label_encoder.fit_transform(df[col])
        return df

    le(train)
    le(test)
    
    st.write("Class distribution in Training set:")
    st.write(train['class'].value_counts())

    # Drop 'num_outbound_cmds' column
    train.drop(['num_outbound_cmds'], axis=1, inplace=True)
    test.drop(['num_outbound_cmds'], axis=1, inplace=True)
    st.write(train.head())

    # Feature Selection
    X_train = train.drop(['class'], axis=1)
    Y_train = train['class']
    rfc = RandomForestClassifier()
    rfe = RFE(estimator=rfc, n_features_to_select=10)
    rfe = rfe.fit(X_train, Y_train)
    selected_features = [feature for support, feature in zip(rfe.get_support(), X_train.columns) if support]
    st.write("Selected Features:", selected_features)

    X_train = X_train[selected_features]
    scale = StandardScaler()
    X_train = scale.fit_transform(X_train)
    test = scale.fit_transform(test)

    x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, train_size=0.70, random_state=2)

    # Model Training and Evaluation
    st.subheader("Model Training and Evaluation")

    # Logistic Regression
    clfl = LogisticRegression(max_iter=1200000)
    clfl.fit(x_train, y_train.values.ravel())
    lg_train, lg_test = clfl.score(x_train, y_train), clfl.score(x_test, y_test)
    st.write(f"Logistic Regression - Training Score: {lg_train}")
    st.write(f"Logistic Regression - Testing Score: {lg_test}")

    # KNN with Optuna
    def objective(trial):
        n_neighbors = trial.suggest_int('KNN_n_neighbors', 2, 16, log=False)
        classifier_obj = KNeighborsClassifier(n_neighbors=n_neighbors)
        classifier_obj.fit(x_train, y_train)
        accuracy = classifier_obj.score(x_test, y_test)
        return accuracy

    study_KNN = optuna.create_study(direction='maximize')
    study_KNN.optimize(objective, n_trials=1)
    st.write("Best Trial for KNN:", study_KNN.best_trial)

    KNN_model = KNeighborsClassifier(n_neighbors=study_KNN.best_trial.params['KNN_n_neighbors'])
    KNN_model.fit(x_train, y_train)
    KNN_train, KNN_test = KNN_model.score(x_train, y_train), KNN_model.score(x_test, y_test)
    st.write(f"KNN - Train Score: {KNN_train}")
    st.write(f"KNN - Test Score: {KNN_test}")

    # Decision Tree with Optuna
    def objective(trial):
        dt_max_depth = trial.suggest_int('dt_max_depth', 2, 32, log=False)
        dt_max_features = trial.suggest_int('dt_max_features', 2, 10, log=False)
        classifier_obj = DecisionTreeClassifier(max_features=dt_max_features, max_depth=dt_max_depth)
        classifier_obj.fit(x_train, y_train)
        accuracy = classifier_obj.score(x_test, y_test)
        return accuracy

    study_dt = optuna.create_study(direction='maximize')
    study_dt.optimize(objective, n_trials=30)
    st.write("Best Trial for Decision Tree:", study_dt.best_trial)

    dt = DecisionTreeClassifier(max_features=study_dt.best_trial.params['dt_max_features'], max_depth=study_dt.best_trial.params['dt_max_depth'])
    dt.fit(x_train, y_train)
    dt_train, dt_test = dt.score(x_train, y_train), dt.score(x_test, y_test)
    st.write(f"Decision Tree - Train Score: {dt_train}")
    st.write(f"Decision Tree - Test Score: {dt_test}")

    # Model comparison
    data = [
        ["KNN", KNN_train, KNN_test],
        ["Logistic Regression", lg_train, lg_test],
        ["Decision Tree", dt_train, dt_test]
    ]
    col_names = ["Model", "Train Score", "Test Score"]
    st.table(pd.DataFrame(data, columns=col_names))

    # Model Validation
    st.subheader("Model Validation")
    models = {
        'KNeighborsClassifier': KNeighborsClassifier(n_neighbors=study_KNN.best_trial.params['KNN_n_neighbors']),
        'LogisticRegression': LogisticRegression(),
        'DecisionTreeClassifier': DecisionTreeClassifier(max_features=study_dt.best_trial.params['dt_max_features'], max_depth=study_dt.best_trial.params['dt_max_depth'])
    }

    scores = {}
    for name in models:
        scores[name] = {}
        for scorer in ['precision', 'recall']:
            scores[name][scorer] = cross_val_score(models[name], x_train, y_train, cv=10, scoring=scorer)

    for name in models:
        st.write(f"{name} Model Validation")
        for scorer in ['precision', 'recall']:
            mean = round(np.mean(scores[name][scorer]) * 100, 2)
            stdev = round(np.std(scores[name][scorer]) * 100, 2)
            st.write(f"Mean {scorer}: {mean}% +- {stdev}")

    scores_df = pd.DataFrame(scores).swapaxes("index", "columns") * 100
    st.bar_chart(scores_df)

    # Model Predictions
    preds = {}
    for name in models:
        models[name].fit(x_train, y_train)
        preds[name] = models[name].predict(x_test)
    st.write("Predictions complete.")

    target_names = ["normal", "anomaly"]

    for name in models:
        st.write(f"{name} Model Testing")
        st.write(confusion_matrix(y_test, preds[name]))
        st.write(classification_report(y_test, preds[name], target_names=target_names))

    f1s = {name: f1_score(y_test, preds[name]) for name in models}
    f1s_df = pd.DataFrame(f1s.values(), index=f1s.keys(), columns=["F1-score"]) * 100
    st.bar_chart(f1s_df)
