from data_prep_utils import check_for_nulls

from datetime import datetime
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn import metrics, model_selection


# used for dating output file names
date_stamp = datetime.today().strftime('%Y%m%d')

'''
1) Create dataframe(s) and check for null values then encode categorical features
2) Separate train, test, and validation datasets
3) Prep continuous features data so a model can be fitted to it.
    - If you plan on doing things like using the mean to replace null values in a column then use the train set
4) Fit model to the training dataset using kfolds cross validation
5) Evaluate and record area under the ROC curve scores from kfolds
6) (Future Enhancement) Tune model on training data using Optuna and reevaluate performance on test data
7) If the tuned model's performance is good enough, validate performance on the validation set
8) Save the final model using joblib
'''

# 1) Create dataframe(s) and check for null values then encode cat vars
df = pd.read_csv('data/binary_clf.csv')
null_fields = check_for_nulls(df)
print(null_fields)

categorical_cols = [
    'cat0', 'cat1', 'cat2', 'cat3', 'cat4', 'cat5', 'cat6', 'cat7',
    'cat8', 'cat9', 'cat10', 'cat11', 'cat12', 'cat13', 'cat14', 'cat15',
    'cat16', 'cat17', 'cat18'
]

df_w_one_hot = pd.get_dummies(df, columns = categorical_cols, dummy_na=False)

X = df_w_one_hot.drop(labels = ['id', 'target'], axis = 1).values
y = df['target'].values

x_train_test, x_val, y_train_test, y_val = model_selection.train_test_split(X, y, test_size=0.33)

# used for lgb specific methods
lgb_binary_clf_params = {
    'objective': 'binary',
    'metric': 'auc'
}

# lgb_train = lgb.Dataset(x_train, y_train)
# lgb_test = lgb.Dataset(x_test, y_test)
# lgb_valid = lgb.Dataset(x_val, y_val)

skfolds = model_selection.StratifiedKFold(n_splits = 5)
roc_auc_kfold_scores = {}
counter = 0

# kfolds

for train_index, test_index in skfolds.split(x_train_test, y_train_test):
    X_train_folds = x_train_test[train_index]
    y_train_folds = y_train_test[train_index]
    lgb_train = lgb.Dataset(X_train_folds, y_train_folds)
    X_test_fold = x_train_test[test_index]
    y_test_fold = y_train_test[test_index]
    lgb_test = lgb.Dataset(X_test_fold, y_test_fold)
    lgb_clf = lgb.train(
        params=lgb_binary_clf_params,
        train_set=lgb_train,
        valid_sets=[lgb_test],
        num_boost_round = 300,
        early_stopping_rounds = 40
    )
    roc_auc_score = metrics.roc_auc_score(y_true = y_test_fold, y_score = lgb_clf.predict(X_test_fold))
    roc_auc_kfold_scores['Fold ' + str(counter)] = roc_auc_score
    counter += 1

print('ROC AUC Scores from kfolds: \n', roc_auc_kfold_scores)

# x_train_test, x_val, y_train_test, y_val = model_selection.train_test_split(X, y, test_size=0.33)
x_train, x_test, y_train, y_test = model_selection.train_test_split(
    x_train_test,
    y_train_test,
    test_size=0.33
)


lgb_train = lgb.Dataset(x_train_test, y_train_test)
lgb_test = lgb.Dataset(x_test, y_test)


lgb_clf = lgb.train(
    params=lgb_binary_clf_params,
    train_set=lgb_train,
    valid_sets=[lgb_test],
    num_boost_round = 300,
    early_stopping_rounds = 40
)

# validation roc_auc_score

val_roc_auc_score = metrics.roc_auc_score(y_true = y_val, y_score = lgb_clf.predict(x_val))

print('Validation ROC_AUC_Score: ', val_roc_auc_score)

# save model and timestamp name for later use (joblib.load())
try:
    joblib.dump(lgb_clf, 'models/binary_clf_lgb_{dt}.joblib'.format(dt = date_stamp))
    print('Model successfully saved!')
except Exception as e:
    print('Unable to save model using joblib. Exception below: \n')
