'''
Author: Anish Gupta and Harshil Prajapati
'''

from base import *

from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from functools import partial
from hyperopt import hp, fmin, tpe


algo = partial(tpe.suggest, n_startup_jobs=10)


def auto_turing(args):
    model = XGBClassifier(
        n_jobs=4, n_estimators=args['n_estimators'], max_depth=6)
    model.fit(X_train['num'], y_train.astype(int))
    confidence_valid = model.predict(X_valid['num'])*2 - 1
    score = accuracy_score(confidence_valid > 0, y_valid)
    print(args, score)
    score = -score
    return score


if __name__ == "__main__":
    model = RandomForestClassifier(n_estimators=50, criterion='gini',
                                   max_depth=None,
                                   min_samples_split=2,
                                   min_samples_leaf=1,
                                   min_weight_fraction_leaf=0.0,
                                   max_features='auto',
                                   max_leaf_nodes=None,
                                   min_impurity_split=1e-07,
                                   bootstrap=True,
                                   oob_score=False,
                                   n_jobs=1,
                                   random_state=None,
                                   verbose=0,
                                   warm_start=False,
                                   class_weight=None)

    model.fit(X_train['num'], y_train.astype(int))
    confidence_valid = model.predict(X_valid['num'])*2 - 1
    score = accuracy_score(confidence_valid > 0, y_valid)
    print(score)

    # calculation of actual metric that is used to calculate final score
    r_valid = r_valid.clip(-1, 1)
    x_t_i = confidence_valid * r_valid * u_valid
    data = {'day': d_valid, 'x_t_i': x_t_i}
    df = pd.DataFrame(data)
    x_t = df.groupby('day').sum().values.flatten()
    mean = np.mean(x_t)
    std = np.std(x_t)
    score_valid = mean / std
    print(score_valid)
