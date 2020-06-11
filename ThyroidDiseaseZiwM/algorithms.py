import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2


def find_best_features(X, y):
    (chi, pvalue) = chi2(X, y)
    result = pd.DataFrame(X.columns, columns=['Feature name'])
    result['chi'] = chi
    result['pvalue'] = pvalue
    result.sort_values(by=['chi'], ascending=False, inplace=True)
    return result


def select_k_best_features(X, Y, k_best_features):
    select_k_best_classifier = SelectKBest(score_func=chi2, k=k_best_features)
    select_k_best_classifier.fit(X, Y)

    # get columns to keep and create new DataFrame with those only
    new_features = select_k_best_classifier.get_support(indices=True)
    X_best_features = X.iloc[:, new_features]

    return X_best_features


