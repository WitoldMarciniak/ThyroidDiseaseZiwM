import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import f_classif


# -1 - ?
# 0 - women
# 1 - men

# -1 - other
# 0 - SVHC
# 1  - SVI
# 2  - STMW
# 3  - SVHD
# 4  - WEST

def load_features():
    features = pd.read_csv('files/features.txt', header=None)
    return features[0].tolist()


def load_data_from_files(features):
    data = pd.read_csv('files/data.csv', sep=',', header=None)
    data.columns = features

    features_data = data.iloc[:, :16]
    print features_data

    diagnosis_data = data.iloc[:, -1]
    diagnosis = pd.DataFrame(diagnosis_data)
    diagnosis.columns = ["Diagnosis"]

    return features_data, diagnosis


def select_k_best_features(X_features, Y_diagnosis, k_best_features):
    select_k_best_classifier = SelectKBest(score_func=chi2, k=k_best_features)
    select_k_best_classifier.fit(X_features, Y_diagnosis)

    new_features = select_k_best_classifier.get_support(indices=True)
    X_best_features = X_features.iloc[:, new_features]

    return X_best_features


def select_k_best_features_train_and_test(X_train, Y_train, X_test, k_best_features):
    select_k_best_classifier = SelectKBest(score_func=chi2, k=k_best_features)
    select_k_best_classifier.fit(X_train, Y_train)

    # get columns to keep and create new DataFrame with those only
    new_features = select_k_best_classifier.get_support(indices=True)
    X_train_best_features = X_train.iloc[:, new_features]
    # create second DataFrame from test set which contains only selected features
    X_test_best_features = X_test.iloc[:, new_features]

    return (X_train_best_features, X_test_best_features)


def create_feature_ranking(X_features, Y_diagnosis):
    (chi, pval) = chi2(X_features, Y_diagnosis)

    result = pd.DataFrame(X_features.columns, columns=['Feature name'])
    result["chi"] = chi
    result["pval"] = pval

    result.sort_values(by=['chi'], ascending=False, inplace=True)

    return result


def run_crossvalid(X_features, Y_diagnosis, n_splits, n_neighbors, k_best_features, metric, random_state):
    scores = []

    split_algorithm = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=True)

    for train_samples_indexes, test_samples_indexes in split_algorithm.split(X_features, Y_diagnosis):
        X_train = X_features.iloc[train_samples_indexes]
        X_test = X_features.iloc[test_samples_indexes]
        Y_train = Y_diagnosis.iloc[train_samples_indexes]
        Y_test = Y_diagnosis.iloc[test_samples_indexes]

        X_train_best, X_test_best = select_k_best_features_train_and_test(X_train, Y_train, X_test,
                                                                          k_best_features)

        knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric)
        knn.fit(X_train_best, Y_train.values.ravel())

        scores.append(knn.score(X_test_best, Y_test))

    return scores


def run(features_, diagnosis_):
    columns = ["n_splits", "metric", "k_best_features", "n_neighbors", "Crossvalid run", "Scores"]
    run_results = pd.DataFrame(columns=columns)

    for metric in ["euclidean", "manhattan"]:
        for k_best_features in range(1, 16): # 1 to 20
            for n_neighbors in [1, 5, 10]:
                for run in range(5):
                    random_states = [69, 420, 911, 1004, 2137]
                    score = run_crossvalid(features_, diagnosis_, 2, n_neighbors, k_best_features, metric, random_states[run])

                    run_results = run_results.append({"n_splits" : 2, "metric" : metric, "k_best_features" : k_best_features, "n_neighbors" : n_neighbors, "Crossvalid run" : run, "Scores" : score}, ignore_index=True)

    return run_results


#  >>>>>>> MAIN
features_headers = load_features()

(features, diagnosis) = load_data_from_files(features_headers)

# creates and prints feature ranking using all samples.
# feature_ranking = create_feature_ranking(features, diagnosis)
# print(feature_ranking)

# run and print score of one cross_validation with sample params.
# score = cross_validation.run_crossvalid(X_features, Y_diagnosis, 2, 3, 5, 'manhattan', 420)
# print(score)

# run a function that tests diffrent set of parameters for knn cross_validation
results = run(features, diagnosis)
results.to_csv("result.csv")
print(results)
