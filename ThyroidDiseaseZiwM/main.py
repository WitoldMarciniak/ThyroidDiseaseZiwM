import pandas as pd
import random
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier


# 20 - other
# 0 - SVHC
# 1  - SVI
# 2  - STMW
# 3  - SVHD
# 4  - WEST

# pobieranie naglowkow cech z fliku features.txt
def get_features():
    features = pd.read_csv('files/features.txt', header=None)
    return features[0].tolist()


# pobieranie danych o cechach i diagnozie z pliku z files/data.csv
def get_data(features):
    data = pd.read_csv('files/data.csv', sep=',', header=None)
    data.columns = features

    features_data = data.iloc[:, :16]
    diagnosis_data = data.iloc[:, -1]
    diagnosis = pd.DataFrame(diagnosis_data)
    diagnosis.columns = ["Result"]

    return features_data, diagnosis


# szukanie K najlepszych cech
def get_best_features(features, results, k):
    classifier = SelectKBest(score_func=chi2, k=k)
    classifier.fit(features, results)
    found_features = classifier.get_support(indices=True)
    return features.iloc[:, found_features]


# dzielenie zbioru danych na testujace i trenujace wedlug najlepszej obliczonej proporcji
def split_for_train_and_test_features(data_train, results_train, data_test, k):
    classifier = SelectKBest(score_func=chi2, k=k)
    classifier.fit(data_train, results_train)
    new_features = classifier.get_support(indices=True)

    train_best = data_train.iloc[:, new_features]
    test_best = data_test.iloc[:, new_features]

    return train_best, test_best


# funkcja tylko do wyswietlenia rankingu cech, nie sluzy do obliczen knn
def get_ranking(features, results):
    (chi, pval) = chi2(features, results)

    result = pd.DataFrame(features.columns, columns=['Feature name'])
    result["chi"] = chi
    result.sort_values(by=['chi'], ascending=False, inplace=True)
    return result


# podwojna walidacja krzyzowa
def cross_valid(features, diagnosis, n, k_best_features, metric, offset):
    split = StratifiedKFold(n_splits=2, random_state=offset, shuffle=True).split(features, diagnosis)
    scores = []

    for train_samples_indexes, test_samples_indexes in split:
        features_train = features.iloc[train_samples_indexes]
        diagnosis_train = diagnosis.iloc[train_samples_indexes]
        features_test = features.iloc[test_samples_indexes]
        diagnosis_test = diagnosis.iloc[test_samples_indexes]

        train, test = split_for_train_and_test_features(features_train, diagnosis_train, features_test,
                                                        k_best_features)

        knn = KNeighborsClassifier(n_neighbors=n, metric=metric)
        knn.fit(train, diagnosis_train.values.ravel())

        scores.append(knn.score(test, diagnosis_test))

    return scores


def run_knn(features, diagnosis):
    columns = ["metric", "k_best_features", "n_neighbors", "Scores"]
    results = pd.DataFrame(columns=columns)
    randoms = [random.randint(0, 10000000), random.randint(0, 10000000), random.randint(0, 10000000),
               random.randint(0, 10000000), random.randint(0, 10000000)]

    for metric in ["euclidean", "manhattan"]:
        for k_best_features in range(1, 16):
            for n_neighbors in [1, 5, 10]:
                estimated_score = 0l
                real_score = 0l
                for run in range(5):
                    score = cross_valid(features, diagnosis, n_neighbors, k_best_features, metric,
                                        randoms[run])
                    estimated_score += score[0]
                    real_score += score[1]

                error = abs(estimated_score - real_score) / real_score

                results = results.append({"Metric": metric, "K": k_best_features, "N": n_neighbors,
                                          "Scores": [estimated_score / 5, real_score / 5], "Relative error": error},
                                         ignore_index=True)

    return results


#  >>>>>>> MAIN

features_headers = get_features()
(features, diagnosis) = get_data(features_headers)

feature_ranking = get_ranking(features, diagnosis)
print(feature_ranking)

results = run_knn(features, diagnosis)
sorted = results.sort_values(by='Relative error')
sorted.to_csv("result2.csv")
print(sorted)
