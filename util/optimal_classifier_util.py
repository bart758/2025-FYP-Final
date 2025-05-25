import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from sklearn.base import clone


def compare_classifiers(X, y, n_iterations=30, test_size=0.2, random_state=42):
    '''
    Plots the average recall performance of the most popular classifiers over certain data.

    :param X: X data.
    :param y: y data.
    :param n_iterations: Number of iterations.
    :return: Plot of classifier recall performance.
    '''

    base_classifiers = {
        "Decision Tree": DecisionTreeClassifier(random_state=0),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        # "Logistic Regression": LogisticRegression(max_iter=1000, solver='liblinear', penalty='l1', class_weight='balanced'),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(random_state=0),
    }

    # voting classifiers
    base_classifiers["Voting Classifier"] = VotingClassifier(
        estimators=[
            ('lr', base_classifiers["Logistic Regression"]),
            ('rf', base_classifiers["Random Forest"]),
            ('knn', base_classifiers["K-Nearest Neighbors"])
        ],
        voting='hard'
    )

    # store recall value for all clfs
    scores = {name: [] for name in base_classifiers}

    for i in range(n_iterations):
        x_train, x_test, y_train, y_test = train_test_split(
            # X, y, test_size=test_size, stratify=y, random_state=random_state + i
            X, y, test_size=test_size, random_state=random_state + i
        )
        for name, clf in base_classifiers.items():
            model = clone(clf)
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            recall = recall_score(y_test, y_pred, average='macro')
            scores[name].append(recall)

    # compute mean recall and std
    classifier_names = list(scores.keys())
    means = [np.mean(scores[name]) for name in classifier_names]
    stds = [np.std(scores[name]) for name in classifier_names]

    # horizontal bar plot with whiskers
    y_pos = np.arange(len(classifier_names))

    plt.figure(figsize=(10, 6))
    plt.barh(y_pos, means, xerr=stds, align='center', color='#FFA500', alpha=0.8, ecolor='#333333', capsize=5)
    plt.yticks(y_pos, classifier_names)
    plt.xlabel('Average Recall (± std)')
    plt.title(f'Classifier Performance Comparison over {n_iterations} Splits')
    plt.xlim(0, 1)
    plt.gca().invert_yaxis() # puts the best score at the top
    plt.grid(True, axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()
