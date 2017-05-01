from sklearn.model_selection import RandomizedSearchCV


def prepare_randomized_search(
        estimator,
        scoring_metric,
        hyperparameter_grid,
        randomized_search,
        **non_randomized_estimator_kwargs):
    """
    Given an estimator and various params, initialize an algorithm with optional randomized search.

    Args:
        estimator: a scikit-learn estimator (for example: KNeighborsClassifier)
        scoring_metric: The scoring metric to optimized for if using random search. See
            http://scikit-learn.org/stable/modules/model_evaluation.html
        hyperparameter_grid: An object containing key value pairs of the specific hyperparameter space to search
            through.
        randomized_search (bool): Whether the method should return a randomized search estimator (as opposed to a
            simple algorithm).
        **non_randomized_estimator_kwargs: Keyword arguments that you can pass directly to the algorithm. Only used when
            radomized_search is False

    Returns:
        estimator: a scikit learn algorithm ready to `.fit()`

    """
    if randomized_search:
        algorithm = RandomizedSearchCV(estimator=estimator(),
                                       scoring=scoring_metric,
                                       param_distributions=hyperparameter_grid,
                                       n_iter=2,
                                       cv=5,
                                       verbose=0,
                                       n_jobs=1)

    else:
        print('No randomized search. Using {}'.format(estimator))
        algorithm = estimator(**non_randomized_estimator_kwargs)

    return algorithm