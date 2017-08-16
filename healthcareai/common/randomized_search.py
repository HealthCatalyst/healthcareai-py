from sklearn.model_selection import RandomizedSearchCV


def get_algorithm(estimator,
                  scoring_metric,
                  hyperparameter_grid,
                  randomized_search,
                  number_iteration_samples=10,
                  **non_randomized_estimator_kwargs):
    """
    Given an estimator and various params, initialize an algorithm with optional randomized search.

    Args:
        estimator (sklearn.base.BaseEstimator): a scikit-learn estimator (for example: KNeighborsClassifier)
        scoring_metric (str): The scoring metric to optimized for if using random search. See
            http://scikit-learn.org/stable/modules/model_evaluation.html
        hyperparameter_grid (dict): An object containing key value pairs of the specific hyperparameter space to search
            through.
        randomized_search (bool): Whether the method should return a randomized search estimator (as opposed to a
            simple algorithm).
        number_iteration_samples (int): If performing randomized search, this is the number of samples that are run in 
            the hyperparameter space. Higher numbers will be slower, but end up with better results, since it is more
            likely that the true optimal hyperparameter is found.
        **non_randomized_estimator_kwargs: Keyword arguments that you can pass directly to the algorithm. Only used when
            radomized_search is False

    Returns:
        sklearn.base.BaseEstimator: a scikit learn algorithm ready to `.fit()`

    """
    if randomized_search:
        algorithm = RandomizedSearchCV(estimator=estimator(),
                                       scoring=scoring_metric,
                                       param_distributions=hyperparameter_grid,
                                       n_iter=number_iteration_samples,
                                       verbose=0,
                                       n_jobs=1)

    else:
        algorithm = estimator(**non_randomized_estimator_kwargs)

    return algorithm
