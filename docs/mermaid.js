graph LR
subgraph Data Sources
    database
    csv
end

database-->dataframe
csv-->dataframe
dataframe-->SMT_init

subgraph TrainedSupervisedModel
    subgraph TSM_Properties
        algorithm_name
        is_classification
        is_regression
        best_hyperparameters
        model_type
        binary_classification_scores
        metrics
    end
    subgraph TSM_Properties2
        model
        feature_model
        fit_pipeline
        column_names
        _model_type
        grain_column
        prediction_column
        test_set_predictions
        test_set_class_labels
        test_set_actual
        _metric_by_name
    end

    subgraph TSM_methods
        TSM_init[__init__]
        save
        make_predictions
        prepare_and_subset
        make_factors
        make_predictions_with_k_factors
        make_original_with_predictions_and_features
        create_catalyst_dataframe
        predict_to_catalyst_sam
        predict_to_sqlite
        roc_curve_plot
        roc
        pr_curve_plot
        pr
        validate_classification
    end
end

subgraph SupervisedModelTrainer
    SMT_init-->ASMT_init
    SMT_init[__init__]-->full_pipeline

    knn-->knn2
    random_forest-->random_forest_classification_a
    random_forest-->random_forest_regression_a
    logistic_regression-->logistic_regression2
    linear_regression-->linear_regression2
 subgraph AdvancedSupervisedModelTrainer
        ASMT_init[__init__]
        knn2-->TSM_init
        random_forest_classification_a-->TSM_init
        random_forest_regression_a-->TSM_init
        logistic_regression2-->TSM_init
        linear_regression2-->TSM_init
    end
end

subgraph toolbox
    subgraph model_eval.py
        compute_roc
        compute_pr
        validate_predictions_and_labels_are_equal_length
        calculate_regression_metrics
        calculate_binary_classification_metrics
        tsm_classification_comparison_plots
        roc_plot_from_thresholds
        pr_plot_from_thresholds
        plot_rf_from_tsm
        plot_random_forest_feature_importance
        get_estimator_from_trained_supervised_model
        get_estimator_from_meta_estimator
        get_hyperparameters_from_meta_estimator
    end

    subgraph data_preparation.py
        full_pipeline
    end

    full_pipeline-->DataFrameImputer
    full_pipeline-->DataFrameConvertTargetToBinary
    full_pipeline-->DataFrameCreateDummyVariables
    full_pipeline-->DataFrameConvertColumnToNumeric
    full_pipeline-->DataFrameUnderSampling
    full_pipeline-->DataFrameOverSampling
    full_pipeline-->DataframeDateTimeColumnSuffixFilter
    full_pipeline-->DataframeColumnRemover
    full_pipeline-->DataframeNullValueFilter

    subgraph transformers.py
        DataFrameImputer
        DataFrameConvertTargetToBinary
        DataFrameCreateDummyVariables
        DataFrameConvertColumnToNumeric
        DataFrameUnderSampling
        DataFrameOverSampling
    end

    subgraph filters.py
        DataframeDateTimeColumnSuffixFilter
        DataframeColumnRemover
        DataframeNullValueFilter
    end
end

class model_eval pythonModule;

classDef pythonClass fill:#00ff33;
classDef pythonModule fill:#ff1100;

class Trainer pythonClass;
class AdvancedTrainer pythonClass;
class TSM pythonClass;