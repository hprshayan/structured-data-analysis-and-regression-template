"Analyzing a dataset and fitting regression models on it"

import warnings

from sklearn.exceptions import DataConversionWarning

from src.constants import COMPARISON_CRITERIA, DATASET_METADATA, SEED, TEST_SIZE
from src.data_analysis import DatasetAnalysis
from src.hparam_search import execute_hparam_search
from src.metrics import depict_predicted_targets
from src.model import Pipeline, STDScaler
from src.utils import (
    CategoricalEncoderDecoder,
    create_directories,
    data_train_test_split,
    load_and_init_process_data,
)

warnings.filterwarnings("ignore", category=DataConversionWarning)


def main():
    # prepare the directories
    create_directories()

    # load the dataset, remove and save rows that contain null values and one-hot encode categorial features
    cured_dataset = load_and_init_process_data(DATASET_METADATA)
    categorical_encoder_decoder = CategoricalEncoderDecoder(
        DATASET_METADATA, cured_dataset.columns
    )
    encoded_dataset = categorical_encoder_decoder.encode(cured_dataset)

    # analyze the dataset
    print("analyzing the dataset...")
    dataset_analyzer = DatasetAnalysis(
        encoded_dataset, categorical_encoder_decoder, DATASET_METADATA.target
    )
    dataset_analyzer.analyze_dataset()

    (
        train_features,
        test_features,
        train_targets,
        test_targets,
        col_name_int_dict,
    ) = data_train_test_split(cured_dataset, TEST_SIZE, SEED, DATASET_METADATA.target)

    # encode categorical features
    categorical_encoder_decoder.set_name_idx(col_name_int_dict)
    encoded_train_features = categorical_encoder_decoder.encode_features(train_features)

    # fitting the dataset transformers
    feature_scaler = STDScaler()
    target_scaler = STDScaler()
    feature_scaler.fit(encoded_train_features)
    target_scaler.fit(train_targets)
    Pipeline.feature_scaler = feature_scaler
    Pipeline.target_scaler = target_scaler
    Pipeline.categorical_encoder_decoder = categorical_encoder_decoder

    # run the hyperparameter search
    print("\nexecuting hyperparameter tunning for regression models...")
    best_pipeline = execute_hparam_search(
        train_features,
        test_features,
        train_targets,
        test_targets,
        COMPARISON_CRITERIA,
    )
    print("predicted and target values in testing dataset:")
    depict_predicted_targets(best_pipeline, test_features, test_targets)

    print("\nall done!")
    print(
        '*** the charts and text reports are saved in "figs" and "texts" directories, respectively *** '
    )


if __name__ == "__main__":
    main()
