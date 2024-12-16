# LPCVC 2025 Track 1

**Image classification for different lighting conditions and styles**

## Overview

This repository contains Python scripts designed for the referee system to compile and evaulate the models.

## File Descriptions

### 1. **referee_compile.py**

-   **Purpose**: Automates the process of compiling machine learning models on aihub.
-   **Key Functions**:
    -   `compile_model(model, device, input_shape)`: Submits a compile job with specific configurations.
    -   `fetch_model(model_id)`: Fetches a model using its ID.
    -   `read_model_ids_from_csv(csv_file)`: Reads model IDs from a CSV file.
-   **Workflow**:
    1.  Reads a CSV containing model IDs (`test_models.csv`).
    2.  Fetches models using the `qai_hub` API.
    3.  Compiles each model for the `Samsung Galaxy S24 (Family)` device.
    4.  Outputs a CSV (`compiled_jobs_batch.csv`) containing the compiled model IDs.


### 2. **referee_evaluate.py**

-   **Purpose**: Evaluates the compiled models by running inference and profiling jobs.
-   **Key Functions**:
    -   `read_compiled_jobs_from_csv(compiled_csv_file)`: Reads compiled model IDs from a CSV file.
    -   `run_profile(model, device)`: Profiles model performance.
    -   `run_inference(model, device, input_dataset)`: Runs inference on a dataset and returns output.
    -   `read_ground_truth_from_csv(csv_file)`: Loads ground truth labels for comparison.
-   **Workflow**:
    1.  Reads compiled model information from `compiled_jobs_batch.csv`.
    2.  Runs inference for each model using a test dataset and compares predictions with ground truth labels.
    3.  Profiles model execution time.
    4.  Logs results, including:
        -   Prediction accuracy (`model_scores_test.csv`).
        -   Per-image predictions (`prediction_results.csv`).