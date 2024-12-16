import qai_hub
import numpy as np
import pandas as pd
from scipy.special import softmax
import time

def read_compiled_jobs_from_csv(compiled_csv_file):
    """Read compiled model IDs from a CSV file."""
    compiled_data = pd.read_csv(compiled_csv_file)
    return [{"model_id": row["model_id"], "compiled_id": row["compiled_id"]} for _, row in compiled_data.iterrows()]

def run_profile(model, device):
    """Submit a profile job for the model."""
    profile_job = qai_hub.submit_profile_job(
        model=model,
        device=device,
        options="--max_profiler_iterations 20"
    )
    return profile_job.job_id

def run_inference(model, device, input_dataset):
    """Submit an inference job for the model."""
    inference_job = qai_hub.submit_inference_job(
        model=model,
        device=device,
        inputs=input_dataset,
        options="--max_profiler_iterations 1"
    )
    return inference_job.download_output_data()

def read_ground_truth_from_csv(csv_file):
    """Read ground truth class indices from a CSV file."""
    ground_truth_data = pd.read_csv(csv_file)
    return ground_truth_data['class_index'].tolist()

# Assign model and device for each model
device = qai_hub.Device("Samsung Galaxy S24 (Family)")

csv_filename = f'model_scores_test.csv' # csv scoring score and execution time
compiled_csv_filename = f'compiled_jobs_batch.csv'  # compiled models
ground_truth_csv = 'key.csv'  # ground truth
result_index_csv = f'prediction_results.csv'    # model results in index

ground_truth_indices = read_ground_truth_from_csv(ground_truth_csv)

pd.DataFrame(columns=["model_id", "score", "time"]).to_csv(csv_filename, index=False)
pd.DataFrame(columns=["image_index", "ground_truth_index", "predicted_index"]).to_csv(result_index_csv, index=False)

# COCO class names
coco_class_names = [
    "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "backpack", "umbrella", "handbag", "tie", "skis", "sports ball", "kite", "tennis racket", "bottle",
    "wine glass", "cup", "knife", "spoon", "bowl", "banana", "apple", "orange", "broccoli", "hot dog",
    "pizza", "donut", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "TV", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "teddy bear", "hair drier"
]

# Read from the compiled models CSV
compiled_jobs = read_compiled_jobs_from_csv(compiled_csv_filename)

# Iterate over each compiled model entry
for compiled_entry in compiled_jobs:
    model_id = compiled_entry["model_id"]
    compiled_id = compiled_entry["compiled_id"]
    job = qai_hub.get_job(compiled_id)

    compiled_model = job.get_target_model()
    print(f"Fetched compiled model {model_id}")

    # Profile the model
    execution_time = -1
    profile_id = run_profile(compiled_model, device)

    # Run inference
    print(f"Running inference for model {compiled_model.model_id} on device {device.name}")
    inference_output = run_inference(compiled_model, device, input_dataset)
    output_array = inference_output['output_0']
    print(len(inference_output['output_0']))

    if len(output_array) != len(ground_truth_indices):
        raise ValueError("Mismatch between inference results and ground truth length.")

    # Keep track of correct predictions
    correct = 0
    total = len(ground_truth_indices)

    # Compare predictions with ground truth
    for i, result in enumerate(output_array):
        softmax_results = softmax(result)  # (1, 64)

        # Get the top prediction (highest probability class index)
        top_prediction = np.argmax(softmax_results)

        # Compare with ground truth
        if top_prediction == ground_truth_indices[i]:
            correct += 1

        # Save prediction result for the current image
        image_result = {
            "image_index": i,
            "ground_truth_index": ground_truth_indices[i],
            "predicted_index": top_prediction
        }
        result_df = pd.DataFrame([image_result])
        result_df.to_csv(result_index_csv, mode='a', header=False, index=False)

    print(f"Correct predictions: {correct}/{total}")

    # Wait for profile to complete
    while True:
        profile_result = qai_hub.get_job_summaries(limit=1, offset=1)
        if profile_result[0].status.finished:
            execution_time = profile_result[0].estimated_inference_time
            print(execution_time)
            break
        time.sleep(3)

    # Log result for this model
    model_result = {"model_id": model_id, "score": (correct / total), "time": execution_time}

    # Append result to CSV after each model run
    result_df = pd.DataFrame([model_result])
    result_df.to_csv(csv_filename, mode='a', header=False, index=False)

print(f"Scores saved to '{csv_filename}'")