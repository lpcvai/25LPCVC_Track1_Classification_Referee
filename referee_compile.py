import qai_hub
import pandas as pd
from datetime import datetime

def compile_model(model, device, input_shape):
    """Submit a compile job for the model."""
    compile_job = qai_hub.submit_compile_job(
        model=model,
        device=device,
        input_specs={"image": input_shape},
        options="--target_runtime tflite"
    )
    return compile_job

def fetch_model(model_id):
    """Fetch a model by its ID."""
    return qai_hub.get_model(model_id)

def read_model_ids_from_csv(csv_file):
    """Read model IDs from a CSV file."""
    model_data = pd.read_csv(csv_file)
    return model_data['model_id'].tolist()

device = qai_hub.Device("Samsung Galaxy S24 (Family)")

# Read model IDs from CSV
model_csv = 'test_models.csv'  # Path to CSV file for models
model_ids = read_model_ids_from_csv(model_csv)

input_shape = (1, 3, 224, 224)

# Fetch models
models = [fetch_model(model_id) for model_id in model_ids]
print(f"Fetched {len(models)} models")

# Get current time for the CSV filename
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_filename = f'model_scores_{current_time}.csv'
compiled_csv_filename = f'compiled_jobs_batch.csv'

# Create an empty CSV with headers
pd.DataFrame(columns=["model_id", "compiled_id"]).to_csv(compiled_csv_filename, index=False)

# Iterate over each model
for model in models:
    print(f"\nCompiling model {model.model_id} for device {device.name}")

    # Compile the model
    compile_job = compile_model(model, device, input_shape)
    compile_job.modify_sharing(add_emails=['lowpowervision@gmail.com'])
    compile_job.set_name(f'{model.model_id}_LPCVC25')
    compiled_id = compile_job.job_id
    print(f"Model {model.model_id} compiled successfully with ID {compiled_id}")

    compiled_model_result = {"model_id": model.model_id, "compiled_id": compiled_id}
    compiled_df = pd.DataFrame([compiled_model_result])
    compiled_df.to_csv(compiled_csv_filename, mode='a', header=False, index=False)