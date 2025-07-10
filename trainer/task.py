import argparse
import logging
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib  # For saving the model

logging.basicConfig(level=logging.INFO)

def train_model(bq_table_id: str, model_dir: str):
    """
    Fetches data from BigQuery, trains a RandomForestRegressor model,
    and saves the model to the specified directory.
    """
    logging.info(f"Starting model training with BigQuery table: {bq_table_id}")

    # Construct the BigQuery table path
    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCP_PROJECT")
    if not project_id:
        raise ValueError("GOOGLE_CLOUD_PROJECT or GCP_PROJECT environment variable not set.")

    full_bq_table_path = f"{project_id}.{bq_table_id}"

    # Query data from BigQuery
    logging.info(f"Fetching data from BigQuery: {full_bq_table_path}")
    df = pd.read_gbq(f"SELECT * FROM `{full_bq_table_path}`", project_id=project_id)

    if df.empty:
        raise ValueError("No data fetched from BigQuery. Check table ID and permissions.")

    logging.info(f"Fetched {len(df)} rows from BigQuery.")

    # Simple feature engineering (example)
    # Assume 'pickup_datetime' and 'dropoff_datetime' exist and 'fare_amount' is target
    # You'll need to adapt this to your actual dataset columns
    if 'pickup_datetime' in df.columns:
        df['pickup_hour'] = pd.to_datetime(df['pickup_datetime']).dt.hour
    if 'dropoff_datetime' in df.columns:
        df['dropoff_hour'] = pd.to_datetime(df['dropoff_datetime']).dt.hour

    # Define features and target
    features = [col for col in df.columns if col not in ['fare_amount', 'pickup_datetime', 'dropoff_datetime']]
    target = 'fare_amount'

    # Filter out non-numeric features if they somehow made it here (or handle them properly)
    numeric_features = df[features].select_dtypes(include=['number']).columns.tolist()
    X = df[numeric_features]
    y = df[target]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logging.info(f"Training data shape: {X_train.shape}")

    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    logging.info("Model training complete.")

    # Save model
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'model.joblib')
    joblib.dump(model, model_path)
    logging.info(f"Model saved to {model_path}")

    # Evaluate model (optional, but recommended)
    score = model.score(X_test, y_test)
    logging.info(f"Model R^2 score on test set: {score:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bq_table_id', type=str, required=True,
                        help='BigQuery table ID (e.g., your_dataset.your_table)')
    parser.add_argument('--model_dir', type=str, default=os.getenv('AIP_MODEL_DIR', '/tmp/model'),
                        help='Directory to save the trained model artifacts.')
    args = parser.parse_args()

    train_model(args.bq_table_id, args.model_dir)