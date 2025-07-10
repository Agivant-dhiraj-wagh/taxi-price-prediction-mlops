# train.py
import os
import argparse
import joblib
import pandas as pd
from google.cloud import bigquery
from google.cloud import aiplatform
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from datetime import datetime

def load_data_from_bigquery(project_id, dataset_id, table_id):
    """Load data from BigQuery table."""
    client = bigquery.Client(project=project_id)
    
    query = f"""
    SELECT *
    FROM `{project_id}.{dataset_id}.{table_id}`
    WHERE fare_amount > 0
    AND trip_distance > 0
    LIMIT 100000
    """
    
    df = client.query(query).to_dataframe()
    return df

def preprocess_data(df):
    """Preprocess the taxi data."""
    # Feature engineering
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
    df['hour'] = df['pickup_datetime'].dt.hour
    df['day_of_week'] = df['pickup_datetime'].dt.dayofweek
    df['month'] = df['pickup_datetime'].dt.month
    
    # Calculate distance using Haversine formula
    def haversine_distance(lat1, lon1, lat2, lon2):
        R = 6371  # Earth's radius in km
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        return R * c
    
    df['calculated_distance'] = haversine_distance(
        df['pickup_latitude'], df['pickup_longitude'],
        df['dropoff_latitude'], df['dropoff_longitude']
    )
    
    # Select features
    features = ['trip_distance', 'calculated_distance', 'hour', 'day_of_week', 'month', 'passenger_count']
    X = df[features].fillna(0)
    y = df['fare_amount']
    
    return X, y

def train_model(X, y):
    """Train the Random Forest model."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model Performance:")
    print(f"MSE: {mse:.4f}")
    print(f"R2 Score: {r2:.4f}")
    print(f"RMSE: {np.sqrt(mse):.4f}")
    
    return model, {"mse": mse, "r2": r2, "rmse": np.sqrt(mse)}

def save_model_to_gcs(model, model_name, bucket_name):
    """Save model to Google Cloud Storage."""
    from google.cloud import storage
    
    # Save model locally first
    model_filename = f"{model_name}.joblib"
    joblib.dump(model, model_filename)
    
    # Upload to GCS
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(f"models/{model_filename}")
    blob.upload_from_filename(model_filename)
    
    model_uri = f"gs://{bucket_name}/models/{model_filename}"
    print(f"Model saved to: {model_uri}")
    return model_uri

def register_model_in_vertex_ai(model_uri, model_name, project_id, region):
    """Register model in Vertex AI Model Registry."""
    aiplatform.init(project=project_id, location=region)
    
    model = aiplatform.Model.upload(
        display_name=model_name,
        artifact_uri=model_uri,
        serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest",
        description=f"Taxi price prediction model trained on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    
    print(f"Model registered in Vertex AI: {model.resource_name}")
    return model

def main():
    parser = argparse.ArgumentParser(description='Train taxi price prediction model')
    parser.add_argument('--project_id', required=True, help='Google Cloud Project ID')
    parser.add_argument('--dataset_id', required=True, help='BigQuery dataset ID')
    parser.add_argument('--table_id', required=True, help='BigQuery table ID')
    parser.add_argument('--model_name', required=True, help='Model name')
    parser.add_argument('--region', default='us-central1', help='Google Cloud region')
    
    args = parser.parse_args()
    
    print("Starting training pipeline...")
    
    # Load data
    print("Loading data from BigQuery...")
    df = load_data_from_bigquery(args.project_id, args.dataset_id, args.table_id)
    print(f"Loaded {len(df)} records")
    
    # Preprocess data
    print("Preprocessing data...")
    X, y = preprocess_data(df)
    
    # Train model
    print("Training model...")
    model, metrics = train_model(X, y)
    
    # Save model to GCS
    bucket_name = "taxi-price-prediction-pipeline-artifacts"
    print("Saving model to GCS...")
    model_uri = save_model_to_gcs(model, args.model_name, bucket_name)
    
    # Register model in Vertex AI
    print("Registering model in Vertex AI...")
    vertex_model = register_model_in_vertex_ai(
        model_uri, args.model_name, args.project_id, args.region
    )
    
    print("Training pipeline completed successfully!")
    print(f"Model metrics: {metrics}")

if __name__ == "__main__":
    main()