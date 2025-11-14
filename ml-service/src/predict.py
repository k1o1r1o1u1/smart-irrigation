#!/usr/bin/env python3
"""
predict.py

Loads model artifact saved by trainA.py (a joblib dict containing keys: "model", "features", ...).
Maps CLI inputs --soil, --temp, --hum to the model's feature names and calls model.predict()
using a pandas.DataFrame so feature names match exactly (no sklearn warning).

Can also fetch sensor data from MongoDB irrigation_db.sensordatas collection.
"""

import argparse
from pathlib import Path
import joblib
import sys
import pandas as pd
from pymongo import MongoClient
from datetime import datetime

# keywords to match model feature column names
_KEYWORDS = {
    "soil": ["soil", "moist"],
    "temp": ["temp", "temperature"],
    "hum":  ["hum", "humid", "humidity"]
}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="path to joblib .pkl produced by trainA.py")
    p.add_argument("--soil", type=float, default=None, help="soil moisture value (numeric)")
    p.add_argument("--temp", type=float, default=None, help="air temperature (numeric)")
    p.add_argument("--hum", type=float, default=None, help="air humidity (numeric)")
    p.add_argument("--from-db", action="store_true", help="fetch sensor values from MongoDB irrigation_db.sensordatas")
    p.add_argument("--device-id", type=str, default=None, help="filter by device ID when using --from-db")
    p.add_argument("--mongo-uri", type=str, default="mongodb://127.0.0.1:27017", help="MongoDB connection URI")
    p.add_argument("--raw", action="store_true", help="print raw 0/1 instead of human text")
    return p.parse_args()

def fetch_latest_sensor_data(mongo_uri, device_id=None):
    """
    Fetch the latest sensor reading from irrigation_db.sensordatas collection.
    Returns dict with keys: moisture, temperature, humidity, deviceId, timestamp
    """
    try:
        client = MongoClient(mongo_uri)
        db = client["irrigation_db"]
        collection = db["sensordatas"]
        
        # Build filter
        filter_query = {}
        if device_id:
            filter_query["deviceId"] = device_id
        
        # Find latest document sorted by timestamp descending
        latest = collection.find_one(filter_query, sort=[("timestamp", -1)])
        
        if not latest:
            raise ValueError(f"No sensor data found{' for device ' + device_id if device_id else ''}")
        
        return {
            "moisture": latest.get("moisture"),
            "temperature": latest.get("temperature"),
            "humidity": latest.get("humidity"),
            "deviceId": latest.get("deviceId"),
            "timestamp": latest.get("timestamp")
        }
    except Exception as e:
        raise RuntimeError(f"Failed to fetch from MongoDB: {e}")

def save_prediction_to_db(mongo_uri, device_id, pump_status, water_mm=10, pump_time_sec=20):
    """
    Save prediction result to irrigation_db.predictions collection.
    
    Args:
        mongo_uri: MongoDB connection URI
        device_id: Device ID from sensor data (or None)
        pump_status: 1 for ON, 0 for OFF
        water_mm: Dummy water amount in mm (default: 10)
        pump_time_sec: Dummy pump time in seconds (default: 20)
    """
    try:
        client = MongoClient(mongo_uri)
        db = client["irrigation_db"]
        collection = db["predictions"]
        
        # Generate predictionId as ISO timestamp
        prediction_id = datetime.utcnow().isoformat() + "Z"
        
        # Set used to true when pump needs to be turned OFF, false when pump needs to be ON
        # If pump is OFF, prediction is "used" (no action needed)
        # If pump is ON, prediction is not "used" yet (action pending)
        used = (pump_status == 0)
        
        prediction_doc = {
            "deviceId": device_id or "unknown",
            "waterMM": water_mm,
            "pumpTimeSec": pump_time_sec,
            "predictionId": prediction_id,
            "used": used
        }
        
        result = collection.insert_one(prediction_doc)
        return result.inserted_id
    except Exception as e:
        raise RuntimeError(f"Failed to save prediction to MongoDB: {e}")

def prompt_float(name):
    while True:
        try:
            v = input(f"Enter {name}: ").strip()
            return float(v)
        except Exception:
            print("Please enter a numeric value.")

def load_artifact(path: Path):
    obj = joblib.load(str(path))
    if isinstance(obj, dict):
        model = obj.get("model") or obj.get("pipeline") or None
        features = obj.get("features") or obj.get("feature_names") or None
        metadata = obj.get("metadata", {})
        # if model not found inside dict, attempt to find any estimator inside
        if model is None:
            for v in obj.values():
                if hasattr(v, "predict"):
                    model = v
                    break
        return model, features, metadata
    if hasattr(obj, "predict"):
        return obj, None, {}
    raise RuntimeError("Model artifact does not contain a sklearn model or expected dict.")

def map_inputs_to_features(features, provided):
    """
    Build a row dict keyed by each feature name expected by the model.
    Matching is done using substring keywords. If no mapping found, prompt user.
    """
    row = {}
    for feat in features:
        fl = feat.lower()
        assigned = False
        for key, kws in _KEYWORDS.items():
            if any(k in fl for k in kws):
                val = provided.get(key)
                if val is None:
                    # prompt
                    val = prompt_float(key)
                row[feat] = float(val)
                assigned = True
                break
        if not assigned:
            # if feature name exactly matches one of soil/temp/hum keys use that
            for key in ("soil","temp","hum"):
                if key == fl and provided.get(key) is not None:
                    row[feat] = float(provided[key])
                    assigned = True
                    break
        if not assigned:
            # last resort: prompt user for this feature
            row[feat] = prompt_float(feat)
    return row

def main():
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.exists():
        print("Model file not found:", model_path)
        sys.exit(1)

    model, features, metadata = load_artifact(model_path)
    if model is None:
        print("No model found inside artifact.")
        sys.exit(1)

    # Track device_id for saving prediction
    device_id = None
    
    # Fetch from database if requested
    if args.from_db:
        try:
            print(f"Fetching latest sensor data from MongoDB...")
            sensor_data = fetch_latest_sensor_data(args.mongo_uri, args.device_id)
            provided = {
                "soil": sensor_data["moisture"],
                "temp": sensor_data["temperature"],
                "hum": sensor_data["humidity"]
            }
            device_id = sensor_data.get("deviceId")
            print(f"Using sensor data from device: {device_id or 'N/A'}")
            print(f"  Moisture: {provided['soil']}%")
            print(f"  Temperature: {provided['temp']}°C")
            print(f"  Humidity: {provided['hum']}%")
        except Exception as e:
            print(f"Error fetching from database: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        provided = {"soil": args.soil, "temp": args.temp, "hum": args.hum}

    if features:
        # ensure features is a plain list of strings
        features = list(features)
        # build a single-row DataFrame with the same column names and order
        row = map_inputs_to_features(features, provided)
        X = pd.DataFrame([row], columns=features)
    else:
        # model has no feature list saved; assume order soil,temp,hum
        for k in ("soil","temp","hum"):
            if provided[k] is None:
                provided[k] = prompt_float(k)
        X = pd.DataFrame([[provided["soil"], provided["temp"], provided["hum"]]],
                         columns=["soil","temp","hum"])

    # call predict
    try:
        yhat = model.predict(X)
    except Exception as e:
        # try passing numpy array as fallback
        try:
            yhat = model.predict(X.values)
        except Exception as e2:
            print("Failed to call model.predict():", e, e2)
            sys.exit(1)

    out = int(yhat[0])
    if args.raw:
        print(out)
    else:
        print("PUMP: ON" if out == 1 else "PUMP: OFF")
    
    # Save prediction to MongoDB
    try:
        print(f"\nSaving prediction to MongoDB...")
        inserted_id = save_prediction_to_db(args.mongo_uri, device_id, out)
        print(f"Prediction saved with ID: {inserted_id}")
    except Exception as e:
        print(f"Warning: Failed to save prediction to database: {e}", file=sys.stderr)
        # Don't exit on save failure, prediction was successful

if __name__ == "__main__":
    main()
