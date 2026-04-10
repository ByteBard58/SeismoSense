from .schema.validation import UserInput
from fastapi import FastAPI, Request , Depends, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from fastapi.exceptions import HTTPException
from fastapi.staticfiles import StaticFiles
from typing import List, Tuple
import numpy as np
import pandas as pd
import shutil
import uuid
from sklearn.pipeline import Pipeline
import joblib
from contextlib import asynccontextmanager
from pathlib import Path
from models.fit import main

# paths to the pickle files (relative to project root)
MODEL_PATH = Path(__file__).parent.parent / "models" / "estimator.pkl"
NAMES_PATH = Path(__file__).parent.parent / "models" / "names.pkl"
# Directory for user-uploaded files
UPLOAD_DIR = Path("app","uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# if pickle files are not found, this helper will train the model
def ensure_models():
    print("🔍 Checking for existing model files...")
    if not MODEL_PATH.is_file() or not NAMES_PATH.is_file():
        print("⚠️ Model files not found. Starting training process...")
        main()
        print("✅ Model training complete. Pickle files created.")
    else:
        print("✅ Model files found. Skipping training.")
    
    print("📦 Loading model and feature names...")
    model = joblib.load(MODEL_PATH)
    names = joblib.load(NAMES_PATH)
    print("✅ Model and feature names loaded successfully.")
    return model, names

# helper for validating the user-uploaded .csv file
def validate_csv(payload:pd.DataFrame, expected_columns:List)-> pd.DataFrame:
    expected_columns = [r for r in expected_columns if r != "alert"]
    if payload.columns.tolist() != expected_columns:
        raise HTTPException(
            status_code=422,
            detail="Uploaded csv file does not match the expected column configuration"
        )
    
    try:
        df = payload.astype(float)
    except Exception:
        raise HTTPException(
            status_code=422,
            detail="Value in the uploaded csv file must be numeric (float-compatible)"
        )
    return df

@asynccontextmanager
async def lifespan(app:FastAPI):
    pipe,feat_names = ensure_models()

    app.state.pipe = pipe
    app.state.feat_names = feat_names

    yield

app = FastAPI(title="Seismosense",version="2.0(FastAPI)",lifespan=lifespan)

app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")

def get_things(request:Request) -> Tuple[Pipeline,np.ndarray]:
    return request.app.state.pipe, request.app.state.feat_names

@app.get("/")
def home():
    index_path = Path(__file__).parent / "templates" / "index.html"
    return FileResponse(index_path)

@app.get("/health",status_code= 200)
def health_check():
    msg = {
        "title":"Seismosense",
        "version":"2.0(FastAPI)",
        "status":"All systems operational"
    }
    return JSONResponse(content=msg,status_code=200)

@app.post("/predict",status_code=201)
def predict_things(value:UserInput,dep:Tuple[Pipeline,np.ndarray] = Depends(get_things)):
    pipe,feat_names = dep
    feat_names:List[str] = feat_names.tolist()
    value:dict = value.model_dump(mode="json")

    # Order check and running prediction
    user_inp:list[float] = []
    for i in feat_names:
        if i in ["alert"]:
            continue
        else:
            user_inp.append(value.get(i))
    
    user_inp = np.array(user_inp).reshape(1,-1)
    pred_label:np.ndarray = pipe.predict(user_inp)[0]
    pred_proba:np.ndarray = pipe.predict_proba(user_inp)[0]

    # Postprocessing
    label_map = {0: "green", 1: "orange", 2: "red", 3: "yellow"}
    pred_label:str = label_map.get(pred_label)
    pred_proba = {key:round(val,3) for key,val in zip(label_map.values(),pred_proba.tolist())}

    # Final Output
    msg = {
        "message": "prediction successful",
        "prediction": pred_label, 
        "probabilities": pred_proba
    }
    return JSONResponse(
        status_code=201, content=msg
    )

@app.post("/predict/batch",status_code=201)
def predict_things_in_batch(payload:UploadFile,dep:Tuple[Pipeline,np.ndarray] = Depends(get_things)):
    pipe,feat_names = dep
    feat_names:List[str] = feat_names.tolist() 

    # Receive and validate incoming upload
    valid_exts = [".csv"]
    extension = Path(payload.filename).suffix
    if extension not in valid_exts:
        raise HTTPException(
            status_code=422, detail=f"Only `.csv` files are accepted as input, got {extension} instead"
        )
    df_prev = pd.read_csv(payload.file)
    df: pd.DataFrame = validate_csv(df_prev,feat_names)

    # Run Predictions
    user_inp = df.to_numpy()
    pred_label:List[float] = pipe.predict(user_inp).tolist()
    pred_proba:List[List[float]] = pipe.predict_proba(user_inp).tolist()

    # Postprocessing
    label_map = {0: "green", 1: "orange", 2: "red", 3: "yellow"}
    pred_label:List[str] = [label_map.get(r) for r in pred_label]
    pred_proba = [[round(proba,3) for proba in sample] for sample in pred_proba]

    # Final Output
    msg = {
        "message": "batch prediction successful",
        "prediction": pred_label, 
        "probabilities": pred_proba
    }
    return JSONResponse(
        status_code=201, content=msg
    )