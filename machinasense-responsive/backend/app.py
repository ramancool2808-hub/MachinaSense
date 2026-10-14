from __future__ import annotations
import io, json, os, uuid
from pathlib import Path
from typing import Optional
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / 'storage' / 'datasets'
MODEL_DIR = ROOT / 'storage' / 'models'
UPSTREAM_ARTIFACTS = ROOT / 'upstream' / 'backend' / 'artifacts'
for p in (DATA_DIR, MODEL_DIR): p.mkdir(parents=True, exist_ok=True)

app = FastAPI(title='MachinaSense Studio API', version='1.0.0', description='Responsive industrial ML workbench inspired by the MachinaSense architecture.')
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['GET','POST','DELETE'], allow_headers=['*'])
app.mount('/static', StaticFiles(directory=ROOT / 'frontend'), name='static')

class PredictRequest(BaseModel):
    values: dict[str, float]

class TrainRequest(BaseModel):
    target_column: str
    feature_columns: list[str]
    model_type: str = 'random_forest'
    test_size: float = 0.2


def safe_csv(raw: bytes) -> pd.DataFrame:
    if len(raw) > 25 * 1024 * 1024: raise HTTPException(413, 'CSV exceeds 25 MB limit.')
    try: df = pd.read_csv(io.BytesIO(raw))
    except Exception as e: raise HTTPException(400, f'Could not read CSV: {e}')
    if df.empty or len(df.columns) < 2: raise HTTPException(400, 'CSV must have a header and at least two columns.')
    df.columns = [str(c).strip() for c in df.columns]
    if df.columns.duplicated().any(): raise HTTPException(400, 'CSV contains duplicate column names.')
    return df


def get_active_model():
    path = MODEL_DIR / 'active_model.joblib'
    if not path.exists(): return None
    try: return joblib.load(path)
    except Exception: return None

@app.get('/')
def home(): return FileResponse(ROOT / 'frontend' / 'index.html')

@app.get('/api/health')
def health():
    model = get_active_model()
    metadata_path = UPSTREAM_ARTIFACTS / 'model_metadata.json'
    upstream_meta = None
    if metadata_path.exists():
        try: upstream_meta = json.loads(metadata_path.read_text(encoding='utf-8'))
        except Exception: pass
    return {
        'status':'online', 'service':'MachinaSense Studio API',
        'active_user_model': bool(model),
        'active_model_task': model.get('task') if model else None,
        'upstream_benchmark_metadata_found': bool(upstream_meta),
        'upstream_benchmark': upstream_meta.get('dataset') if upstream_meta else 'Not installed in this package',
        'note':'The upstream repo benchmark is NASA C-MAPSS turbofan RUL, not a bearing fault classifier.'
    }

@app.post('/api/dataset/preview')
async def preview(file: UploadFile = File(...)):
    if not (file.filename or '').lower().endswith('.csv'): raise HTTPException(400, 'Upload a .csv file.')
    df = safe_csv(await file.read())
    dataset_id = uuid.uuid4().hex[:12]
    path = DATA_DIR / f'{dataset_id}_{Path(file.filename).name}'
    df.to_csv(path, index=False)
    numeric = df.select_dtypes(include=np.number)
    return {
      'dataset_id':dataset_id, 'filename':Path(file.filename).name, 'rows':int(len(df)), 'columns':list(df.columns),
      'numeric_columns':list(numeric.columns), 'missing_cells':int(df.isna().sum().sum()),
      'duplicate_rows':int(df.duplicated().sum()), 'preview':df.head(8).astype(object).where(pd.notna(df.head(8)), None).to_dict(orient='records'),
      'summary':[{ 'column':str(c), 'mean':float(numeric[c].mean()) if numeric[c].notna().any() else None,
                   'std':float(numeric[c].std()) if numeric[c].notna().any() else None,
                   'min':float(numeric[c].min()) if numeric[c].notna().any() else None,
                   'max':float(numeric[c].max()) if numeric[c].notna().any() else None } for c in numeric.columns]
    }

@app.post('/api/train')
async def train(file: UploadFile = File(...), target_column: str = Form(...), feature_columns: str = Form(...), model_type: str = Form('random_forest'), test_size: float = Form(0.2)):
    if not (file.filename or '').lower().endswith('.csv'): raise HTTPException(400, 'Upload a .csv file.')
    df = safe_csv(await file.read())
    try: features = json.loads(feature_columns)
    except Exception: raise HTTPException(400, 'feature_columns must be a JSON list of column names.')
    if not isinstance(features, list) or not features: raise HTTPException(400, 'Select at least one feature column.')
    if target_column not in df.columns: raise HTTPException(400, f'Target column {target_column!r} not found.')
    if target_column in features: raise HTTPException(400, 'Target column cannot also be a feature.')
    missing = [c for c in features if c not in df.columns]
    if missing: raise HTTPException(400, f'Missing feature columns: {missing}')
    if not 0.05 <= test_size <= 0.5: raise HTTPException(400, 'Test split must be between 0.05 and 0.50.')
    frame = df[features + [target_column]].dropna(subset=[target_column]).copy()
    for c in features: frame[c] = pd.to_numeric(frame[c], errors='coerce')
    frame = frame.dropna(subset=features)
    if len(frame) < 10: raise HTTPException(400, 'Need at least 10 complete rows after cleaning.')
    X, raw_y = frame[features], frame[target_column].astype(str)
    classes = sorted(raw_y.unique().tolist())
    if len(classes) < 2: raise HTTPException(400, 'Target must contain at least two classes.')
    if len(classes) > max(10, len(frame)//3): raise HTTPException(400, 'Target appears too continuous/high-cardinality for classification. Choose a categorical label.')
    encoder = LabelEncoder(); y = encoder.fit_transform(raw_y)
    stratify = y if min(np.bincount(y)) >= 2 else None
    try: X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=stratify)
    except ValueError as e: raise HTTPException(400, f'Could not split dataset: {e}')
    if model_type not in {'random_forest','extra_trees'}: raise HTTPException(400, 'Supported classifiers: random_forest, extra_trees.')
    from sklearn.ensemble import ExtraTreesClassifier
    estimator = RandomForestClassifier(n_estimators=350, class_weight='balanced_subsample', random_state=42, n_jobs=-1) if model_type=='random_forest' else ExtraTreesClassifier(n_estimators=350, class_weight='balanced', random_state=42, n_jobs=-1)
    pipe = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scale', StandardScaler()), ('model', estimator)])
    pipe.fit(X_train, y_train); pred = pipe.predict(X_test)
    metrics = {'accuracy':round(float(accuracy_score(y_test,pred)),4),'f1_macro':round(float(f1_score(y_test,pred,average='macro',zero_division=0)),4),'precision_macro':round(float(precision_score(y_test,pred,average='macro',zero_division=0)),4),'recall_macro':round(float(recall_score(y_test,pred,average='macro',zero_division=0)),4),'train_rows':int(len(X_train)),'test_rows':int(len(X_test))}
    bundle = {'task':'condition_classification','model':pipe,'feature_columns':features,'target_column':target_column,'label_encoder':encoder,'classes':classes,'metrics':metrics,'model_type':model_type,'dataset_name':Path(file.filename).name}
    joblib.dump(bundle, MODEL_DIR/'active_model.joblib')
    (MODEL_DIR/'active_model.json').write_text(json.dumps({k:v for k,v in bundle.items() if k not in {'model','label_encoder'}}, indent=2, default=str), encoding='utf-8')
    return {'status':'trained','task':bundle['task'],'model_type':model_type,'features':features,'target_column':target_column,'classes':classes,'metrics':metrics,'message':'Model trained and saved as the active local model.'}

@app.post('/api/predict')
def predict(payload: PredictRequest):
    bundle = get_active_model()
    if not bundle: raise HTTPException(409, 'No active user-trained model yet. Upload a labelled CSV and train one first.')
    cols = bundle['feature_columns']; missing = [c for c in cols if c not in payload.values]
    if missing: raise HTTPException(400, f'Missing required features: {missing}')
    try: row = pd.DataFrame([[float(payload.values[c]) for c in cols]], columns=cols)
    except (ValueError,TypeError): raise HTTPException(400, 'All selected feature values must be numeric.')
    model = bundle['model']; encoded = int(model.predict(row)[0]); label = str(bundle['label_encoder'].inverse_transform([encoded])[0])
    probabilities = model.predict_proba(row)[0] if hasattr(model,'predict_proba') else None
    confidence = float(max(probabilities)) if probabilities is not None else None
    return {'prediction':label,'confidence':round(confidence,4) if confidence is not None else None,'probabilities':{str(bundle['label_encoder'].inverse_transform([i])[0]):round(float(p),4) for i,p in enumerate(probabilities)} if probabilities is not None else {},'model_type':bundle['model_type'],'features_used':cols,'warning':'Prediction quality depends on representative training data and independent validation.'}

@app.get('/api/model')
def model_info():
    b=get_active_model()
    if not b: return {'loaded':False,'task':None,'message':'No active model trained yet.'}
    return {'loaded':True,'task':b.get('task'),'model_type':b.get('model_type'),'features':b.get('feature_columns'),'target_column':b.get('target_column'),'classes':b.get('classes'),'metrics':b.get('metrics'),'dataset_name':b.get('dataset_name')}

@app.delete('/api/model')
def delete_model():
    p=MODEL_DIR/'active_model.joblib'; q=MODEL_DIR/'active_model.json'
    for f in (p,q):
        if f.exists(): f.unlink()
    return {'status':'cleared'}
