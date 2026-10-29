"""Optional local inference API. Deploy behind TLS, authentication and request limits."""
from contextlib import asynccontextmanager
from typing import Literal
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from pathdsp.inference import model_resource, dataset_resource, predict_pairs

@asynccontextmanager
async def lifespan(app):
    model_resource()
    for name in ('GDSC', 'CCLE'):
        dataset_resource(name)
    yield

app = FastAPI(title='PathDSP Research Inference', version='1.0.0', lifespan=lifespan)

class Pair(BaseModel):
    model_config = ConfigDict(extra="forbid")
    drug: str = Field(min_length=1, max_length=128)
    cell: str = Field(min_length=1, max_length=128)

class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset: Literal['GDSC', 'CCLE'] = 'GDSC'
    pairs: list[Pair] = Field(min_length=1, max_length=256)

@app.get('/health/ready')
def ready():
    try:
        _, _, version = model_resource()
        for name in ('GDSC', 'CCLE'):
            bundle = dataset_resource(name)
            predict_pairs(name, [(bundle.drug_ids[0], bundle.cell_ids[0])])
    except (ValueError, RuntimeError, OSError) as exc:
        raise HTTPException(503, 'Model or bundled data failed readiness validation.') from exc
    return {'status': 'ready', 'model_version': version, 'purpose': 'research only'}

@app.post('/v1/predict')
def predict(request: PredictionRequest):
    try:
        return predict_pairs(request.dataset, [(pair.drug, pair.cell) for pair in request.pairs])
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(503, 'Inference unavailable; retry shortly.') from exc
