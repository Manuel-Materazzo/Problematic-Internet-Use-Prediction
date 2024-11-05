import pickle
import json
import uvicorn
import pandas as pd
from pydoc import locate
from typing import List
from fastapi import FastAPI
from pydantic import BaseModel, create_model
from typing import Any, Dict

from src.pipelines.dt_pipeline import DTPipeline

model = pickle.load(open("../target/model.pkl", "rb"))
pipeline: DTPipeline = pickle.load(open("../target/pipeline.pkl", "rb"))

# Load configuration file
with open('../target/data-model.json', 'r') as f:
    config = json.load(f)


def create_pydantic_data_model(config: Dict[str, Any]) -> BaseModel:
    fields = config['fields']
    model = create_model(
        'InputData',
        **{name: (locate(type_name), ...) for name, type_name in fields.items()}
    )
    return model


# Generate the data model from config
InputData = create_pydantic_data_model(config)

app = FastAPI()


@app.post("/predict", response_model=List[float])
async def predict_post(datas: List[InputData]):
    dataframe = pd.DataFrame([data.dict() for data in datas])
    processed_data = pipeline.transform(dataframe)
    return model.predict(processed_data).tolist()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
