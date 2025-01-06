import pickle
import json
import uvicorn
import pandas as pd
from pydoc import locate
from typing import List, Type
from fastapi import FastAPI
from pydantic import create_model
from typing import Any, Dict

from pydantic.main import ModelT

from src.pipelines.dt_pipeline import DTPipeline
from src.preprocessors.data_preprocessor import DataPreprocessor

model = pickle.load(open("../target/model.pkl", "rb"))
preprocessor: DataPreprocessor = pickle.load(open("../target/preprocessor.pkl", "rb"))
pipeline: DTPipeline = pickle.load(open("../target/pipeline.pkl", "rb"))

# Load configuration file
with open('../target/data-model.json', 'r') as f:
    config = json.load(f)


def calculate_sii(pciat_scores, weight):
    # define a utility method to calculate sii
    pciat_scores = pciat_scores * weight
    bins = pd.cut(pciat_scores, bins=[-float('inf'), 30, 50, 80, float('inf')], labels=[0, 1, 2, 3], right=False)
    return bins.astype(int)


def create_pydantic_data_model(model_config: Dict[str, Any]) -> Type[ModelT]:
    fields = model_config['fields']
    model_type = create_model(
        'InputData',
        **{name: (locate(type_name), ...) for name, type_name in fields.items()}
    )
    return model_type


# Generate the data model from config
InputData = create_pydantic_data_model(config)

app = FastAPI()


@app.post("/predict", response_model=List[float])
async def predict_post(datas: List[InputData]):
    dataframe = pd.DataFrame([data.dict() for data in datas])
    preprocessor.preprocess_data(dataframe)
    processed_data = pipeline.transform(dataframe)
    # predict pciat score
    pciat_preds = model.predict(processed_data).tolist()
    # calculate sii
    return calculate_sii(pciat_preds, 1.2525)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
