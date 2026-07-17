from fastapi import FastAPI
from netflow import NetFlow
import json

app = FastAPI()

'''
    model_json:
    '{ 
        "model_id": <str>,
        "model_dict": {<model_dict>}, 
        "in_out": [(in_feat, out_feat), ...], 
        "inits":(<init type: "he", "xavier", "zero", None>, ...)
    }'

'''

model_store = {}

@app.post("/modelify")
def model(model_json):
    model_info = json.loads(model_json)
    model_id = model_info.get("model_id", "default")
    net = NetFlow(model_info["model_dict"], model_info["in_out"], model_info["inits"])
    net.modelify()
    model_store[model_id] = net
    return {"status": "model created", "model_id": model_id}

@app.post("/train")
def train(training_json):
    training_info = json.loads(training_json)
    model_id = training_info.get("model_id", "default")
    net = model_store.get(model_id)
    if net is None:
        return {"error": f"no model with id {model_id}"}

    net.trainier(training_info["lr"], training_info["epochs"], training_info["training_data"])
    return {"status": "training complete"}
