# Deploying to IBM Watson Machine Learning

These are the steps to host `models/best_model.pkl` on IBM Watson Machine
Learning (WML) so predictions can be served from the cloud instead of a
local Flask process.

## 1. Prerequisites

- An IBM Cloud account with a **Watson Machine Learning** service instance
- A **Cloud Object Storage** instance (WML uses this to stage model
  artifacts)
- The `ibm-watson-machine-learning` Python package:

```bash
pip install ibm-watson-machine-learning
```

## 2. Authenticate

```python
from ibm_watson_machine_learning import APIClient

wml_credentials = {
    "url": "https://<region>.ml.cloud.ibm.com",
    "apikey": "<YOUR_IBM_CLOUD_API_KEY>",
}
client = APIClient(wml_credentials)
client.set.default_space("<YOUR_DEPLOYMENT_SPACE_ID>")
```

## 3. Package and store the model

Watson ML expects a scikit-learn-compatible model plus metadata describing
the software runtime it should run in.

```python
import pickle

with open("../models/best_model.pkl", "rb") as f:
    model = pickle.load(f)

sofware_spec_uid = client.software_specifications.get_id_by_name("runtime-23.1-py3.10")

metadata = {
    client.repository.ModelMetaNames.NAME: "credit-card-approval-model",
    client.repository.ModelMetaNames.TYPE: "scikit-learn_1.1",
    client.repository.ModelMetaNames.SOFTWARE_SPEC_UID: sofware_spec_uid,
}

stored_model = client.repository.store_model(
    model=model, meta_props=metadata
)
model_uid = client.repository.get_model_id(stored_model)
```

## 4. Create an online deployment

```python
deployment_metadata = {
    client.deployments.ConfigurationMetaNames.NAME: "credit-card-approval-deployment",
    client.deployments.ConfigurationMetaNames.ONLINE: {},
}

deployment = client.deployments.create(model_uid, meta_props=deployment_metadata)
deployment_uid = client.deployments.get_id(deployment)
```

## 5. Score (predict) against the endpoint

```python
scoring_payload = {
    client.deployments.ScoringMetaNames.INPUT_DATA: [
        {
            "fields": feature_columns,   # from models/feature_columns.pkl
            "values": [row_of_feature_values],
        }
    ]
}

result = client.deployments.score(deployment_uid, scoring_payload)
print(result)
```

## 6. Point the Flask app at the cloud endpoint (optional)

Instead of loading `best_model.pkl` locally, `app/app.py` can be modified
to POST the same feature row to the WML scoring URL
(`client.deployments.get_scoring_href(deployment)`) and parse the JSON
response — everything downstream (the form, the templates, the confidence
gauge) stays the same.

This keeps the model itself swappable: retrain locally, push a new version
to Watson ML, and the web app automatically serves the latest deployed
model without a code change.
