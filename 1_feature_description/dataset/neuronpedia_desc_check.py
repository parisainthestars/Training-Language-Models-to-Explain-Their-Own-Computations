import os
from neuronpedia.np_sae_feature import SAEFeature
import json

os.environ["NEURONPEDIA_API_KEY"] = "YOUR_KEY"


feat = SAEFeature.get(
    model_id="llama3.1-8b",
    source="4-llamascope-res-131k",   # residual stream SAE at layer 9
    index="23756",                   # feature index as string
)   

# If it's a string, parse it to a dict
if isinstance(feat.jsonData, str):
    data = json.loads(feat.jsonData)
else:
    data = feat.jsonData   # already a dict

# Now extract the first explanation description
desc = data["explanations"][0]["description"].strip()
print(desc)
