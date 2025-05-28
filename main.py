from fastapi import FastAPI, Request
from transformers import BertTokenizer, BertForSequenceClassification
import tarfile
import os
import requests
import torch
import joblib
import uvicorn

app = FastAPI()

# Hugging Face model URL (your username is ap6617)
model_url = "https://huggingface.co/ap6617/bert-financial-tagger/resolve/main/bert_model.tar.gz"
model_dir = "bert_model"
tar_path = "bert_model.tar.gz"

# Step 1: Download and extract model
if not os.path.exists(model_dir):
    print("Downloading model...")
    with requests.get(model_url, stream=True) as r:
        r.raise_for_status()
        with open(tar_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    print("Extracting model...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall()
    os.remove(tar_path)

# Step 2: Load model and tokenizer
model = BertForSequenceClassification.from_pretrained(model_dir)
tokenizer = BertTokenizer.from_pretrained(model_dir)

# Step 3: Load label binarizer
binarizer_url = "https://huggingface.co/ap6617/bert-financial-tagger/resolve/main/bert_label_binarizer.pkl"
binarizer_file = "bert_label_binarizer.pkl"
if not os.path.exists(binarizer_file):
    print("Downloading label binarizer...")
    r = requests.get(binarizer_url)
    with open(binarizer_file, "wb") as f:
        f.write(r.content)
mlb = joblib.load(binarizer_file)

# API route
@app.post("/predict")
async def predict(request: Request):
    data = await request.json()
    text = data.get("text", "")
    if not text:
        return {"error": "No text provided"}
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.sigmoid(outputs.logits).cpu().numpy()
        tags = mlb.inverse_transform(probs > 0.5)
    return {"tags": tags[0] if tags else []}

# For Render's port
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
