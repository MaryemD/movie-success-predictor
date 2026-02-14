# Quick Start Guide - Movie Success Predictor API

## ⚡ 5-Minute Setup

### Step 1: Install Dependencies
```bash
pip install -r requirements-api.txt
```

### Step 2: Test Installation
```bash
python test_api.py
```

Expected output: "✓ All tests passed!"

### Step 3: Start API Server
```bash
uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
```

### Step 4: Access API
Open browser to: **http://localhost:8000/docs**

You're done! The interactive API documentation is ready.

---

## 📝 Using the API

### Prediction Dashboard (Recommended)
Visit **http://localhost:8000/docs** for interactive testing on all endpoints

### Python Requests Example
```python
import requests

url = "http://localhost:8000/predict/classification"

data = {
    "dataset": "all",
    "return_probabilities": True,
    "movies": [
        {
            "budget": 100000000,
            "popularity": 50.0,
            "runtime": 120,
            "release_year": 2020,
            "release_month": 5,
            "vote_average": 7.2,
            "vote_count": 1500,
            "is_collection": 0,
            "is_english": 1,
            "num_genres": 2,
            "num_production_companies": 3,
            "num_production_countries": 1,
            "num_spoken_languages": 1,
            "num_cast": 50,
            "num_crew": 120,
            "num_keywords": 15,
            "has_top_director": 1,
            "has_top_actor": 1,
            "has_top_lead_actor": 1,
            "primary_genre_Adventure": 1,
            "primary_genre_Animation": 0,
            "primary_genre_Comedy": 0,
            "primary_genre_Crime": 0,
            "primary_genre_Documentary": 0,
            "primary_genre_Drama": 0,
            "primary_genre_Family": 0,
            "primary_genre_Fantasy": 0,
            "primary_genre_Foreign": 0,
            "primary_genre_History": 0,
            "primary_genre_Horror": 0,
            "primary_genre_Music": 0,
            "primary_genre_Mystery": 0,
            "primary_genre_Romance": 0,
            "primary_genre_Science Fiction": 0,
            "primary_genre_TV Movie": 0,
            "primary_genre_Thriller": 0,
            "primary_genre_Unknown": 0,
            "primary_genre_War": 0,
            "primary_genre_Western": 0
        }
    ]
}

response = requests.post(url, json=data)
result = response.json()

print(f"Success prediction: {result['predictions']}")
print(f"Probability: {result['probabilities']}")
```

### cURL Example
```bash
curl -X POST "http://localhost:8000/predict/classification" \
  -H "Content-Type: application/json" \
  -d '{
    "dataset": "all",
    "movies": [
      {
        "budget": 100000000,
        "popularity": 50.0,
        "runtime": 120,
        "release_year": 2020,
        "release_month": 5,
        "vote_average": 7.2,
        "vote_count": 1500,
        "is_collection": 0,
        "is_english": 1,
        "num_genres": 2,
        "num_production_companies": 3,
        "num_production_countries": 1,
        "num_spoken_languages": 1,
        "num_cast": 50,
        "num_crew": 120,
        "num_keywords": 15,
        "has_top_director": 1,
        "has_top_actor": 1,
        "has_top_lead_actor": 1,
        "primary_genre_Adventure": 1,
        "primary_genre_Animation": 0,
        "primary_genre_Comedy": 0,
        "primary_genre_Crime": 0,
        "primary_genre_Documentary": 0,
        "primary_genre_Drama": 0,
        "primary_genre_Family": 0,
        "primary_genre_Fantasy": 0,
        "primary_genre_Foreign": 0,
        "primary_genre_History": 0,
        "primary_genre_Horror": 0,
        "primary_genre_Music": 0,
        "primary_genre_Mystery": 0,
        "primary_genre_Romance": 0,
        "primary_genre_Science Fiction": 0,
        "primary_genre_TV Movie": 0,
        "primary_genre_Thriller": 0,
        "primary_genre_Unknown": 0,
        "primary_genre_War": 0,
        "primary_genre_Western": 0
      }
    ]
  }'
```

---

## 📚 Key Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | API info |
| `/health` | GET | Server status |
| `/info` | GET | Feature requirements |
| `/predict/classification` | POST | Success/failure |
| `/predict/regression` | POST | Revenue estimate |
| `/predict/combined` | POST | Both predictions |

---

## 🎯 Different Dataset Configurations

```python
# Metadata only (fewer features required)
response = requests.post(url, json={
    "dataset": "metadata",  # ← use this
    "movies": [...]
})

# With cast/crew information
response = requests.post(url, json={
    "dataset": "meta_credits",  # ← or this
    "movies": [...]
})

# With keywords/tags
response = requests.post(url, json={
    "dataset": "meta_keywords",  # ← or this
    "movies": [...]
})

# Most complete (requires all features)
response = requests.post(url, json={
    "dataset": "all",  # ← or this
    "movies": [...]
})
```

---

## 🔍 Understanding Responses

### Classification Response
```json
{
  "predictions": [0, 1, 1],           // 0=fail, 1=success
  "probabilities": [0.23, 0.78, 0.91],  // Success probability
  "threshold_used": 0.51               // Decision threshold
}
```

### Regression Response  
```json
{
  "predictions": [                    // Estimated values
    185420000.53,
    -45000000.12,
    230567890.33
  ]
}
```

---

## ❌ Troubleshooting

### "Models not found"
- Verify `models/` folder exists with `.pkl` files
- Ensure you're running from project root directory

### "Missing required features"
- Call `/info` endpoint to see required features
- Ensure your DataFrame has all columns
- Check column names match exactly

### "Port 8000 already in use"
```bash
# Use different port
uvicorn src.api:app --port 8001
```

### "ModuleNotFoundError"
```bash
# Reinstall dependencies
pip install -r requirements-api.txt --force-reinstall
```

---

## 📊 Direct Python Usage (Without API)

```python
from src.models.predict_model import PredictionPipeline
import pandas as pd

# Initialize
pipeline = PredictionPipeline(
    model_dir='./models',
    data_dir='./data/processed'
)

# Prepare data
X = pd.DataFrame({
    'feature1': [100, 200],
    'feature2': [50, 75],
    # ... all required features
})

# Classify
clf_results = pipeline.predict_classification(X, dataset='all')
print(f"Predictions: {clf_results['predictions']}")

# Regress
reg_results = pipeline.predict_regression(X, dataset='all')
print(f"Estimates: {reg_results['predictions']}")

# Combined
combined = pipeline.predict_combined(X, dataset='all')
```

---

## 📋 Dataset Configuration Comparison

| Config | Features | Features Needed | Speed | Use When |
|--------|----------|-----------------|-------|----------|
| `metadata` | Basic | ~10 | Fast | Quick predictions, uncertain data |
| `meta_credits` | +Cast/Crew | ~20 | Medium | Have actor info |
| `meta_keywords` | +Tags | ~25 | Medium | Have keywords |
| `all` | Complete | ~40+ | Slower | Maximum accuracy desired |

---

## 🚀 Next Steps

1. **Test endpoints** at http://localhost:8000/docs
2. **Check requirements** at http://localhost:8000/info
3. **Make predictions** using your data
4. **Monitor health** at http://localhost:8000/health
5. **Scale up** for batch processing

---

## 📖 Reference Documentation

- **Full API Guide:** `API_README.md`
- **Implementation Details:** `IMPLEMENTATION_SUMMARY.md`
- **Direct Python Usage:** `src/example_usage.py`
- **Run Tests:** `python test_api.py`

---

## ⚙️ Configuration

### Change Model Directory
```python
pipeline = PredictionPipeline(
    model_dir='/path/to/models',
    data_dir='/path/to/data'
)
```

### Change API Port
```bash
uvicorn src.api:app --port 5000
```

### Enable CORS for Specific Origins
Edit `src/api.py`:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://myapp.com"],  # ← specify
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## 📞 Common Questions

**Q: Can I use different models?**
A: Yes, place new models in `models/` folder with naming convention `best_clf_model_{dataset}.pkl`

**Q: What's the batch size limit?**
A: Maximum 1000 records per request

**Q: Do I need all features for all datasets?**
A: Each dataset requires only its own feature set. Use `/info` endpoint to check.

**Q: Can I run this in production?**
A: Yes, use production ASGI server like Gunicorn:
```bash
gunicorn -w 4 -k uvicorn.workers.UvicornWorker src.api:app
```

**Q: How fast are predictions?**
A: ~10-50ms per batch depending on dataset configuration

---

## 🎉 You're All Set!

Visit **http://localhost:8000/docs** to start making predictions!

Need help? Check `API_README.md` or run `python test_api.py` for diagnostics.
