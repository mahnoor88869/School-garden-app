# models/

Included and ready:

| File | Used for |
|---|---|
| `supervised.joblib` | Decision tree (max depth 4), tab 1 |
| `clusters.joblib` | StandardScaler + KMeans (3 groups), tab 2 |
| `images.keras` | Small CNN (3 epochs), tab 3 |
| `model_info.json` | Tells the app which image model to prefer |

Optional upgrade: add your own `school_garden_final_cnn.keras` (the fine-tuned MobileNetV2 from Colab)
to this folder. The app then uses it automatically instead of `images.keras`.
