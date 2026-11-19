# 🌱 School Garden Watering & Leaf Image Assistant (Project 04)

This is a classroom prototype for a teacher or student volunteer at a demonstration school garden. It is a five-tab Streamlit app:

| Tab | What it does | Model / method |
|---|---|---|
| 💧 Garden overview & watering review | Three charts. You check one record (a supplied example or your own readings) and get a **watering review flag**. | Decision tree (max depth 4) |
| 🧭 Condition profiles | Three groups of similar conditions, a profile table in original units, a scatter plot and the group of your record | StandardScaler + KMeans (k = 3) |
| 🍃 Bean leaf image | Compares a sample or uploaded leaf with 3 Beans dataset classes. A person then confirms or corrects the class. | CNN (MobileNetV2 transfer model) |
| 💬 Garden comments | Sentiment and topic for the 30 supplied comments, an evaluation against the reference labels, and a demo for one comment | NLTK VADER (offline) + keyword rules |
| 📋 Volunteer summary | An **Offline summary** built by a fixed Python template, plus downloads and prompt-practice material | No API, no key |

The app **only loads saved models**. Nothing is retrained when you click a button.

## Repository layout

```
app.py
requirements.txt
README.md
.gitignore
.streamlit/config.toml          # colour theme (optional)
data/        main.csv, feedback.csv, example_inputs.csv, meta.json,
             topic_keywords.json, prompt_test_cases.json
models/      supervised.joblib, clusters.joblib,
             school_garden_final_cnn.keras, images.keras, (model_info.json)
nltk_data/   offline VADER lexicon (supplied)
sample_images/   one example per leaf class (supplied)
source_notices/  licences and attribution (keep these)
```

`data/images.npz` (the full training arrays) is **not** needed in this repository.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy (free)

1. Create a **GitHub Free** repository. Upload the files with **Add file → Upload files**.
2. Go to **share.streamlit.io**, sign in with GitHub and choose **Create app**. Pick this repository, the branch `main` and the file `app.py`.
3. Under **Advanced settings**, choose **Python 3.12**. Then click **Deploy**.

## Limits (please read)

* The tabular readings are **generated classroom scenarios**. The rule "needs_water = 1 when next-morning moisture is below 25 %" is a teaching threshold, not real advice for watering.
* The KMeans group numbers are arbitrary. The groups describe conditions only and are not labels.
* The leaf class is a **dataset class comparison, not a diagnosis**. The model covers only 3 bean-leaf classes. Its score is not a calibrated probability, and an informed person must check every result. The app gives no pesticide or treatment advice.
* The comments are authored teaching examples. VADER handles English only and may miss sarcasm or mixed opinions.
* All results are demonstration results.

## Sources

* Beans images are from Makerere AI Lab via TensorFlow Datasets. See `source_notices/`.
* The VADER lexicon is from NLTK. See `source_notices/`.

Demo link: _add your Streamlit URL here_
