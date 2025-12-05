
# PLAYER_MARKET_PRICE_ESTIMATOR

End-to-end project to estimate football player market values using tree-based regression models. Suitable to upload as a GitHub project on your resume.

## What is included
- Full preprocessing, feature engineering, model training and evaluation pipeline in `src/`.
- A synthetic dataset in `data/raw/` so you can run the project end-to-end locally.
- Jupyter notebook `notebooks/EDA_and_Modeling.ipynb` with EDA, training walkthrough and SHAP explanations.
- Saved model example and scripts to evaluate predictions.
- Unit test stubs in `tests/` to show testable code structure.
- `requirements.txt`, `LICENSE` and `.gitignore`.

## Quickstart (run locally)
1. Clone or unzip repository.
2. Create virtual env and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```
3. Preprocess and split dataset:
```bash
python src/data_preprocessing.py --input data/raw/players_full.csv --output data/processed/players_processed.csv
```
4. Train models and save best model:
```bash
python src/train_models.py --input data/processed/players_processed_train.csv --out_dir models/
```
5. Evaluate:
```bash
python src/evaluate.py --model models/best_model.pkl --test data/processed/players_processed_test.csv
```

## GitHub upload notes
- Create a new repository on GitHub and follow the commands:
```bash
git init
git add .
git commit -m "Initial commit: Player Market Price Estimator"
git branch -M main
git remote add origin https://github.com/<your-username>/<repo-name>.git
git push -u origin main
```
Replace `<your-username>` and `<repo-name>` accordingly.
