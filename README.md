# Smart Crop Advisor – Using Genetic Algorithm and Machine Learning

## Domain
Machine Learning + Precision Agriculture

## Core ML Approach
Genetic Algorithm (GA) + Random Forest (RF)

---

## Module 1: Dataset Management

### Purpose
Loads and validates the agricultural dataset (`Crop_recommendation.csv`) before preprocessing. Ensures structural integrity, column presence, data types, missing values, duplicates, and target class distribution without altering raw data.

### Input
`dataset/Crop_recommendation.csv`

### Input Features (7)
- `N` (Nitrogen)
- `P` (Phosphorus)
- `K` (Potassium)
- `temperature` (Temperature)
- `humidity` (Humidity)
- `ph` (Soil pH)
- `rainfall` (Rainfall)

### Target Column
- `label` (Crop name)

### Execution
```bash
python module1/dataset_manager.py
```

---

## Module 2: Data Preprocessing

### Purpose
Converts the validated raw dataset into a clean, verified processed dataset (`dataset/processed/crop_data_clean.csv`) without data leakage or unnecessary numerical scaling.

### Inputs & Outputs
- **Input Raw Dataset**: `dataset/Crop_recommendation.csv`
- **Output Processed Dataset**: `dataset/processed/crop_data_clean.csv`
- **Preprocessing Report**: `results/module2_preprocessing_report.txt`

### Execution
```bash
python module2/preprocessing.py
```

---

## Module 3: Feature Analysis

### Purpose
Performs Exploratory Data Analysis (EDA) on the preprocessed agricultural dataset (`dataset/processed/crop_data_clean.csv`) to analyze statistical distributions, correlations, crop-wise feature ranges, and exploratory mutual information.

### Inputs & Outputs
- **Input Dataset**: `dataset/processed/crop_data_clean.csv`
- **Results Directory**: `results/module3/`
- **Plots Directory**: `results/module3/plots/`

### Execution
```bash
python module3/feature_analysis.py
```

---

## Module 4: Train / Validation Split

### Purpose
Separates the preprocessed agricultural dataset into an 80% Training set and a 20% Validation set using stratified splitting (`random_state=42`). Preserves exact crop class proportions across splits without performing feature scaling, feature selection, or target label encoding.

### Inputs & Outputs
- **Input Dataset**: `dataset/processed/crop_data_clean.csv`
- **Split CSVs**: `dataset/splits/`
- **Results & Diagnostics**: `results/module4/`

### Execution
```bash
python module4/train_validation_split.py
```

---

## Module 5: Baseline Random Forest

### Purpose
Establishes an unoptimized reference Random Forest classifier trained on Module 4 training data (`X_train.csv`, `y_train.csv`) and evaluated on validation data (`X_validation.csv`, `y_validation.csv`). Provides reference accuracy, precision, recall, F1, confusion matrix, feature importances, and serialized model artifacts for comparison against subsequent Genetic Algorithm optimization.

### Inputs & Outputs
- **Inputs**: Split CSVs from `dataset/splits/`
- **Model Artifacts**: `models/`
  - `baseline_rf_model.pkl`
  - `crop_label_encoder.pkl`
  - `baseline_model_metadata.json`
- **Results & Diagnostics**: `results/module5/`
  - `baseline_metrics.csv`
  - `baseline_predictions.csv`
  - `classification_report.csv`
  - `baseline_feature_importance.csv`
  - `baseline_summary.json`
  - `baseline_random_forest_report.txt`
- **Plots**: `results/module5/plots/`
  - `baseline_confusion_matrix.png`
  - `baseline_feature_importance.png`
  - `train_validation_accuracy.png`
  - `train_validation_f1.png`

### Baseline Hyperparameters
- `n_estimators`: `100`
- `criterion`: `"gini"`
- `random_state`: `42`
- `n_jobs`: `-1`

### Execution
```bash
python module5/baseline_random_forest.py
```

### Expected Output
- Complete console execution summary with `PASS` status.
- Trained model (`.pkl`) and label encoder (`.pkl`) serialized in `models/`.
- Detailed prediction CSVs, metrics, and classification reports saved to `results/module5/`.
- 4 high-resolution PNG plots generated in `results/module5/plots/`.
- Text report generated at `results/module5/baseline_random_forest_report.txt`.
