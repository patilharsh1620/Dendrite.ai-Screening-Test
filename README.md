
# Dendrite.ai Screening Test Submission

## Project Structure

```
├── dendrite_test.py
├── iris.csv
├── algoparams_from_ui.json
└── README.md
```

## How to Run

1. **Clone the repository**

```bash
git clone <your-repo-link>
cd <your-repo-folder>
```

2. **Install required packages**

Make sure you have Python 3.x installed.

```bash
pip install pandas numpy scikit-learn
```

3. **Prepare files**
- Ensure `iris.csv` and `algoparams_from_ui.json` are in the same directory as `dendrite_test.py`

4. **Run the script**

```bash
python dendrite_test.py
```

The script will:
- Load the dataset and configuration.
- Handle missing values.
- Generate new features.
- Reduce features.
- Train and tune the machine learning model(s).
- Print the model performance metrics (R² Score, MAE, RMSE).

---

## Important Notes
- **Generic Design**: The code can handle any JSON that follows the format provided.
- **Modular Structure**: Functions are separated for readability and maintainability.
- **Pipelines and GridSearchCV** are used for model building and hyperparameter tuning.

---

## Contact
If you have any questions about this submission, please feel free to reach out via the provided communication channels.

---

**Thank you!** ✨
