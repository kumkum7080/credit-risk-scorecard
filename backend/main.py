import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score

from scorecard import CustomWoEBinning, ScorecardScaler, compute_applicant_score
from decision_engine import calculate_ks_statistic, calculate_expected_loss, optimize_decision_cutoff

app = FastAPI(title="Advanced Credit Default Scoring & Decision Engine Backend")

# Serve frontend static assets directly from root
FRONTEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))

@app.get("/", response_class=HTMLResponse)
def read_root():
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    with open(index_path, "r", encoding="utf-8") as f:
        return f.read()

@app.get("/style.css")
def read_style():
    return FileResponse(os.path.join(FRONTEND_DIR, "style.css"))

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_PATH = os.path.join(os.path.dirname(__file__), "mock_credit_data.csv")

# Global models and parameters
binning_model = None
logistic_model = None
scorecard_points = None
optimal_pd_cutoff = 0.15
optimal_score_cutoff = 600
score_scaler = None
model_metrics = {}

class ApplicantData(BaseModel):
    age: float
    income: float
    loan_amount: float
    emp_length: float
    home_ownership: str
    purpose: str
    credit_history_length: float
    historical_default: int

def generate_mock_credit_data():
    """
    Generates a realistic credit default dataset of 1,500 bank loan applicants.
    """
    print("Generating mock credit default data...")
    np.random.seed(42)
    n_samples = 1500
    
    age = np.random.randint(20, 66, n_samples)
    income = np.random.uniform(20000, 150000, n_samples)
    loan_amount = np.random.uniform(2000, 45000, n_samples)
    emp_length = np.minimum(age - 18, np.random.exponential(6, n_samples)).astype(int)
    emp_length = np.maximum(0, emp_length)
    
    home_ownership = np.random.choice(['RENT', 'MORTGAGE', 'OWN'], n_samples, p=[0.4, 0.5, 0.1])
    credit_history_length = np.minimum(age - 18, np.random.randint(1, 20, n_samples))
    credit_history_length = np.maximum(1, credit_history_length)
    
    purpose = np.random.choice(
        ['DEBT_CONSOLIDATION', 'HOME_IMPROVEMENT', 'EDUCATION', 'VENTURE', 'MEDICAL'], 
        n_samples, 
        p=[0.4, 0.2, 0.15, 0.15, 0.10]
    )
    
    historical_default = np.random.choice([0, 1], n_samples, p=[0.88, 0.12])
    
    # Calculate log odds of default
    # Higher debt-to-income increases risk, rent increases risk, past default is heavy risk
    dti = loan_amount / income
    log_odds = (
        0.8 * (dti * 10) 
        - 0.03 * age 
        - 0.08 * emp_length 
        + 0.6 * (home_ownership == 'RENT') 
        - 0.4 * (home_ownership == 'OWN')
        + 1.6 * historical_default 
        - 0.02 * credit_history_length 
        - 1.8 # Intercept
    )
    
    # Convert log odds to probability
    prob = 1 / (1 + np.exp(-log_odds))
    
    # Draw binary default label
    default = np.random.binomial(1, prob)
    
    df = pd.DataFrame({
        'age': age,
        'income': income,
        'loan_amount': loan_amount,
        'emp_length': emp_length,
        'home_ownership': home_ownership,
        'purpose': purpose,
        'credit_history_length': credit_history_length,
        'historical_default': historical_default,
        'default': default
    })
    
    df.to_csv(DATA_PATH, index=False)
    print(f"Mock credit default dataset saved at {DATA_PATH}. Shape: {df.shape}")

# Generate mock data on startup if not present
if not os.path.exists(DATA_PATH):
    generate_mock_credit_data()

def train_scorecard_model():
    """
    Fits optimal binning, trains Logistic Regression model, 
    and scales coefficients to scorecard points.
    """
    global binning_model, logistic_model, scorecard_points, optimal_pd_cutoff, optimal_score_cutoff, score_scaler, model_metrics
    
    if not os.path.exists(DATA_PATH):
        generate_mock_credit_data()
        
    df = pd.read_csv(DATA_PATH)
    
    X = df.drop(columns=['default'])
    y = df['default']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # 1. Fit custom WoE Binning
    binning_model = CustomWoEBinning(max_bins=5)
    binning_model.fit(X_train, y_train)
    
    # 2. Transform datasets to WoE
    X_train_woe = binning_model.transform(X_train)
    X_test_woe = binning_model.transform(X_test)
    
    # 3. Fit regularized Logistic Regression
    # We fit y_train on the WoE variables
    logistic_model = LogisticRegression(penalty='l2', C=1.0, random_state=42)
    logistic_model.fit(X_train_woe, y_train)
    
    # Predict default probabilities on test set
    y_test_prob = logistic_model.predict_proba(X_test_woe)[:, 1]
    
    # 4. Model Evaluation Metrics
    auc = roc_auc_score(y_test, y_test_prob)
    ks_stat, ks_thresh = calculate_ks_statistic(y_test.values, y_test_prob)
    
    # Optimal Cutoff using expected profits
    # Interest rate 12%, LGD 45% (standard corporate benchmark)
    opt_cutoff_prob, thresholds, profits = optimize_decision_cutoff(
        y_test.values, y_test_prob, X_test['loan_amount'].values, interest_rate=0.12, lgd=0.45
    )
    
    # 5. Scale to Scorecard Points
    score_scaler = ScorecardScaler(target_score=600, target_odds=50, pdo=20)
    scorecard_points = score_scaler.scale_scorecard(logistic_model, X_train_woe, binning_model)
    
    # Convert test set probabilities to credit scores
    # Score = Offset + Factor * ln(Odds)
    # Odds = P(Good) / P(Bad) = (1 - P(Default)) / P(Default)
    # Prevent division by zero
    y_test_prob_clamped = np.clip(y_test_prob, 1e-7, 1 - 1e-7)
    test_odds = (1 - y_test_prob_clamped) / y_test_prob_clamped
    test_scores = score_scaler.offset + score_scaler.factor * np.log(test_odds)
    test_scores = np.clip(test_scores, 300, 850).round().astype(int)
    
    # Map optimal PD cutoff to Score cutoff
    opt_cutoff_prob_clamped = np.clip(opt_cutoff_prob, 1e-7, 1 - 1e-7)
    optimal_score_cutoff = int(round(score_scaler.offset + score_scaler.factor * np.log((1 - opt_cutoff_prob_clamped) / opt_cutoff_prob_clamped)))
    optimal_score_cutoff = np.clip(optimal_score_cutoff, 300, 850)
    optimal_pd_cutoff = opt_cutoff_prob
    
    # ROC Curve coordinates
    fpr, tpr, roc_thresholds = roc_curve(y_test, y_test_prob)
    
    # Distribution of scores (histogram)
    score_hist, score_bin_edges = np.histogram(test_scores, bins=25, range=(300, 850))
    
    model_metrics = {
        "roc_auc": float(auc),
        "ks_statistic": float(ks_stat),
        "optimal_pd_cutoff": float(optimal_pd_cutoff),
        "optimal_score_cutoff": int(optimal_score_cutoff),
        "test_scores": test_scores.tolist(),
        "test_defaults": y_test.tolist(),
        "roc_curve": {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist()
        },
        "profit_curve": {
            "thresholds": thresholds,
            "profits": profits
        },
        "score_distribution": {
            "counts": score_hist.tolist(),
            "bin_edges": score_bin_edges.tolist()
        }
    }
    print("Model trained and scorecard scaled successfully.")

# Initial model training
train_scorecard_model()

@app.get("/api/train_metrics")
def get_metrics():
    global scorecard_points, model_metrics
    if scorecard_points is None:
        train_scorecard_model()
    return {
        "metrics": {
            "roc_auc": model_metrics["roc_auc"],
            "ks_statistic": model_metrics["ks_statistic"],
            "optimal_pd_cutoff": model_metrics["optimal_pd_cutoff"],
            "optimal_score_cutoff": model_metrics["optimal_score_cutoff"]
        },
        "scorecard": scorecard_points,
        "roc_curve": model_metrics["roc_curve"],
        "profit_curve": model_metrics["profit_curve"],
        "score_distribution": model_metrics["score_distribution"]
    }

@app.post("/api/appraise")
def appraise_applicant(applicant: ApplicantData):
    global scorecard_points, score_scaler, optimal_score_cutoff, logistic_model, binning_model
    
    if scorecard_points is None:
        train_scorecard_model()
        
    try:
        # Convert applicant model to dict
        applicant_dict = applicant.model_dump()
        
        # 1. Calculate scorecard points and details
        score, breakdown = compute_applicant_score(applicant_dict, scorecard_points)
        
        # Clamp score between regulatory boundaries
        score = int(np.clip(score, 300, 850))
        
        # 2. Predict default probability (PD) directly from the logistic regression model
        # We must transform the applicant dict to WoE
        app_df = pd.DataFrame([applicant_dict])
        app_woe = binning_model.transform(app_df)
        pd_val = float(logistic_model.predict_proba(app_woe)[0, 1])
        
        # 3. Calculate Expected Loss
        # EL = PD * EAD * LGD (using default recovery loss benchmark of 45%)
        expected_loss = calculate_expected_loss(pd_val, applicant.loan_amount, lgd=0.45)
        
        # 4. Underwriting decision
        decision = "APPROVED" if score >= optimal_score_cutoff else "REJECTED"
        
        return {
            "score": score,
            "probability_of_default": pd_val,
            "expected_loss": expected_loss,
            "score_cutoff": optimal_score_cutoff,
            "decision": decision,
            "breakdown": breakdown
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Appraisal error: {str(e)}")

@app.get("/api/download_dataset")
def download_dataset():
    if not os.path.exists(DATA_PATH):
        generate_mock_credit_data()
    return FileResponse(DATA_PATH, media_type="text/csv", filename="loan_dataset.csv")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8003)
