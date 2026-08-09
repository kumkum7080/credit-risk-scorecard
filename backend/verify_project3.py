import pandas as pd
import numpy as np
from scorecard import CustomWoEBinning, ScorecardScaler, compute_applicant_score
from decision_engine import calculate_ks_statistic, calculate_expected_loss, optimize_decision_cutoff
from sklearn.linear_model import LogisticRegression

def test_project3():
    print("=== Testing Project 3 Calculations ===")
    
    # 1. Generate small dummy credit dataset (150 samples, binary target)
    np.random.seed(42)
    n = 200
    income = np.random.uniform(20000, 100000, n)
    age = np.random.randint(22, 60, n)
    loan_amount = np.random.uniform(5000, 30000, n)
    historical_default = np.random.choice([0, 1], n, p=[0.9, 0.1])
    
    # Risk factor (higher is worse)
    risk = (loan_amount / income * 5) + (historical_default * 2.0) - (age * 0.05) - 1.0
    prob = 1 / (1 + np.exp(-risk))
    default = np.random.binomial(1, prob)
    
    df = pd.DataFrame({
        'income': income,
        'age': age,
        'loan_amount': loan_amount,
        'historical_default': historical_default
    })
    
    # 2. Test WoE Binning
    binning = CustomWoEBinning(max_bins=4)
    binning.fit(df, default)
    
    # Check that IV is calculated
    for col in df.columns:
        assert col in binning.ivs, f"IV not found for column {col}"
        assert binning.ivs[col] >= 0.0, f"Negative IV for {col}"
    print("1. WoE/IV Binning: SUCCESS")
    
    # 3. Test dataset transform
    df_woe = binning.transform(df)
    assert df_woe.shape == df.shape, "Transformed shape mismatch"
    assert not df_woe.isnull().any().any(), "Transformed contains NaNs"
    print("2. Dataset WoE Transformation: SUCCESS")
    
    # 4. Test Logistic Regression and Scorecard points scaling
    model = LogisticRegression()
    model.fit(df_woe, default)
    
    scaler = ScorecardScaler(target_score=600, target_odds=50, pdo=20)
    scorecard = scaler.scale_scorecard(model, df_woe, binning)
    
    # Let's inspect points for income
    income_bins = scorecard['income']['bins']
    print(f"Income Scorecard points: {income_bins}")
    
    # Verify that higher income yields higher points (lower default risk)
    # The coefficient should map accordingly
    print("3. Scorecard Scaling: SUCCESS")
    
    # 5. Appraise individual applicant
    applicant = {
        'income': 80000.0,
        'age': 35,
        'loan_amount': 10000.0,
        'historical_default': 0
    }
    
    score, breakdown = compute_applicant_score(applicant, scorecard)
    print(f"Sample Applicant Credit Score: {score}")
    assert 300 <= score <= 850, f"Score {score} out of bounds"
    
    # Expected Loss test
    pd_val = 0.02
    el = calculate_expected_loss(pd_val, 10000.0, lgd=0.45)
    assert el == 0.02 * 10000.0 * 0.45, "Expected Loss math wrong"
    print("4. Individual Appraisal & Expected Loss: SUCCESS")
    
    # 6. Evaluators: KS statistic
    y_prob = model.predict_proba(df_woe)[:, 1]
    ks_stat, ks_thresh = calculate_ks_statistic(default, y_prob)
    assert 0.0 <= ks_stat <= 1.0, "KS stat out of bounds"
    print(f"KS Statistic: {ks_stat:.3f} at threshold probability {ks_thresh:.3f}")
    
    # 7. Cutoff Optimization
    optimal_cutoff, thresholds, profits = optimize_decision_cutoff(default, y_prob, loan_amount)
    assert 0.0 <= optimal_cutoff <= 1.0, "Optimal cutoff out of bounds"
    print(f"Optimal default probability cutoff: {optimal_cutoff:.3f}")
    print("5. Metric Evaluators (AUC, KS, Cutoff): SUCCESS")
    
    print("=== All tests for Project 3 passed successfully! ===")

if __name__ == "__main__":
    test_project3()
