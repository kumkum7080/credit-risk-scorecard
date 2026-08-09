# Project 3: Loan Default Risk Scoring & Credit Decision Engine

This project builds a regulatory-grade credit risk scorecard engine that bins numerical and categorical features using Weight of Evidence (WoE) and Information Value (IV), trains a regularized Logistic Regression model, and scales parameters to standardized credit scorecards ($300 - 850$ points). It also determines the optimal cutoff threshold by maximizing net portfolio profits and calculates Expected Loss ($\text{EL}$).

## 🚀 Setup & Execution

### 1. Requirements
Ensure you have the required Python packages installed in your environment:
```bash
pip install numpy pandas scikit-learn fastapi uvicorn
```

### 2. Run the Backend Server
Navigate to the `backend` directory and run:
```bash
python main.py
```
This spins up the FastAPI server on `http://127.0.0.1:8003`. On startup, it automatically generates a realistic credit risk history dataset (`mock_credit_data.csv`) and fits the scorecard.

### 3. Open the Frontend
Simply open `frontend/index.html` directly in any web browser. You can input applicant parameters on the sidebar form and click **Execute Credit Appraisal** to generate a formal decision slip with a detailed points breakdown, or view model ROC curves and score distributions on the diagnostics tab.

---

## 📈 Credit Scorecard Math & Video Guidance Reference

This project is built strictly following the guidelines from the **AIMLModeling** tutorial: *Credit Risk Modeling in Python*.

1. **Weight of Evidence (WoE) & Information Value (IV)**:
   Discretizes continuous parameters using optimal split points (via a Decision Tree classifier). For each bin $i$:
   $$\text{WoE}_i = \ln\left( \frac{\% \text{ Good}_i}{\% \text{ Bad}_i} \right)$$
   $$\text{IV}_i = \left( \% \text{ Good}_i - \% \text{ Bad}_i \right) \times \text{WoE}_i$$
   Variables with $\text{IV} < 0.02$ are pruned as non-predictive.
2. **Logistic Regression on WoE Features**:
   Fits a regularized logistic regression model to estimate probability of default:
   $$\ln\left(\frac{P(\text{Good})}{1 - P(\text{Good})}\right) = \beta_0 + \beta_1 X_{1, \text{WoE}} + \dots + \beta_k X_{k, \text{WoE}}$$
3. **Scorecard Point Scaling**:
   Transforms coefficients $\beta_j$ and intercept $\beta_0$ to points:
   $$\text{Score} = \text{Offset} + \text{Factor} \times \ln(\text{Odds})$$
   Points contribution for variable $j$ in bin $i$ is derived as:
   $$\text{Points}_{i, j} = \left( \beta_j \times \text{WoE}_{i, j} + \frac{\beta_0}{k} \right) \times \text{Factor} + \frac{\text{Offset}}{k}$$
   where $k$ is the number of features. Calibrated with a target score of $600$ at $50:1$ odds and a Points to Double Odds (PDO) of $20$.
4. **Expected Loss (EL)**:
   Computes Expected Loss for each loan:
   $$\text{EL} = \text{PD} \times \text{EAD} \times \text{LGD}$$
   where PD is default probability, EAD is the loan amount requested, and LGD is set to the standard $45\%$ benchmark.
