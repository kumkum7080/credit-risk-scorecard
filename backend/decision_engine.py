import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

def calculate_ks_statistic(y_true, y_prob):
    """
    Computes the Kolmogorov-Smirnov (KS) statistic.
    KS is the maximum separation between cumulative default and non-default distributions.
    """
    df = pd.DataFrame({'y': y_true, 'prob': y_prob})
    
    # Sort by probability descending (high risk to low risk or vice versa, typically ascending for score/descending for prob)
    df = df.sort_values(by='prob', ascending=True).reset_index(drop=True)
    
    total_goods = np.sum(df['y'] == 0)
    total_bads = np.sum(df['y'] == 1)
    
    if total_goods == 0 or total_bads == 0:
        return 0.0, 0.0
        
    # Cumulative counts
    df['cum_goods'] = (df['y'] == 0).cumsum()
    df['cum_bads'] = (df['y'] == 1).cumsum()
    
    # Cumulative percentages
    df['cum_goods_pct'] = df['cum_goods'] / total_goods
    df['cum_bads_pct'] = df['cum_bads'] / total_bads
    
    # Absolute difference
    df['difference'] = np.abs(df['cum_goods_pct'] - df['cum_bads_pct'])
    
    ks_stat = df['difference'].max()
    # Find the threshold probability at which max separation occurs
    ks_index = df['difference'].idxmax()
    ks_threshold = df.loc[ks_index, 'prob']
    
    return float(ks_stat), float(ks_threshold)

def calculate_expected_loss(pd_val, loan_amount, lgd=0.45):
    """
    Expected Loss (EL) = PD * EAD * LGD
    Here EAD (Exposure at Default) is the loan amount.
    LGD (Loss Given Default) is set to 45% (standard corporate unsecured recovery benchmark).
    """
    return pd_val * loan_amount * lgd

def optimize_decision_cutoff(y_true, y_prob, loan_amounts, interest_rate=0.12, lgd=0.45):
    """
    Determines the optimal credit score/default probability cutoff that maximizes expected profit.
    For a loan:
      - If approved and defaults: Loss = - (loan_amount * lgd)
      - If approved and does not default: Profit = loan_amount * interest_rate
      - If rejected: Profit = 0
    Returns:
      optimal_cutoff: probability threshold below which we approve
      profit_curve: list of expected profits at various cutoffs
    """
    thresholds = np.linspace(0.01, 0.99, 100)
    profits = []
    
    # Calculate for each threshold
    for t in thresholds:
        approved_mask = (y_prob <= t)
        
        # Approved defaults (Bad)
        approved_bads = (approved_mask & (y_true == 1))
        # Approved non-defaults (Good)
        approved_goods = (approved_mask & (y_true == 0))
        
        loss = np.sum(loan_amounts[approved_bads] * lgd)
        revenue = np.sum(loan_amounts[approved_goods] * interest_rate)
        
        net_profit = revenue - loss
        profits.append(float(net_profit))
        
    optimal_idx = np.argmax(profits)
    optimal_cutoff = thresholds[optimal_idx]
    
    return float(optimal_cutoff), thresholds.tolist(), profits
