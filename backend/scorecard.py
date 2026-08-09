import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

class CustomWoEBinning:
    """
    Applies custom Weight of Evidence (WoE) and Information Value (IV) binning 
    for continuous and categorical variables.
    """
    def __init__(self, max_bins=5, min_bin_size_pct=0.05):
        self.max_bins = max_bins
        self.min_bin_size_pct = min_bin_size_pct
        self.bins = {} # Stores bin boundaries and WoE values for each feature
        self.ivs = {}  # Stores IV values for each feature
        
    def fit(self, X, y):
        # Reset storage
        self.bins = {}
        self.ivs = {}
        
        # Calculate total Good (0) and Bad (1)
        total_good = np.sum(y == 0)
        total_bad = np.sum(y == 1)
        
        # If no bad or no good, cannot compute log-odds
        if total_good == 0 or total_bad == 0:
            raise ValueError("Target variable y must contain both classes (0 and 1)")
            
        for col in X.columns:
            series = X[col]
            
            if pd.api.types.is_numeric_dtype(series):
                # Numeric variables: use a decision tree to find optimal splits
                # This matches optbinning's optimal binning behavior
                self._fit_numeric(col, series, y, total_good, total_bad)
            else:
                # Categorical variables: group by categories
                self._fit_categorical(col, series, y, total_good, total_bad)
                
    def _fit_numeric(self, col_name, series, y, total_good, total_bad):
        # Handle missing values
        clean_mask = ~series.isnull()
        if clean_mask.sum() == 0:
            # All nulls
            self.bins[col_name] = {
                'type': 'numeric',
                'splits': [],
                'bins_woe': {0: 0.0},
                'missing_woe': 0.0
            }
            self.ivs[col_name] = 0.0
            return
            
        X_clean = series[clean_mask].values.reshape(-1, 1)
        y_clean = y[clean_mask]
        
        # Use a simple DecisionTreeClassifier to find optimal bins
        dt = DecisionTreeClassifier(
            max_leaf_nodes=self.max_bins,
            min_samples_leaf=max(1, int(len(X_clean) * self.min_bin_size_pct))
        )
        dt.fit(X_clean, y_clean)
        
        # Extract splits from decision tree
        thresholds = dt.tree_.threshold
        splits = sorted(list(set(thresholds[thresholds != -2]))) # -2 represents leaves
        
        # Create bin boundaries
        boundaries = [-np.inf] + splits + [np.inf]
        
        # Digitize continuous values into bins
        bin_indices = np.digitize(X_clean.flatten(), splits)
        
        # Compute WoE for each bin
        bin_woe = {}
        col_iv = 0.0
        
        # Bins are 0-indexed index numbers from digitize
        for idx in range(len(boundaries) - 1):
            mask = (bin_indices == idx)
            good_count = np.sum(y_clean[mask] == 0)
            bad_count = np.sum(y_clean[mask] == 1)
            
            # Epsilon adjustment to avoid log(0) or div by 0
            good_pct = (good_count / total_good) if total_good > 0 else 0
            bad_pct = (bad_count / total_bad) if total_bad > 0 else 0
            
            # Add small epsilon
            good_pct = max(good_pct, 1e-5)
            bad_pct = max(bad_pct, 1e-5)
            
            woe = np.log(good_pct / bad_pct)
            bin_woe[idx] = woe
            
            col_iv += (good_pct - bad_pct) * woe
            
        # Handle missing values if any
        missing_woe = 0.0
        if (~clean_mask).any():
            missing_good = np.sum(y[~clean_mask] == 0)
            missing_bad = np.sum(y[~clean_mask] == 1)
            good_pct = max(missing_good / total_good, 1e-5)
            bad_pct = max(missing_bad / total_bad, 1e-5)
            missing_woe = np.log(good_pct / bad_pct)
            
        self.bins[col_name] = {
            'type': 'numeric',
            'splits': splits,
            'bins_woe': bin_woe,
            'missing_woe': missing_woe
        }
        self.ivs[col_name] = col_iv
        
    def _fit_categorical(self, col_name, series, y, total_good, total_bad):
        # Convert series to string
        series_str = series.fillna('Missing').astype(str)
        
        unique_cats = series_str.unique()
        bin_woe = {}
        col_iv = 0.0
        
        for cat in unique_cats:
            mask = (series_str == cat)
            good_count = np.sum(y[mask] == 0)
            bad_count = np.sum(y[mask] == 1)
            
            good_pct = max(good_count / total_good, 1e-5)
            bad_pct = max(bad_count / total_bad, 1e-5)
            
            woe = np.log(good_pct / bad_pct)
            bin_woe[cat] = woe
            
            col_iv += (good_pct - bad_pct) * woe
            
        self.bins[col_name] = {
            'type': 'categorical',
            'bins_woe': bin_woe,
            'missing_woe': bin_woe.get('Missing', 0.0)
        }
        self.ivs[col_name] = col_iv
        
    def transform(self, X):
        """
        Transforms raw features into WoE values.
        """
        X_woe = pd.DataFrame(index=X.index)
        
        for col in X.columns:
            if col not in self.bins:
                continue
                
            series = X[col]
            bin_info = self.bins[col]
            
            if bin_info['type'] == 'numeric':
                # Convert to numpy array and digitize
                vals = series.values
                splits = bin_info['splits']
                
                # Check for nulls
                null_mask = pd.isnull(series)
                
                # Digitize non-nulls
                non_null_vals = series[~null_mask].values
                if len(splits) > 0:
                    digitized = np.digitize(non_null_vals, splits)
                else:
                    digitized = np.zeros(len(non_null_vals), dtype=int)
                    
                # Assemble transformed column
                transformed = np.zeros(len(series))
                # Fill null values with missing_woe
                transformed[null_mask] = bin_info['missing_woe']
                # Fill non-null values with respective bin woes
                woes = [bin_info['bins_woe'][idx] for idx in digitized]
                transformed[~null_mask] = woes
                
                X_woe[col] = transformed
            else:
                # Categorical mapping
                series_str = series.fillna('Missing').astype(str)
                woes = series_str.map(bin_info['bins_woe']).fillna(bin_info['missing_woe'])
                X_woe[col] = woes
                
        return X_woe

class ScorecardScaler:
    """
    Scales Logistic Regression coefficients and WoE mappings into scorecard points.
    Rules:
      Score = Offset + Factor * ln(Odds)
    """
    def __init__(self, target_score=600, target_odds=50, pdo=20):
        self.target_score = target_score
        self.target_odds = target_odds
        self.pdo = pdo
        
        # Calculate scaling factor and offset
        # Factor = pdo / ln(2)
        self.factor = pdo / np.log(2)
        # Offset = target_score - Factor * ln(target_odds)
        self.offset = target_score - (self.factor * np.log(target_odds))
        
    def scale_scorecard(self, model, X_train_woe, binning_obj):
        """
        Derives integer scorecard points for each variable's bin.
        Points = (beta_j * WoE_j_bin + intercept/k) * Factor + Offset/k
        """
        coefs = model.coef_[0]
        intercept = model.intercept_[0]
        features = X_train_woe.columns
        k = len(features)
        
        scorecard_points = {}
        
        # Base points (Offset / k + Intercept * Factor / k)
        # This will be distributed across features for convenience
        base_points_per_feature = (self.offset / k) + (intercept * self.factor / k)
        
        for idx, col in enumerate(features):
            bin_info = binning_obj.bins[col]
            scorecard_points[col] = {
                'type': bin_info['type'],
                'bins': {}
            }
            
            beta_j = coefs[idx]
            
            if bin_info['type'] == 'numeric':
                splits = bin_info['splits']
                # Labels for bins (e.g. "< 30", "30 to 45", ">= 45")
                bin_labels = {}
                
                for bin_idx, woe in bin_info['bins_woe'].items():
                    # Generate string labels for formatting
                    if len(splits) == 0:
                        lbl = "All Values"
                    elif bin_idx == 0:
                        lbl = f"< {splits[0]:.2f}"
                    elif bin_idx == len(splits):
                        lbl = f">= {splits[-1]:.2f}"
                    else:
                        lbl = f"{splits[bin_idx-1]:.2f} to {splits[bin_idx]:.2f}"
                        
                    points = (beta_j * woe) * self.factor + base_points_per_feature
                    bin_labels[lbl] = int(round(points))
                    
                scorecard_points[col]['bins'] = bin_labels
                scorecard_points[col]['splits'] = splits
                scorecard_points[col]['coef'] = beta_j
                
            else:
                # Categorical
                bin_labels = {}
                for cat, woe in bin_info['bins_woe'].items():
                    points = (beta_j * woe) * self.factor + base_points_per_feature
                    bin_labels[cat] = int(round(points))
                    
                scorecard_points[col]['bins'] = bin_labels
                scorecard_points[col]['coef'] = beta_j
                
        return scorecard_points

def compute_applicant_score(applicant_data, scorecard):
    """
    Computes credit scorecard points and contribution for an individual applicant.
    applicant_data: dict of raw application variables.
    """
    total_score = 0
    breakdown = {}
    
    for col, card in scorecard.items():
        val = applicant_data.get(col)
        
        if card['type'] == 'numeric':
            if val is None or pd.isna(val):
                # Fallback to missing or first label
                lbl = list(card['bins'].keys())[0] # simple fallback
                pts = card['bins'][lbl]
            else:
                splits = card['splits']
                # Find appropriate bin index
                bin_idx = np.digitize([val], splits)[0]
                
                # Retrieve label corresponding to bin index
                lbl = ""
                if len(splits) == 0:
                    lbl = "All Values"
                elif bin_idx == 0:
                    lbl = f"< {splits[0]:.2f}"
                elif bin_idx == len(splits):
                    lbl = f">= {splits[-1]:.2f}"
                else:
                    lbl = f"{splits[bin_idx-1]:.2f} to {splits[bin_idx]:.2f}"
                
                pts = card['bins'].get(lbl, 0)
                
            total_score += pts
            breakdown[col] = {
                'value': float(val) if val is not None else None,
                'bin': lbl,
                'points': pts
            }
        else:
            # Categorical
            val_str = str(val) if val is not None else 'Missing'
            pts = card['bins'].get(val_str, card['bins'].get('Missing', 0))
            total_score += pts
            breakdown[col] = {
                'value': val_str,
                'bin': val_str,
                'points': pts
            }
            
    return total_score, breakdown
