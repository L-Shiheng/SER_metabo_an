import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests

def run_pairwise_statistics(df, group_col, case, control, features, equal_var=False):
    """运行两两比较的 T 检验并计算 FC 和 FDR"""
    g1 = df[df[group_col] == case]
    g2 = df[df[group_col] == control]
    res = []
    
    for f in features:
        v1, v2 = g1[f].values, g2[f].values
        fc = np.mean(v1) - np.mean(v2)
        try:
            t, p = stats.ttest_ind(v1, v2, equal_var=equal_var)
        except:
            p = 1.0
        res.append({'Metabolite': f, 'Log2_FC': fc, 'P_Value': p if not np.isnan(p) else 1.0})
        
    res_df = pd.DataFrame(res).dropna()
    if not res_df.empty: 
        _, res_df['FDR'], _, _ = multipletests(res_df['P_Value'], method='fdr_bh')
        res_df['-Log10_P'] = -np.log10(res_df['P_Value'])
        
    return res_df
