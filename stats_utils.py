import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests

def run_pairwise_statistics(df, group_col, case, control, features, equal_var=False, df_unscaled_log2=None):
    """
    运行两两比较的 T 检验并计算准确的 Log2_FC 和 FDR。
    若提供 df_unscaled_log2（归一化并取 Log2、但未做 Pareto/Auto 缩放的矩阵），
    则优先在其上计算真实的 Log2_FC 与 T 检验，避免特征缩放标准差扭曲差异倍数。
    """
    target_df = df_unscaled_log2 if df_unscaled_log2 is not None else df
    g1 = target_df[target_df[group_col] == case]
    g2 = target_df[target_df[group_col] == control]
    res = []
    
    for f in features:
        if f not in target_df.columns:
            continue
        v1, v2 = g1[f].values.astype(float), g2[f].values.astype(float)
        fc = np.mean(v1) - np.mean(v2)
        try:
            t, p = stats.ttest_ind(v1, v2, equal_var=equal_var)
        except Exception:
            p = 1.0
        p_val = float(p) if (not np.isnan(p) and p >= 0) else 1.0
        res.append({'Metabolite': f, 'Log2_FC': fc, 'P_Value': p_val})
        
    res_df = pd.DataFrame(res).dropna()
    if not res_df.empty: 
        _, res_df['FDR'], _, _ = multipletests(res_df['P_Value'], method='fdr_bh')
        res_df['-Log10_P'] = -np.log10(np.clip(res_df['P_Value'].astype(float), 1e-300, 1.0))
        
    return res_df
