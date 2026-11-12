import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests

def run_pairwise_statistics(
    df, group_col, case, control, features,
    equal_var=True, df_unscaled_log2=None, df_linear_norm=None
):
    """
    运行两两比较的 T 检验并计算与 MetaboAnalyst 标准一致的 Log2_FC 和 FDR：
    1. Log2_FC：优先在归一化后、对数转化前的线性丰度矩阵 (df_linear_norm) 上计算两组算术均值比值的 Log2；
    2. P_Value & FDR：在归一化并取 Log2（未做特征缩放）的矩阵 (df_unscaled_log2) 上执行 T 检验与 BH-FDR 校正。
    """
    ttest_df = df_unscaled_log2 if df_unscaled_log2 is not None else df
    
    # 自动从 attrs 或显式参数中获取线性归一化矩阵（用于计算真实的算术均值 Fold Change）
    if df_linear_norm is None and df_unscaled_log2 is not None:
        df_linear_norm = df_unscaled_log2.attrs.get('linear_norm_df', None)

    g1_t = ttest_df[ttest_df[group_col] == case]
    g2_t = ttest_df[ttest_df[group_col] == control]

    if df_linear_norm is not None and group_col in df_linear_norm.columns:
        g1_lin = df_linear_norm[df_linear_norm[group_col] == case]
        g2_lin = df_linear_norm[df_linear_norm[group_col] == control]
        has_linear = True
    else:
        g1_lin, g2_lin = g1_t, g2_t
        has_linear = False

    res = []
    for f in features:
        if f not in ttest_df.columns:
            continue
            
        v1_t = g1_t[f].values.astype(float)
        v2_t = g2_t[f].values.astype(float)
        
        # 1. 计算 Fold Change (对标 MetaboAnalyst: 线性归一化丰度的算术均值比值)
        if has_linear and f in g1_lin.columns:
            v1_l = np.clip(g1_lin[f].values.astype(float), 1e-12, None)
            v2_l = np.clip(g2_lin[f].values.astype(float), 1e-12, None)
            mean_case = float(np.mean(v1_l))
            mean_ctrl = float(np.mean(v2_l))
            fc_ratio = mean_case / max(mean_ctrl, 1e-12)
            log2_fc = float(np.log2(max(fc_ratio, 1e-12)))
        else:
            log2_fc = float(np.mean(v1_t) - np.mean(v2_t))
            
        # 2. 计算两独立样本 T 检验 P-value
        try:
            _, p = stats.ttest_ind(v1_t, v2_t, equal_var=equal_var, nan_policy='omit')
        except Exception:
            p = 1.0
            
        p_val = float(p) if (p is not None and not np.isnan(p) and p >= 0) else 1.0
        res.append({
            'Metabolite': f,
            'Log2_FC': log2_fc,
            'P_Value': p_val
        })
        
    res_df = pd.DataFrame(res).dropna()
    if not res_df.empty: 
        _, res_df['FDR'], _, _ = multipletests(res_df['P_Value'], method='fdr_bh')
        res_df['-Log10_P'] = -np.log10(np.clip(res_df['P_Value'].astype(float), 1e-300, 1.0))
        
    return res_df
