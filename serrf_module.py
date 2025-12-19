# 文件名: serrf_module.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from joblib import Parallel, delayed
import streamlit as st

# 添加缓存装饰器，避免重复计算
@st.cache_data(show_spinner=False)
def serrf_normalization(df_data, df_meta, run_order_col, sample_type_col, qc_label, n_trees=30):
    """
    SERRF 批次校正算法 (高性能版)
    n_trees: 决策树数量，默认30 (速度快且效果足够)，原版通常为100
    """
    
    # 1. 数据准备
    common_idx = df_data.index.intersection(df_meta.index)
    if len(common_idx) < len(df_data):
        return None, "索引不匹配"
    
    X_raw = df_data.loc[common_idx].copy()
    meta = df_meta.loc[common_idx].copy()
    
    try:
        meta[run_order_col] = pd.to_numeric(meta[run_order_col])
    except:
        return None, "进样顺序包含非数字"
    
    qc_mask = meta[sample_type_col] == qc_label
    if qc_mask.sum() < 3:
        return None, f"QC样本不足 ({qc_mask.sum()}<3)"

    run_orders = meta[[run_order_col]].values
    
    # 2. 定义处理函数
    def process_metabolite(met_name, y_all, is_qc, run_orders):
        valid_qc = is_qc & (~np.isnan(y_all))
        if valid_qc.sum() < 3: return y_all
        
        y_qc = y_all[valid_qc]
        X_qc = run_orders[valid_qc]
        
        # 优化1: 减少 n_estimators 到 30，大幅提速
        rf = RandomForestRegressor(n_estimators=n_trees, min_samples_leaf=3, random_state=42, n_jobs=1)
        rf.fit(X_qc, y_qc)
        
        y_pred = rf.predict(run_orders)
        qc_median = np.median(y_qc)
        y_pred[y_pred <= 0] = 1e-6
        
        correction_factor = qc_median / y_pred
        y_corrected = y_all * correction_factor
        
        # QC 交叉验证 (防止过拟合)
        kf = KFold(n_splits=min(5, valid_qc.sum()))
        y_qc_cv_pred = np.zeros_like(y_qc)
        
        for train_idx, test_idx in kf.split(X_qc):
            # CV 模型也使用轻量化参数
            rf_cv = RandomForestRegressor(n_estimators=n_trees, min_samples_leaf=3, random_state=42, n_jobs=1)
            rf_cv.fit(X_qc[train_idx], y_qc[train_idx])
            y_qc_cv_pred[test_idx] = rf_cv.predict(X_qc[test_idx])
            
        qc_indices = np.where(valid_qc)[0]
        for i, idx in enumerate(qc_indices):
            pred_val = y_qc_cv_pred[i] if y_qc_cv_pred[i] > 0 else 1e-6
            y_corrected[idx] = y_all[idx] * (qc_median / pred_val)
            
        return y_corrected

    # 3. 并行计算
    # 优化2: 使用 threading 后端可能比 loky 更省内存，防止大规模数据导致崩溃
    # n_jobs=-1 自动调用所有核
    results = Parallel(n_jobs=-1, backend='loky')(
        delayed(process_metabolite)(col, X_raw[col].values, qc_mask.values, run_orders)
        for col in X_raw.columns
    )
    
    df_corrected = pd.DataFrame(np.array(results).T, index=X_raw.index, columns=X_raw.columns)
    
    # 计算 RSD 用于评估
    def calc_rsd(df, mask):
        qc = df.loc[mask]
        return (qc.std() / qc.mean() * 100).median()
    
    rsd_info = {
        "RSD_Before": calc_rsd(X_raw, qc_mask),
        "RSD_After": calc_rsd(df_corrected, qc_mask)
    }
    
    return df_corrected, rsd_info
