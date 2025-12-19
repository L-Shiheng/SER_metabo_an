# 文件名: serrf_module.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from joblib import Parallel, delayed
import streamlit as st

@st.cache_data(show_spinner=False)
def serrf_normalization(df_data, df_meta, run_order_col, sample_type_col, qc_label, n_trees=100):
    """
    SERRF 批次校正算法 (稳健增强版 v4.0)
    """
    # 1. 数据对齐与检查
    common_idx = df_data.index.intersection(df_meta.index)
    if len(common_idx) < len(df_data): return None, "索引不匹配"
    
    X_raw = df_data.loc[common_idx].copy()
    meta = df_meta.loc[common_idx].copy()
    
    # 确保进样顺序是数字
    try:
        meta[run_order_col] = pd.to_numeric(meta[run_order_col], errors='coerce')
        # 移除没有顺序的样本
        valid_order_mask = meta[run_order_col].notna()
        if (~valid_order_mask).any():
            X_raw = X_raw[valid_order_mask]
            meta = meta[valid_order_mask]
    except:
        return None, "进样顺序包含无法转换的字符"
    
    # 标记 QC
    qc_mask = meta[sample_type_col] == qc_label
    if qc_mask.sum() < 3: return None, f"QC样本不足 ({qc_mask.sum()}<3)"

    run_orders = meta[[run_order_col]].values
    
    # 2. 核心校正函数
    def process_metabolite(met_name, y_all, is_qc, run_orders):
        # 排除 NaN 和 0 值 (0值取对数或做除法会出问题)
        valid_val = (y_all > 0) & (~np.isnan(y_all))
        valid_qc = is_qc & valid_val
        
        # 如果有效 QC 太少，不校正，直接返回原值
        if valid_qc.sum() < 3: return y_all
        
        y_qc = y_all[valid_qc]
        X_qc = run_orders[valid_qc]
        
        # --- 策略优化：使用对数空间进行拟合 ---
        # 代谢物浓度通常服从对数正态分布，在 log 空间拟合漂移更准确
        y_qc_log = np.log1p(y_qc)
        
        # 训练 RF (参数调优：增加 min_samples_leaf 防止过拟合)
        rf = RandomForestRegressor(
            n_estimators=n_trees, 
            min_samples_leaf=6,  # 增大此值，让曲线更平滑，减少对噪音的敏感度
            max_features='sqrt', # 减少特征随机性
            random_state=42, 
            n_jobs=1
        )
        rf.fit(X_qc, y_qc_log)
        
        # 预测漂移曲线
        y_pred_log = rf.predict(run_orders)
        y_pred = np.expm1(y_pred_log)
        
        # 计算校正因子
        qc_target = np.median(y_qc)
        
        # 保护机制：防止预测值过小导致因子爆炸
        # 限制预测值不低于 QC 中位数的 1/10
        min_pred = qc_target * 0.1
        y_pred = np.maximum(y_pred, min_pred)
        
        correction_factor = qc_target / y_pred
        
        # 应用校正
        y_corrected = y_all * correction_factor
        
        # --- QC 内部交叉验证 (仅用于评估 QC 自身质量) ---
        # 为了让 QC 在图上看起来也是被校正过的，我们对 QC 使用 OOB 或 CV 预测值
        # 这里使用 KFold CV
        kf = KFold(n_splits=min(5, valid_qc.sum()))
        y_qc_cv_pred_log = np.zeros_like(y_qc)
        
        for train_idx, test_idx in kf.split(X_qc):
            rf_cv = RandomForestRegressor(n_estimators=n_trees, min_samples_leaf=6, max_features='sqrt', random_state=42, n_jobs=1)
            rf_cv.fit(X_qc[train_idx], y_qc_log[train_idx])
            y_qc_cv_pred_log[test_idx] = rf_cv.predict(X_qc[test_idx])
            
        y_qc_cv_pred = np.expm1(y_qc_cv_pred_log)
        y_qc_cv_pred = np.maximum(y_qc_cv_pred, min_pred)
        
        # 更新 QC 的校正值
        qc_indices = np.where(valid_qc)[0]
        for i, idx in enumerate(qc_indices):
            y_corrected[idx] = y_all[idx] * (qc_target / y_qc_cv_pred[i])
            
        return y_corrected

    # 3. 并行计算
    results = Parallel(n_jobs=-1, backend='loky')(
        delayed(process_metabolite)(col, X_raw[col].values, qc_mask.values, run_orders)
        for col in X_raw.columns
    )
    
    df_corrected = pd.DataFrame(np.array(results).T, index=X_raw.index, columns=X_raw.columns)
    
    # 4. 计算 RSD 评估
    def calc_rsd(df, mask):
        qc_data = df.loc[mask]
        # 过滤掉全 0 列
        qc_data = qc_data.loc[:, (qc_data != 0).any(axis=0)]
        rsd = (qc_data.std() / qc_data.mean()) * 100
        return rsd.median()
    
    rsd_info = {
        "RSD_Before": calc_rsd(X_raw, qc_mask),
        "RSD_After": calc_rsd(df_corrected, qc_mask)
    }
    
    return df_corrected, rsd_info
