# 文件名: serrf_module.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from joblib import Parallel, delayed
import streamlit as st

def serrf_normalization(df_data, df_meta, run_order_col, sample_type_col, qc_label):
    """
    SERRF 批次校正算法 (基于随机森林)
    
    参数:
    - df_data: 丰度矩阵 (行=样本, 列=代谢物)
    - df_meta: 样本元数据 (必须包含进样顺序和样本类型)
    - run_order_col: 进样顺序的列名
    - sample_type_col: 样本类型的列名
    - qc_label: 样本类型列中代表 QC 的字符 (如 'QC')
    
    返回:
    - corrected_df: 校正后的数据表
    - rsd_improvement: 包含校正前后 QC RSD 的字典，用于绘图评估
    """
    
    # 1. 数据准备
    # 确保元数据和数据索引对齐
    common_idx = df_data.index.intersection(df_meta.index)
    if len(common_idx) < len(df_data):
        return None, "数据表与样本信息表索引不匹配，请检查 SampleID 是否一致。"
    
    X_raw = df_data.loc[common_idx].copy()
    meta = df_meta.loc[common_idx].copy()
    
    # 确保进样顺序是数字
    try:
        meta[run_order_col] = pd.to_numeric(meta[run_order_col])
    except:
        return None, f"'{run_order_col}' 列包含非数字字符，无法作为进样顺序。"
    
    # 区分 QC 和 样本
    qc_mask = meta[sample_type_col] == qc_label
    sample_mask = ~qc_mask
    
    if qc_mask.sum() < 3:
        return None, f"QC 样本数量不足 ({qc_mask.sum()})，SERRF 至少需要 3 个 QC。"

    # 提取进样顺序作为特征
    run_orders = meta[[run_order_col]].values
    
    # 2. 定义单个代谢物的 SERRF 校正函数 (用于并行)
    def process_metabolite(met_name, y_all, is_qc, run_orders):
        # y_all: 该代谢物所有样本的强度
        # is_qc: boolean mask
        
        # 移除 NaN (RF 不支持 NaN，这里简单填充或忽略，SERRF通常建议先填充缺失值)
        # 这里采用: 仅使用非空 QC 训练
        valid_qc = is_qc & (~np.isnan(y_all))
        
        if valid_qc.sum() < 3:
            return y_all # QC 太少无法训练，返回原值
        
        y_qc = y_all[valid_qc]
        X_qc = run_orders[valid_qc]
        
        # --- 步骤 A: 训练 RF ---
        # n_estimators=100 是 SERRF 的默认推荐配置
        rf = RandomForestRegressor(n_estimators=100, min_samples_leaf=3, random_state=42, n_jobs=1)
        rf.fit(X_qc, y_qc)
        
        # --- 步骤 B: 预测所有样本的系统漂移 ---
        # 预测所有样本的期望强度
        y_pred = rf.predict(run_orders)
        
        # --- 步骤 C: 计算校正因子 ---
        # SERRF 逻辑: Corrected = Raw * (Median_QC / Predicted)
        qc_median = np.median(y_qc)
        
        # 防止除以 0
        y_pred[y_pred <= 0] = 1e-6
        
        correction_factor = qc_median / y_pred
        y_corrected = y_all * correction_factor
        
        # --- 步骤 D: QC 内部交叉验证 (为了防止过拟合 QC) ---
        # 对 QC 样本本身，使用 Leave-one-out 或 K-Fold 预测的值来校正
        # 这样能真实反映 QC 的校正效果，而不是因为过拟合导致 QC 变平
        kf = KFold(n_splits=min(5, valid_qc.sum()))
        y_qc_cv_pred = np.zeros_like(y_qc)
        
        for train_idx, test_idx in kf.split(X_qc):
            X_train, X_test = X_qc[train_idx], X_qc[test_idx]
            y_train = y_qc[train_idx]
            
            rf_cv = RandomForestRegressor(n_estimators=100, min_samples_leaf=3, random_state=42, n_jobs=1)
            rf_cv.fit(X_train, y_train)
            y_qc_cv_pred[test_idx] = rf_cv.predict(X_test)
            
        # 更新 QC 的校正值 (用 CV 预测值)
        # 找到对应原数组的位置
        qc_indices = np.where(valid_qc)[0]
        for i, idx in enumerate(qc_indices):
            pred_val = y_qc_cv_pred[i] if y_qc_cv_pred[i] > 0 else 1e-6
            y_corrected[idx] = y_all[idx] * (qc_median / pred_val)
            
        return y_corrected

    # 3. 并行计算 (使用 joblib)
    # n_jobs=-1 使用所有 CPU 核心
    results = Parallel(n_jobs=-1)(
        delayed(process_metabolite)(col, X_raw[col].values, qc_mask.values, run_orders)
        for col in X_raw.columns
    )
    
    # 4. 重组数据
    df_corrected = pd.DataFrame(np.array(results).T, index=X_raw.index, columns=X_raw.columns)
    
    # 5. 计算评估指标 (QC RSD)
    def calc_rsd(df, mask):
        qc_data = df.loc[mask]
        rsd = (qc_data.std() / qc_data.mean()) * 100
        return rsd.median()
    
    rsd_before = calc_rsd(X_raw, qc_mask)
    rsd_after = calc_rsd(df_corrected, qc_mask)
    
    return df_corrected, {"RSD_Before": rsd_before, "RSD_After": rsd_after}
