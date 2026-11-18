import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from joblib import Parallel, delayed

@st.cache_data(show_spinner=False)
def serrf_normalization(df_data, df_meta, run_order_col, sample_type_col, qc_label, n_trees=100, top_k_corr=10):
    """
    SERRF (Systematic Error Removal using Random Forest) 跨代谢物相关性系统误差消除算法
    结合进样顺序 (run_orders) 与同批次 QC 强相关协变量代谢物 (top_k_corr) 协同构建随机森林回归。
    """
    try:
        data = df_data.copy()
        meta = df_meta.loc[data.index].copy()

        meta[run_order_col] = pd.to_numeric(meta[run_order_col], errors='coerce')
        if meta[run_order_col].isnull().all():
            return None, f"进样顺序列 [{run_order_col}] 无法转换为有效数字。"
        if meta[run_order_col].isnull().any():
            meta[run_order_col] = meta[run_order_col].fillna(pd.Series(np.arange(1, len(meta) + 1), index=meta.index))

        meta = meta.sort_values(by=run_order_col)
        data = data.loc[meta.index]

        qc_mask = (meta[sample_type_col].astype(str).str.strip().str.lower() == str(qc_label).strip().lower()).values
        n_qc = int(np.sum(qc_mask))
        if n_qc < 3:
            return None, f"QC 样本数量不足（当前匹配到 {n_qc} 个 '{qc_label}'，至少需要 3 个）。"

        run_orders = meta[run_order_col].values.reshape(-1, 1).astype(float)
        X_raw = data.values.astype(float)
        n_samples, n_features = X_raw.shape

        # 构建 log1p 协变量矩阵 X_log：对 <= 0 或 NaN 的位置用该特征在 QC 样本的中位数稳健填充
        X_filled = X_raw.copy()
        invalid_mask = np.isnan(X_filled) | (X_filled <= 0)
        X_masked_nan = np.where(invalid_mask, np.nan, X_filled)

        with np.errstate(all='ignore'):
            qc_medians = np.nanmedian(X_masked_nan[qc_mask, :], axis=0)
            all_medians = np.nanmedian(X_masked_nan, axis=0)

        fill_vals = np.where(np.isnan(qc_medians) | (qc_medians <= 0), all_medians, qc_medians)
        fill_vals = np.where(np.isnan(fill_vals) | (fill_vals <= 0), 1e-6, fill_vals)

        row_idx, col_idx = np.where(invalid_mask)
        X_filled[row_idx, col_idx] = fill_vals[col_idx]
        X_log = np.log1p(np.maximum(X_filled, 0.0))

        # 筛选在 QC 样本中检出率 >= 70% 且标准差 > 1e-8 的稳定特征作为候选协变量池
        qc_valid_rate = np.mean(~invalid_mask[qc_mask, :], axis=0)
        qc_log_std = np.std(X_log[qc_mask, :], axis=0)
        candidate_pool = np.where((qc_valid_rate >= 0.70) & (qc_log_std > 1e-8))[0]

        if len(candidate_pool) > 0:
            X_qc_pool = X_log[qc_mask][:, candidate_pool]
            Z_qc_pool = (X_qc_pool - np.mean(X_qc_pool, axis=0)) / (np.std(X_qc_pool, axis=0) + 1e-12)
        else:
            Z_qc_pool = None

        def process_metabolite(j):
            y_all = X_raw[:, j]
            valid_qc = qc_mask & (y_all > 0) & (~np.isnan(y_all))
            if np.sum(valid_qc) < 3:
                return y_all

            # 挑选在 QC 样本上与目标特征 j 绝对相关性最高的前 top_k_corr 个其他代谢物
            cov_cols = []
            if Z_qc_pool is not None and len(candidate_pool) > 1:
                y_qc_vec = X_log[qc_mask, j]
                y_std = np.std(y_qc_vec)
                if y_std > 1e-8:
                    z_y = (y_qc_vec - np.mean(y_qc_vec)) / (y_std + 1e-12)
                    corrs = np.abs(np.dot(Z_qc_pool.T, z_y) / float(n_qc))
                    order_idx = np.argsort(corrs)[::-1]
                    for idx_p in order_idx:
                        cand_col = int(candidate_pool[idx_p])
                        if cand_col != j:
                            cov_cols.append(cand_col)
                            if len(cov_cols) >= top_k_corr:
                                break

            if len(cov_cols) > 0:
                X_feat_all = np.hstack([run_orders, X_log[:, cov_cols]])
            else:
                X_feat_all = run_orders

            y_qc_log = np.log1p(y_all[valid_qc])
            rf = RandomForestRegressor(
                n_estimators=n_trees,
                max_depth=6,
                min_samples_leaf=2,
                n_jobs=1,
                random_state=42
            )
            rf.fit(X_feat_all[valid_qc], y_qc_log)

            y_pred_log = rf.predict(X_feat_all)
            y_pred = np.maximum(np.expm1(y_pred_log), 1e-6)
            qc_target = float(np.median(y_all[valid_qc]))

            y_out = y_all.copy()
            valid_val = (y_all > 0) & (~np.isnan(y_all))
            y_out[valid_val] = y_all[valid_val] * (qc_target / y_pred[valid_val])
            return np.clip(y_out, 0, None)

        results = Parallel(n_jobs=-1, prefer="threads")(
            delayed(process_metabolite)(j) for j in range(n_features)
        )
        X_corrected = np.column_stack(results)
        df_corrected = pd.DataFrame(X_corrected, index=data.index, columns=data.columns)

        def get_rsd(df_in):
            qc_sub = df_in.iloc[qc_mask]
            qc_clean = qc_sub.mask(qc_sub <= 0, np.nan)
            rsds = (qc_clean.std(ddof=1) / qc_clean.mean() * 100.0).dropna()
            return float(rsds.median()) if len(rsds) > 0 else 0.0

        rsd_before = get_rsd(data)
        rsd_after = get_rsd(df_corrected)
        return df_corrected, {'RSD_Before': rsd_before, 'RSD_After': rsd_after}
    except Exception as e:
        return None, str(e)
