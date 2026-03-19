import pandas as pd
import numpy as np
import re
import os
import streamlit as st
from sklearn.impute import KNNImputer
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import r2_score

# ====================
# SIMCA: OPLS-DA 算法与置换检验
# ====================
class OPLS_DA:
    def __init__(self):
        self.t = None        
        self.t_ortho = None  
        self.p = None        
        self.p_corr = None   
        self.vip = None      
        self.R2Y = 0
        self.Q2 = 0

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y).flatten()
        
        # NIPALS 提取 1个预测 + 1个正交分量
        w = np.dot(X.T, y) / np.dot(y.T, y)
        w /= np.linalg.norm(w)
        t = np.dot(X, w) / np.dot(w.T, w)
        p = np.dot(X.T, t) / np.dot(t.T, t)
        
        w_ortho = p - (np.dot(w.T, p) / np.dot(w.T, w)) * w
        w_ortho /= np.linalg.norm(w_ortho)
        t_ortho = np.dot(X, w_ortho) / np.dot(w_ortho.T, w_ortho)
        
        self.t = t.flatten()
        self.t_ortho = t_ortho.flatten()
        self.p = p.flatten()
        
        # p(corr) 和 VIP
        self.p_corr = np.array([np.corrcoef(X[:, i], self.t)[0, 1] for i in range(X.shape[1])])
        w_norm = (w / np.linalg.norm(w)).flatten()
        self.vip = np.sqrt(len(w_norm) * (w_norm ** 2))
        return self

    def evaluate(self, X, y, n_splits=7):
        n_samples = len(y)
        cv_splits = min(n_splits, n_samples)
        if cv_splits < 2: return 0, 0
            
        pls = PLSRegression(n_components=1)
        pls.fit(X, y)
        y_pred_fit = pls.predict(X)
        self.R2Y = r2_score(y, y_pred_fit)
        
        kf = KFold(n_splits=cv_splits, shuffle=True, random_state=42)
        y_cv = cross_val_predict(pls, X, y, cv=kf)
        self.Q2 = r2_score(y, y_cv)
        return self.R2Y, self.Q2

    def permutation_test(self, X, y, n_permutations=100):
        """SIMCA 核心：置换检验 (Permutation Test)"""
        orig_R2Y, orig_Q2 = self.evaluate(X, y)
        r2_perm, q2_perm, correlations = [], [], []
        
        pls = PLSRegression(n_components=1)
        cv_splits = min(7, len(y))
        kf = KFold(n_splits=cv_splits, shuffle=True)
        
        for i in range(n_permutations):
            y_shuffled = np.random.permutation(y)
            # 计算打乱后的 Y 与原 Y 的相关性
            corr = np.abs(np.corrcoef(y, y_shuffled)[0, 1])
            correlations.append(corr)
            
            pls.fit(X, y_shuffled)
            r2_perm.append(r2_score(y_shuffled, pls.predict(X)))
            q2_perm.append(r2_score(y_shuffled, cross_val_predict(pls, X, y_shuffled, cv=kf)))
            
        return np.array(correlations), np.array(r2_perm), np.array(q2_perm), orig_R2Y, orig_Q2

# ====================
# 数据解析与合并
# ====================
def make_unique(series):
    seen = set(); result = []
    for item in series:
        new_item = item; counter = 1
        while new_item in seen:
            new_item = f"{item}_{counter}"; counter += 1
        seen.add(new_item); result.append(new_item)
    return result

def parse_metdna_file(file_buffer, file_name, file_type='csv'):
    try:
        if file_type == 'csv':
            try: df = pd.read_csv(file_buffer, engine='pyarrow')
            except: file_buffer.seek(0); df = pd.read_csv(file_buffer)
        else: df = pd.read_excel(file_buffer)
    except Exception as e: return None, None, f"读取失败: {str(e)}"

    known_meta_cols = {'peak_name', 'mz', 'rt', 'id', 'id_zhulab', 'name', 'formula', 'confidence_level', 'smiles', 'inchikey', 'isotope', 'adduct', 'total_score', 'mz_error', 'rt_error_abs', 'rt_error_rela', 'ms2_score', 'iden_score', 'iden_type', 'peak_group_id', 'base_peak', 'num_peaks', 'cons_formula_pred', 'id_kegg', 'id_hmdb', 'id_metacyc', 'stereo_isomer_id', 'stereo_isomer_name'}
    potential_cols = [c for c in df.columns if c not in known_meta_cols]
    sample_cols = []
    if potential_cols:
        subset = df[potential_cols].head(5)
        is_numeric = subset.apply(lambda x: pd.to_numeric(x, errors='coerce').notna().all())
        sample_cols = is_numeric[is_numeric].index.tolist()
            
    if not sample_cols: return None, None, "未找到样本数据列。"

    file_tag = os.path.splitext(os.path.basename(file_name))[0]
    clean_tag = re.sub(r'[^a-zA-Z0-9_\-\.]', '_', file_tag)
    if 'name' not in df.columns: df['name'] = ""
    if 'confidence_level' not in df.columns: df['confidence_level'] = 'Unknown'
    
    df['name'] = df['name'].fillna("").astype(str)
    mask_annotated = (df['name'] != "") & (df['name'].str.lower() != "nan")
    clean_names = df['name'].str.split(';', expand=True)[0]
    mz_str = df['mz'].map('{:.4f}'.format).astype(str) if 'mz' in df.columns else ""
    rt_str = df['rt'].map('{:.2f}'.format).astype(str) if 'rt' in df.columns else ""
    unannotated_ids = "m/z" + mz_str + "_RT" + rt_str + "_" + clean_tag
    final_ids = np.where(mask_annotated, clean_names + "_" + clean_tag, unannotated_ids)
    final_ids = make_unique(final_ids)

    meta_df = pd.DataFrame({"Metabolite_ID": final_ids, "Original_Name": df['name'], "Clean_Name": np.where(mask_annotated, clean_names, final_ids), "Confidence_Level": df['confidence_level'], "Is_Annotated": mask_annotated, "Source_File": clean_tag})
    meta_df.set_index('Metabolite_ID', inplace=True)
    
    df_data = df[sample_cols].copy()
    df_data.index = meta_df.index
    df_transposed = df_data.T
    df_transposed.reset_index(inplace=True)
    df_transposed.rename(columns={'index': 'SampleID'}, inplace=True)
    df_transposed['Source_Files'] = clean_tag
    df_transposed['Group'] = df_transposed['SampleID'].astype(str).str.extract(r'([^\d]+)')[0].str.strip('._-').fillna("Unknown")
    
    return df_transposed, meta_df, None

def merge_multiple_dfs(results_list):
    if not results_list: return None, None, "无数据"
    best_features = {}; sample_source_map = {}
    for file_idx, (df, meta, fname) in enumerate(results_list):
        if 'SampleID' in df.columns and 'Source_Files' in df.columns:
            current_tag = df['Source_Files'].iloc[0]
            for sid in df['SampleID']:
                if sid not in sample_source_map: sample_source_map[sid] = set()
                sample_source_map[sid].add(current_tag)
        numeric_df = df.select_dtypes(include=[np.number])
        intensities = numeric_df.sum(axis=0)
        for feat_id in numeric_df.columns:
            try: clean_name = meta.loc[feat_id, 'Clean_Name']
            except KeyError: continue
            curr_score = intensities.get(feat_id, 0)
            if clean_name not in best_features or curr_score > best_features[clean_name][2]:
                best_features[clean_name] = (file_idx, feat_id, curr_score)
    
    files_features_to_keep = {i: [] for i in range(len(results_list))}
    for c_name, (f_idx, f_id, score) in best_features.items(): files_features_to_keep[f_idx].append(f_id)
        
    dfs_to_concat = []; base_group_series = None
    for i, (df, meta, fname) in enumerate(results_list):
        if 'SampleID' in df.columns: df = df.set_index('SampleID')
        cols_to_drop = [c for c in ['Group', 'Source_Files'] if c in df.columns]
        if 'Group' in df.columns and base_group_series is None: base_group_series = df['Group']
        df_clean = df.drop(columns=cols_to_drop, errors='ignore')
        dfs_to_concat.append(df_clean[[c for c in files_features_to_keep[i] if c in df_clean.columns]])
        
    try: full_df = pd.concat(dfs_to_concat, axis=1, join='outer')
    except Exception as e: return None, None, f"合并出错: {str(e)}"
    
    full_df.fillna(0, inplace=True)
    if base_group_series is not None: full_df.insert(0, 'Group', base_group_series.reindex(full_df.index).fillna('Unknown'))
    else: full_df.insert(0, 'Group', 'Unknown')
    full_df.reset_index(inplace=True)
    full_df.rename(columns={'index': 'SampleID'}, inplace=True)
    full_df['Source_Files'] = full_df['SampleID'].apply(lambda sid: "; ".join(sorted(list(sample_source_map.get(sid, set())))))
    
    final_ids = [fid for f_list in files_features_to_keep.values() for fid in f_list]
    merged_meta = pd.concat([res[1] for res in results_list]).loc[final_ids]
    return full_df, merged_meta, None

def align_sample_info(data_df, info_df, sample_col_name=None):
    target_col = sample_col_name if sample_col_name and sample_col_name in info_df.columns else info_df.columns[0]
    info_map = {re.sub(r'[^a-zA-Z0-9]', '', str(r[target_col])).lower(): r for _, r in info_df.iterrows()}
    aligned_data = [info_map.get(re.sub(r'[^a-zA-Z0-9]', '', str(sid)).lower(), pd.Series([np.nan]*len(info_df.columns), index=info_df.columns)) for sid in data_df['SampleID']]
    aligned_df = pd.DataFrame(aligned_data)
    aligned_df.index = data_df.index 
    return aligned_df

def data_cleaning_pipeline(df, group_col, missing_thresh=0.5, impute_method='min', norm_method='None', log_transform=True, scale_method='Pareto'):
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in [group_col, 'SampleID', 'Source_Files']]
    meta_cols = [c for c in df.columns if c not in numeric_cols]
    data_df = df[numeric_cols].copy()
    meta_df = df[meta_cols].copy()
    
    data_df = data_df[data_df.isnull().mean()[data_df.isnull().mean() <= missing_thresh].index]
    
    if data_df.isnull().sum().sum() > 0:
        if impute_method == 'min': data_df = data_df.fillna(data_df.min() * 0.5)
        elif impute_method == 'mean': data_df = data_df.fillna(data_df.mean())
        elif impute_method == 'KNN': data_df = pd.DataFrame(KNNImputer(n_neighbors=5).fit_transform(data_df), columns=data_df.columns, index=data_df.index)
        else: data_df = data_df.fillna(0)

    if norm_method == 'Sum': data_df = data_df.div(data_df.sum(axis=1), axis=0) * data_df.sum(axis=1).mean()
    elif norm_method == 'Median': data_df = data_df.div(data_df.median(axis=1), axis=0) * data_df.median(axis=1).mean()
    elif norm_method == 'PQN':
        ref = data_df.median(axis=0); ref[ref <= 0] = 1e-6
        data_df = data_df.div(data_df.div(ref, axis=1).median(axis=1), axis=0)

    if log_transform: data_df = np.log2(data_df + 1) if (data_df <= 0).any().any() else np.log2(data_df)

    if scale_method != 'None':
        mean = data_df.mean(); std = data_df.std()
        if scale_method == 'Auto': data_df = (data_df - mean) / std
        elif scale_method == 'Pareto': data_df = (data_df - mean) / np.sqrt(std)

    data_df = data_df.loc[:, data_df.var() > 1e-9]
    return pd.concat([meta_df, data_df], axis=1), data_df.columns.tolist()
