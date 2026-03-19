import pandas as pd
import numpy as np
import re
import os
import streamlit as st
from sklearn.impute import KNNImputer
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import r2_score
from scipy.stats import hypergeom

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
        orig_R2Y, orig_Q2 = self.evaluate(X, y)
        r2_perm, q2_perm, correlations = [], [], []
        
        pls = PLSRegression(n_components=1)
        cv_splits = min(7, len(y))
        kf = KFold(n_splits=cv_splits, shuffle=True)
        
        for i in range(n_permutations):
            y_shuffled = np.random.permutation(y)
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

# ====================
# 通路富集分析核心引擎 (支持动态加载外部完整数据库)
# ====================
def run_pathway_enrichment(sig_metabolites, background_metabolites, custom_db_source=None):
    """
    优先读取外部数据库 (csv/gmt)，如果没找到则使用极小回退库防止崩溃。
    """
    raw_pathways = {}
    
    # 辅助清理函数
    def clean_met_name(name):
        return re.sub(r'[^a-z0-9]', '', str(name).lower())

    # 1. 尝试从本地或上传文件加载完整库
    if custom_db_source is not None:
        try:
            # 如果是 Streamlit 侧边栏上传的文件对象
            if hasattr(custom_db_source, 'name'):
                fname = custom_db_source.name
                if fname.endswith('.gmt'):
                    content = custom_db_source.getvalue().decode("utf-8")
                    for line in content.strip().split('\n'):
                        parts = line.split('\t')
                        if len(parts) >= 3: raw_pathways[parts[0]] = parts[2:]
                elif fname.endswith('.csv'):
                    df_db = pd.read_csv(custom_db_source)
                    if 'Pathway' in df_db.columns and 'Metabolite' in df_db.columns:
                        raw_pathways = df_db.groupby('Pathway')['Metabolite'].apply(lambda x: list(x.dropna().astype(str))).to_dict()
                    else:
                        for _, row in df_db.iterrows():
                            vals = row.dropna().astype(str).tolist()
                            if len(vals) > 1: raw_pathways[vals[0]] = vals[1:]
            
            # 如果是一个字符串路径 (例如仓库根目录下的 "kegg_pathways.csv")
            elif isinstance(custom_db_source, str) and os.path.exists(custom_db_source):
                if custom_db_source.endswith('.gmt'):
                    with open(custom_db_source, 'r', encoding='utf-8') as f:
                        for line in f:
                            parts = line.strip().split('\t')
                            if len(parts) >= 3: raw_pathways[parts[0]] = parts[2:]
                elif custom_db_source.endswith('.csv'):
                    df_db = pd.read_csv(custom_db_source)
                    if 'Pathway' in df_db.columns and 'Metabolite' in df_db.columns:
                        raw_pathways = df_db.groupby('Pathway')['Metabolite'].apply(lambda x: list(x.dropna().astype(str))).to_dict()
                    else:
                        for _, row in df_db.iterrows():
                            vals = row.dropna().astype(str).tolist()
                            if len(vals) > 1: raw_pathways[vals[0]] = vals[1:]
        except Exception as e:
            print(f"外部数据库加载异常: {e}")

    # 2. 如果库是空的（比如文件还没准备好），防止程序崩溃，提供极简备用库
    if not raw_pathways:
        raw_pathways = {
            "Please upload custom database (请上传或配置完整通路库)": ["glucose", "citrate", "pyruvate"]
        }

    # 构建处理后的库字典
    processed_pathways = {}
    for pw, mets in raw_pathways.items():
        processed_pathways[pw] = set([clean_met_name(m) for m in mets])

    # 用户数据处理
    sig_set = set([clean_met_name(m) for m in sig_metabolites])
    bg_set = set([clean_met_name(m) for m in background_metabolites])
    
    N = len(bg_set) if len(bg_set) > 0 else 1000 
    n = len(sig_set)
    results = []
    
    for pathway_name, pw_set in processed_pathways.items():
        K_set = pw_set.intersection(bg_set)
        K = len(K_set)
        if K == 0: K = len(pw_set) # 如果在背景完全未检出，以全库数量兜底防止除零
            
        hits_set = pw_set.intersection(sig_set)
        k = len(hits_set)
        
        if k > 0:
            hit_originals = [orig for orig in sig_metabolites if clean_met_name(orig) in hits_set]
            expected = (K / N) * n
            enrichment_factor = k / expected if expected > 0 else 0
            
            p_val = hypergeom.sf(k - 1, N, K, n)
            
            results.append({
                "Pathway": pathway_name,
                "Total_in_Pathway": K,
                "Hits": k,
                "Hit_Metabolites": ", ".join(hit_originals),
                "Enrichment_Factor": enrichment_factor,
                "P_Value": p_val
            })
            
    res_df = pd.DataFrame(results)
    if not res_df.empty:
        from statsmodels.stats.multitest import multipletests
        try:
            _, fdr, _, _ = multipletests(res_df['P_Value'], method='fdr_bh')
            res_df['FDR'] = fdr
        except: res_df['FDR'] = res_df['P_Value']
        res_df['-Log10_P'] = -np.log10(res_df['P_Value'].astype(float) + 1e-300)
        res_df = res_df.sort_values("P_Value")
        
    return res_df
