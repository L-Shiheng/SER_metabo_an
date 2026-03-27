import pandas as pd
import numpy as np
import re
import os
from sklearn.impute import KNNImputer
from sklearn.cross_decomposition import PLSRegression
from scipy import stats
import statsmodels.stats.multitest

# ==========================================
# 0. 超级大字典构建引擎 (🌟 增强了抗干扰与大小写兼容)
# ==========================================
def build_kegg_dictionary(dict_files):
    """提取 MetDNA 文件中的名称与 KEGG ID，构建抗干扰的超级全局字典"""
    kegg_mapping = {}
    if not dict_files: return kegg_mapping
    for file in dict_files:
        try:
            file.seek(0)
            if file.name.endswith('.csv'):
                try: df = pd.read_csv(file, engine='pyarrow')
                except: file.seek(0); df = pd.read_csv(file, low_memory=False)
            else: df = pd.read_excel(file)
            
            # 智能嗅探名字列和 KEGG 列
            name_col = next((c for c in df.columns if str(c).lower() in ['name', 'metabolite', 'peak_name', '化合物名称']), df.columns[0])
            kegg_col = next((c for c in df.columns if 'KEGG' in str(c).upper()), None)
            
            if kegg_col:
                for _, row in df.iterrows():
                    n = str(row[name_col]).strip()
                    k = str(row[kegg_col]).strip()
                    if n and n.lower() != 'nan' and k and k.lower() != 'nan':
                        # 只要第一个精准的 KEGG ID
                        k_clean = k.split(';')[0].strip()
                        # 全小写映射，极大增加匹配率
                        kegg_mapping[n.lower()] = k_clean
                        # 分号前的干净名字映射
                        clean_n = n.split(';')[0].strip().lower()
                        kegg_mapping[clean_n] = k_clean
        except Exception: pass
    return kegg_mapping

# ==========================================
# 1. 数据读取与解析模块
# ==========================================
def make_unique(series):
    seen = set(); result = []
    for item in series:
        new_item = str(item).strip()
        base_item = new_item
        counter = 1
        while new_item in seen:
            new_item = f"{base_item}_{counter}"
            counter += 1
        seen.add(new_item); result.append(new_item)
    return result

def parse_metdna_file(file_buffer, file_name, file_type='csv'):
    try:
        if file_type == 'csv':
            try: df = pd.read_csv(file_buffer, engine='pyarrow')
            except: file_buffer.seek(0); df = pd.read_csv(file_buffer, low_memory=False)
        else: df = pd.read_excel(file_buffer)
    except Exception as e: return None, None, f"读取失败: {str(e)}"

    known_meta_cols = {'peak_name', 'mz', 'rt', 'id', 'id_zhulab', 'name', 'formula', 'confidence_level', 'smiles', 'inchikey', 'isotope', 'adduct', 'total_score', 'mz_error', 'rt_error_abs', 'rt_error_rela', 'ms2_score', 'iden_score', 'iden_type', 'peak_group_id', 'base_peak', 'num_peaks', 'cons_formula_pred', 'id_kegg', 'id_hmdb', 'id_metacyc', 'stereo_isomer_id', 'stereo_isomer_name'}
    potential_cols = [c for c in df.columns if str(c).lower() not in known_meta_cols]
    sample_cols = []
    if potential_cols:
        subset = df[potential_cols].head(5)
        is_numeric = subset.apply(lambda x: pd.to_numeric(x, errors='coerce').notna().all())
        sample_cols = is_numeric[is_numeric].index.tolist()
            
    if not sample_cols: return None, None, "未找到样本数据列。"

    file_tag = os.path.splitext(os.path.basename(file_name))[0]
    clean_tag = re.sub(r'[^a-zA-Z0-9_\-\.]', '_', file_tag)
    if 'name' not in df.columns: df['name'] = ""
    
    df['name'] = df['name'].fillna("").astype(str)
    mask_annotated = (df['name'] != "") & (df['name'].str.lower() != "nan")
    clean_names = df['name'].str.split(';', expand=True)[0]
    mz_str = df['mz'].map('{:.4f}'.format).astype(str) if 'mz' in df.columns else ""
    rt_str = df['rt'].map('{:.2f}'.format).astype(str) if 'rt' in df.columns else ""
    unannotated_ids = "m/z" + mz_str + "_RT" + rt_str + "_" + clean_tag
    final_ids = np.where(mask_annotated, clean_names + "_" + clean_tag, unannotated_ids)

    # 🌟 强行将 KEGG ID 焊死在列名上，确保导出可见
    kegg_col = next((c for c in df.columns if 'KEGG' in str(c).upper()), None)
    if kegg_col is not None:
        kegg_vals = df[kegg_col].fillna('').astype(str).values
        final_ids_with_kegg = [f"{n} | {k}" if str(k).strip() and str(k).lower() != 'nan' else str(n) for n, k in zip(final_ids, kegg_vals)]
    else:
        final_ids_with_kegg = final_ids
        
    final_ids_with_kegg = make_unique(final_ids_with_kegg)

    meta_df = pd.DataFrame({
        "Original_Name": final_ids_with_kegg, 
        "Clean_Name": np.where(mask_annotated, clean_names, final_ids), 
        "Is_Annotated": mask_annotated
    }, index=final_ids_with_kegg)
    
    df_data = df[sample_cols].copy()
    df_data.index = meta_df.index
    df_transposed = df_data.T
    
    df_transposed = df_transposed.apply(pd.to_numeric, errors='coerce')
    df_transposed.reset_index(inplace=True)
    df_transposed.rename(columns={'index': 'SampleID'}, inplace=True)
    df_transposed['Source_Files'] = clean_tag
    df_transposed['Group'] = 'Unknown'
    
    return df_transposed, meta_df, None

def parse_manual_targeted_files(file_list, metric_suffix=" : 面积 ", mode_regex=r'-(P|N|RP-P|RP-N|HILIC-P|HILIC-N|POS|NEG)-', external_kegg_dict=None):
    if external_kegg_dict is None: external_kegg_dict = {}
    try:
        all_dfs = []
        local_kegg_mapping = {} 
        
        for file in file_list:
            file.seek(0)
            if file.name.endswith('.csv'): df = pd.read_csv(file)
            else: df = pd.read_excel(file)
            
            comp_col = df.columns[0] 
            kegg_col = next((c for c in df.columns if 'KEGG' in str(c).upper()), None)
            if kegg_col:
                for _, row in df.iterrows():
                    n = str(row[comp_col]).strip()
                    k = str(row[kegg_col]).strip()
                    if k and k.lower() != 'nan': local_kegg_mapping[n.lower()] = k
                    
            metric_cols = [c for c in df.columns if metric_suffix in str(c)]
            if not metric_cols: continue 
                
            sub_df = df[[comp_col] + metric_cols].copy()
            sub_df.rename(columns={comp_col: '__Compound__'}, inplace=True)
            
            def clean_col_name(c):
                c = str(c).replace(metric_suffix, "").strip() 
                c = re.sub(mode_regex, '-', c, flags=re.IGNORECASE) 
                return c.replace('--', '-') 
                
            sub_df.rename(columns={c: clean_col_name(c) for c in metric_cols}, inplace=True)
            for c in sub_df.columns[1:]:
                sub_df[c] = pd.to_numeric(sub_df[c], errors='coerce')
                
            sub_df['__mean_resp__'] = sub_df.iloc[:, 1:].mean(axis=1)
            all_dfs.append(sub_df)
            
        if not all_dfs: return None, None, "未找到指定的提取指标！"
            
        combined = pd.concat(all_dfs, ignore_index=True)
        combined = combined.sort_values('__mean_resp__', ascending=False).drop_duplicates(subset=['__Compound__'])
        combined = combined.drop(columns=['__mean_resp__'])
        
        combined.set_index('__Compound__', inplace=True)
        
        # 🌟 核心：查字典，并将查到的 KEGG ID 强行焊死在列名里！
        orig_names = []
        for n in combined.index:
            n_lower = str(n).strip().lower()
            if n_lower in local_kegg_mapping:
                orig_names.append(f"{n} | {local_kegg_mapping[n_lower]}")
            elif n_lower in external_kegg_dict:
                orig_names.append(f"{n} | {external_kegg_dict[n_lower]}")
            else:
                n_split = n_lower.split(';')[0].strip()
                if n_split in external_kegg_dict:
                    orig_names.append(f"{n} | {external_kegg_dict[n_split]}")
                else:
                    orig_names.append(n)
                    
        # 用焊上了 KEGG 的名字替换掉原来的名字
        combined.index = orig_names
        
        df_t = combined.T
        df_t.index.name = 'SampleID'
        df_t = df_t.reset_index()
        df_t['Group'] = 'Unknown'
        df_t['Source_Files'] = 'Manual_Targeted_Merged'
        
        meta = pd.DataFrame(index=combined.index)
        meta['Clean_Name'] = [str(n).split(' | ')[0] for n in combined.index]
        meta['Original_Name'] = combined.index
        meta['Is_Annotated'] = True 
        
        return df_t, meta, None
    except Exception as e:
        return None, None, f"手动表格解析失败: {str(e)}"

def merge_multiple_dfs(parsed_results):
    df_list = [r[0] for r in parsed_results]
    meta_list = [r[1] for r in parsed_results]
    merged_df = pd.concat(df_list, axis=0, ignore_index=True)
    merged_df = merged_df.groupby('SampleID', as_index=False).first()
    merged_meta = pd.concat(meta_list, axis=0)
    merged_meta = merged_meta[~merged_meta.index.duplicated(keep='first')]
    return merged_df, merged_meta, None

def align_sample_info(df, info_df, sample_col_name='SampleName'):
    df['SampleID_Clean'] = df['SampleID'].astype(str).apply(lambda x: re.sub(r'[^a-zA-Z0-9]', '', x.lower()))
    info_df['InfoID_Clean'] = info_df[sample_col_name].astype(str).apply(lambda x: re.sub(r'[^a-zA-Z0-9]', '', x.lower()))
    merged = pd.merge(df, info_df, left_on='SampleID_Clean', right_on='InfoID_Clean', how='left')
    merged.drop(columns=['SampleID_Clean', 'InfoID_Clean'], inplace=True, errors='ignore')
    return merged

# ==========================================
# 2. 数据清洗与预处理核心流水线
# ==========================================
def impute_missing_values(df, features, method='knn'):
    df_imp = df.copy()
    df_imp[features] = df_imp[features].replace(0.0, np.nan)
    X = df_imp[features].values
    n_neighbors = min(5, max(1, len(X) - 1))
    
    with np.errstate(all='ignore'):
        if method == 'knn': X_imp = KNNImputer(n_neighbors=n_neighbors).fit_transform(X)
        elif method == 'min': X_imp = np.where(np.isnan(X), np.nanmin(X, axis=0) * 0.5, X)
        elif method == 'mean': X_imp = np.where(np.isnan(X), np.nanmean(X, axis=0), X)
        else: X_imp = np.nan_to_num(X, nan=0.0)
    
    X_imp = np.nan_to_num(X_imp, nan=0.0, posinf=0.0, neginf=0.0)
    df_imp[features] = X_imp
    return df_imp

def normalize_data(df, features, method='pqn'):
    X = df[features].values
    if method == 'pqn':
        ref = np.median(X, axis=0)
        ref[ref == 0] = 1e-8
        quotient = X / ref
        median_quotient = np.median(quotient, axis=1, keepdims=True)
        median_quotient[median_quotient == 0] = 1.0
        X_norm = X / median_quotient
    elif method == 'median':
        m = np.median(X, axis=1, keepdims=True)
        m[m == 0] = 1.0
        X_norm = X / m
    elif method == 'sum':
        s = np.sum(X, axis=1, keepdims=True)
        s[s == 0] = 1.0
        X_norm = X / s
    else: X_norm = X
    
    df_norm = df.copy()
    df_norm[features] = np.nan_to_num(X_norm, nan=0.0, posinf=0.0, neginf=0.0)
    return df_norm

def scale_data(df, features, method='pareto'):
    X = df[features].values
    if method == 'pareto':
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        std[std == 0] = 1.0
        X_scaled = (X - mean) / np.sqrt(std)
    elif method == 'auto':
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        std[std == 0] = 1.0
        X_scaled = (X - mean) / std
    else: X_scaled = X
    
    df_scaled = df.copy()
    df_scaled[features] = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    return df_scaled

def data_cleaning_pipeline(df, group_col, miss_th=0.2, impute_m='knn', norm_m='pqn', do_log=True, scale_m='pareto'):
    features = [c for c in df.columns if c not in ['SampleID', group_col, 'Source_Files'] and pd.api.types.is_numeric_dtype(df[c])]
    
    miss_rates = df[features].isnull().mean()
    keep_feats = miss_rates[miss_rates <= miss_th].index.tolist()
    
    qc_mask = df[group_col].astype(str).str.contains('QC', case=False, na=False) | df['SampleID'].astype(str).str.contains('QC', case=False, na=False)
    if qc_mask.any():
        qc_miss_rates = df.loc[qc_mask, keep_feats].isnull().mean()
        keep_feats = qc_miss_rates[qc_miss_rates <= miss_th].index.tolist()
        
    df_proc = df[['SampleID', group_col] + keep_feats].copy()
    df_proc = impute_missing_values(df_proc, keep_feats, method=impute_m)
    df_proc = normalize_data(df_proc, keep_feats, method=norm_m)
    
    if do_log:
        X = df_proc[keep_feats].values
        X = np.log2(np.clip(X, 0, None) + 1)
        df_proc[keep_feats] = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
    df_proc = scale_data(df_proc, keep_feats, method=scale_m)
    
    variances = df_proc[keep_feats].var()
    keep_feats = variances[variances > 1e-10].index.tolist()
    df_proc = df_proc[['SampleID', group_col] + keep_feats]
    
    return df_proc, keep_feats

# ==========================================
# 3. OPLS-DA 算法核心
# ==========================================
class OPLS_DA:
    def __init__(self, n_components=1):
        self.n_components = n_components
    
    def _clean_matrix(self, matrix):
        mat = np.array(matrix, dtype=np.float64)
        return np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)

    def fit(self, X, y):
        self.X_ = self._clean_matrix(X)
        self.y_ = self._clean_matrix(y)
        
        self.pls = PLSRegression(n_components=self.n_components, scale=False)
        self.pls.fit(self.X_, self.y_)
        
        self.t = self.pls.x_scores_[:, 0]
        self.w = self.pls.x_weights_[:, 0]
        self.p = np.dot(self.X_.T, self.t) / (np.dot(self.t.T, self.t) + 1e-8)
        
        w_ortho = self.p - (np.dot(self.w.T, self.p) / (np.dot(self.w.T, self.w) + 1e-8)) * self.w
        w_ortho_norm = np.linalg.norm(w_ortho)
        w_ortho = w_ortho / w_ortho_norm if w_ortho_norm > 0 else w_ortho
        self.t_ortho = np.dot(self.X_, w_ortho)
        
        self.vip = self._calculate_vip()
        self.p_corr = self._calculate_p_corr()
        return self
        
    def _calculate_vip(self):
        t = np.asarray(self.pls.x_scores_, dtype=float)
        w = np.asarray(self.pls.x_weights_, dtype=float)
        q = np.asarray(self.pls.y_loadings_, dtype=float)
        p, h = w.shape
        
        vips = np.zeros(p)
        s = np.zeros(h)
        for a in range(h):
            t_a = t[:, a]
            q_a = float(q[0, a]) if q.ndim > 1 else float(q[a])
            s[a] = float(np.dot(t_a, t_a) * (q_a ** 2))
            
        total_s = float(np.sum(s))
        if total_s == 0: return vips
            
        for i in range(p):
            val = 0.0
            for a in range(h):
                norm_w = float(np.linalg.norm(w[:, a]))
                if norm_w > 0:
                    weight_a = (float(w[i, a]) / norm_w) ** 2
                    val += float(s[a]) * weight_a
            
            vip_val = float(p) * val / total_s
            vips[i] = np.sqrt(max(0.0, vip_val))
        return vips
        
    def _calculate_p_corr(self):
        p_corr = np.zeros(self.X_.shape[1])
        t_var = np.var(self.t)
        if t_var == 0: return p_corr
        for i in range(self.X_.shape[1]):
            x_i = self.X_[:, i]
            p_corr[i] = np.cov(x_i, self.t)[0, 1] / (np.std(x_i) * np.std(self.t) + 1e-8)
        return p_corr
        
    def permutation_test(self, X, y, n_permutations=100):
        X_clean = self._clean_matrix(X)
        y_clean = self._clean_matrix(y)
        
        corrs = []; r2s = []; q2s = []
        original_r2 = self.pls.score(X_clean, y_clean)
        y_pred = self.pls.predict(X_clean)
        original_q2 = 1 - np.sum((y_clean - y_pred.flatten())**2) / (np.sum((y_clean - np.mean(y_clean))**2) + 1e-8)
        
        for _ in range(n_permutations):
            y_perm = np.random.permutation(y_clean)
            corrs.append(np.abs(np.corrcoef(y_clean, y_perm)[0, 1]))
            pls_perm = PLSRegression(n_components=self.n_components, scale=False)
            pls_perm.fit(X_clean, y_perm)
            r2s.append(pls_perm.score(X_clean, y_perm))
            y_pred_perm = pls_perm.predict(X_clean)
            q2 = 1 - np.sum((y_perm - y_pred_perm.flatten())**2) / (np.sum((y_perm - np.mean(y_perm))**2) + 1e-8)
            q2s.append(q2)
            
        return np.array(corrs), np.array(r2s), np.array(q2s), original_r2, original_q2

# ==========================================
# 4. 极致背景校验的通路富集算法
# ==========================================
def run_pathway_enrichment(sig_metabolites, all_measured_metabolites, custom_db_source=None):
    if custom_db_source is not None:
        try:
            if isinstance(custom_db_source, str): db = pd.read_csv(custom_db_source)
            else: db = pd.read_csv(custom_db_source)
        except: return pd.DataFrame()
    else: return pd.DataFrame()
    
    if 'Pathway' not in db.columns or 'Compounds' not in db.columns: return pd.DataFrame()
    
    def _extract_terms(met_list):
        terms = set()
        for x in met_list:
            parts = str(x).split('|')
            for p in parts:
                c = p.strip().lower()
                if not c or c == 'nan': continue
                c_clean = re.sub(r'_\d+$', '', c)
                terms.add(c_clean)
                terms.add(c) 
                if re.match(r'^c\d{5}$', c):
                    terms.add('cpd:' + c)
        return terms
        
    sig_set = _extract_terms(sig_metabolites)
    bg_set = _extract_terms(all_measured_metabolites)
    N = len(bg_set); K = len(sig_set)
    if N == 0 or K == 0: return pd.DataFrame()
    
    results = []
    for _, row in db.iterrows():
        pw = row['Pathway']; comp_str = str(row['Compounds'])
        if comp_str == 'nan': continue
        
        pw_comps = set([c.lower().strip() for c in comp_str.split(';')])
        M = len(pw_comps) 
        hits = pw_comps.intersection(sig_set)
        k = len(hits)
        
        if k > 0:
            p_val = stats.hypergeom.sf(k - 1, N, M, K)
            expected = (K * M) / N
            enrich_factor = k / expected if expected > 0 else 0
            results.append({
                'Pathway': pw, 'Total_in_Pathway': M, 'Hits': k,
                'P_Value': p_val, 'Enrichment_Factor': enrich_factor,
                'Hit_Metabolites': ", ".join(list(hits))
            })
            
    res_df = pd.DataFrame(results)
    if not res_df.empty:
        res_df = res_df.sort_values('P_Value')
        _, res_df['FDR'], _, _ = statsmodels.stats.multitest.multipletests(res_df['P_Value'], method='fdr_bh')
    return res_df
