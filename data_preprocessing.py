import pandas as pd
import numpy as np
import re
from sklearn.impute import KNNImputer
from sklearn.cross_decomposition import PLSRegression
from scipy import stats
import statsmodels.stats.multitest

# ==========================================
# 1. 数据读取与解析模块 (🌟 彻底强化 MetDNA 与手工表的双引擎)
# ==========================================

def parse_metdna_file(file, unique_name, file_type='csv'):
    try:
        if file_type == 'csv': df = pd.read_csv(file)
        else: df = pd.read_excel(file)
        
        # 🌟 防御 1：自动处理 MetDNA 中常见的同名化合物（加后缀去重）
        def make_unique(names):
            seen = set()
            res = []
            for n in names:
                base = str(n).strip()
                new_n = base
                i = 1
                while new_n in seen:
                    new_n = f"{base}_{i}"
                    i += 1
                seen.add(new_n)
                res.append(new_n)
            return res

        if 'Sample' in df.columns and 'Metabolite' in df.columns:
            # Long format 处理
            df['Metabolite'] = make_unique(df['Metabolite'].values)
            df_wide = df.pivot_table(index='Sample', columns='Metabolite', values='Value').reset_index()
            df_wide.rename(columns={'Sample': 'SampleID'}, inplace=True)
            df_wide['Group'] = 'Unknown'
            df_wide['Source_Files'] = unique_name
            
            # 🌟 防御 2：强制数值转换，防止文本 'NA' 破坏矩阵
            for c in df_wide.columns:
                if c not in ['SampleID', 'Group', 'Source_Files']:
                    df_wide[c] = pd.to_numeric(df_wide[c], errors='coerce')
                    
            meta = pd.DataFrame({'Original_Name': df_wide.columns[2:], 'Clean_Name': df_wide.columns[2:], 'Is_Annotated': True}, index=df_wide.columns[2:])
            return df_wide, meta, None
        else:
            # Wide format 处理 (MetDNA 默认格式)
            id_col = df.columns[0]
            df[id_col] = make_unique(df[id_col].values) # 防止同名列名崩溃
            
            sample_cols = [c for c in df.columns if c not in [id_col, 'name', 'mz', 'rt', 'adduct', 'Formula', 'KEGG', 'HMDB']]
            df_t = df.set_index(id_col)[sample_cols].T.reset_index()
            df_t.columns.name = None
            df_t.rename(columns={'index': 'SampleID'}, inplace=True)
            df_t['Group'] = 'Unknown'
            df_t['Source_Files'] = unique_name
            
            # 🌟 防御 2：强制数值转换，防止矩阵被判为 Object 而被过滤
            for c in df_t.columns:
                if c not in ['SampleID', 'Group', 'Source_Files']:
                    df_t[c] = pd.to_numeric(df_t[c], errors='coerce')
            
            meta_idx = df[id_col].values
            meta = pd.DataFrame(index=meta_idx)
            
            kegg_col = next((c for c in df.columns if 'KEGG' in str(c).upper()), None)
            if kegg_col is not None:
                kegg_vals = df[kegg_col].fillna('').astype(str).values
                meta['Original_Name'] = [f"{n} | {k}" if k.strip() and k.lower() != 'nan' else str(n) for n, k in zip(meta_idx, kegg_vals)]
            else:
                meta['Original_Name'] = meta_idx
                
            meta['Clean_Name'] = meta_idx
            meta['Is_Annotated'] = True 
            if kegg_col is not None:
                kegg_mask = df[kegg_col].astype(str).str.strip().str.lower()
                meta['Is_Annotated'] = (kegg_mask != '') & (kegg_mask != 'nan') & (kegg_mask != 'none')
            return df_t, meta, None
    except Exception as e:
        return None, None, str(e)

def parse_manual_targeted_files(file_list, metric_suffix=" : 面积 ", mode_regex=r'-(P|N|POS|NEG|HILIC-P|HILIC-N)-'):
    try:
        all_dfs = []
        kegg_mapping = {} 
        
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
                    if k and k.lower() != 'nan': kegg_mapping[n] = k
                    
            metric_cols = [c for c in df.columns if metric_suffix in str(c)]
            if not metric_cols:
                continue 
                
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
            
        if not all_dfs:
            return None, None, "所有上传的文件中都没有找到您指定的提取指标！请检查后缀拼写（注意空格）。"
            
        combined = pd.concat(all_dfs, ignore_index=True)
        combined = combined.sort_values('__mean_resp__', ascending=False).drop_duplicates(subset=['__Compound__'])
        combined = combined.drop(columns=['__mean_resp__'])
        
        combined.set_index('__Compound__', inplace=True)
        df_t = combined.T
        df_t.index.name = 'SampleID'
        df_t = df_t.reset_index()
        df_t['Group'] = 'Unknown'
        df_t['Source_Files'] = 'Manual_Targeted_Merged'
        
        meta = pd.DataFrame(index=combined.index)
        meta['Clean_Name'] = combined.index
        
        orig_names = []
        for name in combined.index:
            if name in kegg_mapping: orig_names.append(f"{name} | {kegg_mapping[name]}")
            else: orig_names.append(name)
        meta['Original_Name'] = orig_names
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
# 4. 极致背景校验的通路富集算法 (🌟 自动屏蔽重复后缀，确撞库成功)
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
                # 🌟 核心修复：把自动生成的 _1, _2 等后缀去掉，否则会导致撞库失败！
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
