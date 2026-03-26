import pandas as pd
import numpy as np
import re
from sklearn.impute import KNNImputer
from sklearn.cross_decomposition import PLSRegression
from scipy import stats
import statsmodels.stats.multitest

# ==========================================
# 1. 数据读取与解析模块
# ==========================================

def parse_metdna_file(file, unique_name, file_type='csv'):
    try:
        if file_type == 'csv': df = pd.read_csv(file)
        else: df = pd.read_excel(file)
        
        if 'Sample' in df.columns and 'Metabolite' in df.columns:
            df_wide = df.pivot_table(index='Sample', columns='Metabolite', values='Value').reset_index()
            df_wide.rename(columns={'Sample': 'SampleID'}, inplace=True)
            df_wide['Group'] = 'Unknown'
            df_wide['Source_Files'] = unique_name
            meta = pd.DataFrame({'Original_Name': df_wide.columns[2:], 'Clean_Name': df_wide.columns[2:], 'Is_Annotated': True}, index=df_wide.columns[2:])
            return df_wide, meta, None
        else:
            sample_cols = [c for c in df.columns if c not in ['name', 'mz', 'rt', 'adduct', 'Formula', 'KEGG', 'HMDB']]
            df_t = df[sample_cols].set_index(sample_cols[0]).T.reset_index()
            df_t.columns.name = None
            df_t.rename(columns={'index': 'SampleID'}, inplace=True)
            df_t['Group'] = 'Unknown'
            df_t['Source_Files'] = unique_name
            
            meta_idx = df[sample_cols[0]].values
            meta = pd.DataFrame(index=meta_idx)
            meta['Original_Name'] = meta_idx
            meta['Clean_Name'] = meta_idx
            meta['Is_Annotated'] = True 
            if 'KEGG' in df.columns:
                meta['Is_Annotated'] = df['KEGG'].notna() & (df['KEGG'] != '')
            return df_t, meta, None
    except Exception as e:
        return None, None, str(e)

def parse_manual_targeted_files(file_list, metric_suffix=" : 面积 ", mode_regex=r'-(P|N|POS|NEG|HILIC-P|HILIC-N)-'):
    try:
        all_dfs = []
        for file in file_list:
            file.seek(0)
            if file.name.endswith('.csv'): df = pd.read_csv(file)
            else: df = pd.read_excel(file)
            
            comp_col = df.columns[0] 
            metric_cols = [c for c in df.columns if metric_suffix in str(c)]
            if not metric_cols:
                continue 
                
            sub_df = df[[comp_col] + metric_cols].copy()
            
            # 🌟 新增防御：强制统一所有文件第一列的名字，防止 4 个文件拼接时错位！
            sub_df.rename(columns={comp_col: '__Compound__'}, inplace=True)
            
            def clean_col_name(c):
                c = str(c).replace(metric_suffix, "").strip() 
                c = re.sub(mode_regex, '-', c, flags=re.IGNORECASE) 
                return c.replace('--', '-') # 防止出现双横杠
                
            sub_df.rename(columns={c: clean_col_name(c) for c in metric_cols}, inplace=True)
            
            for c in sub_df.columns[1:]:
                sub_df[c] = pd.to_numeric(sub_df[c], errors='coerce')
            sub_df['__mean_resp__'] = sub_df.iloc[:, 1:].mean(axis=1)
            all_dfs.append(sub_df)
            
        if not all_dfs:
            return None, None, "所有上传的文件中都没有找到您指定的提取指标！请检查后缀拼写（注意空格）。"
            
        combined = pd.concat(all_dfs, ignore_index=True)
        # 按照最高响应去重，绝对的生信金标准
        combined = combined.sort_values('__mean_resp__', ascending=False).drop_duplicates(subset=['__Compound__'])
        combined = combined.drop(columns=['__mean_resp__'])
        
        combined.set_index('__Compound__', inplace=True)
        df_t = combined.T
        df_t.index.name = 'SampleID'
        df_t = df_t.reset_index()
        df_t['Group'] = 'Unknown'
        df_t['Source_Files'] = 'Manual_Targeted_Merged'
        
        meta = pd.DataFrame(index=combined.index)
        meta['Original_Name'] = combined.index
        meta['Clean_Name'] = combined.index
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
    
    # 🌟 防御层 1：全局缺失率过滤
    miss_rates = df[features].isnull().mean()
    keep_feats = miss_rates[miss_rates <= miss_th].index.tolist()
    
    # 🌟 防御层 2 (完美保留您的需求)：QC 专属缺失率过滤
    qc_mask = df[group_col].astype(str).str.contains('QC', case=False, na=False) | df['SampleID'].astype(str).str.contains('QC', case=False, na=False)
    if qc_mask.any():
        qc_miss_rates = df.loc[qc_mask, keep_feats].isnull().mean()
        # 只要 QC 中的缺失率超过阈值，立刻把这个化合物剔除
        keep_feats = qc_miss_rates[qc_miss_rates <= miss_th].index.tolist()
        
    df_proc = df[['SampleID', group_col] + keep_feats].copy()
    
    df_proc = impute_missing_values(df_proc, keep_feats, method=impute_m)
    df_proc = normalize_data(df_proc, keep_feats, method=norm_m)
    
    if do_log:
        X = df_proc[keep_feats].values
        X = np.log2(np.clip(X, 0, None) + 1)
        df_proc[keep_feats] = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
    df_proc = scale_data(df_proc, keep_feats, method=scale_m)
    
    # 防御层 3：剔除零方差特征
    variances = df_proc[keep_feats].var()
    keep_feats = variances[variances > 1e-10].index.tolist()
    df_proc = df_proc[['SampleID', group_col] + keep_feats]
    
    return df_proc, keep_feats

# ==========================================
# 3. OPLS-DA 算法核心 (🌟 彻底重构纯数字循环机制，断绝数组碰撞)
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
        # 🌟 强行将矩阵全部拉成 Float 类型，不用 Numpy 高阶乘法，改为纯数字累加
        t = np.asarray(self.pls.x_scores_, dtype=float)
        w = np.asarray(self.pls.x_weights_, dtype=float)
        q = np.asarray(self.pls.y_loadings_, dtype=float)
        p, h = w.shape
        
        vips = np.zeros(p)
        s = np.zeros(h)
        for a in range(h):
            t_a = t[:, a]
            q_a = q[:, a] if q.ndim > 1 else q[a]
            s[a] = np.dot(t_a, t_a) * (q_a ** 2)
            
        total_s = np.sum(s)
        if total_s == 0:
            return vips
            
        # 🌟 最安全的循环：保证内部运算全是 Scalar 纯数字
        for i in range(p):
            val = 0.0
            for a in range(h):
                norm_w = np.linalg.norm(w[:, a])
                if norm_w > 0:
                    weight_a = (w[i, a] / norm_w) ** 2
                    val += s[a] * weight_a
            
            vip_val = p * val / total_s
            vips[i] = np.sqrt(max(0.0, float(vip_val))) # 彻底规避负数开方和数组赋值
            
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
    
    sig_set = set([str(x).lower().strip() for x in sig_metabolites])
    bg_set = set([str(x).lower().strip() for x in all_measured_metabolites])
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
