import pandas as pd
import numpy as np
import re
from sklearn.impute import KNNImputer
from sklearn.cross_decomposition import PLSRegression
from scipy import stats

# ==========================================
# 1. 数据读取与解析模块
# ==========================================

def parse_metdna_file(file, unique_name, file_type='csv'):
    """解析传统的 MetDNA 结果文件"""
    try:
        if file_type == 'csv': df = pd.read_csv(file)
        else: df = pd.read_excel(file)
        
        if 'Sample' in df.columns and 'Metabolite' in df.columns:
            # Long format
            df_wide = df.pivot_table(index='Sample', columns='Metabolite', values='Value').reset_index()
            df_wide.rename(columns={'Sample': 'SampleID'}, inplace=True)
            df_wide['Group'] = 'Unknown'
            df_wide['Source_Files'] = unique_name
            meta = pd.DataFrame({'Original_Name': df_wide.columns[2:], 'Clean_Name': df_wide.columns[2:], 'Is_Annotated': True}, index=df_wide.columns[2:])
            return df_wide, meta, None
        else:
            # Wide format (MetDNA Default)
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
    """解析手动靶向宽表 (岛津MRM等多文件融合与智能去重)"""
    try:
        all_dfs = []
        for file in file_list:
            file.seek(0)
            # 兼容 csv 和 excel
            if file.name.endswith('.csv'): df = pd.read_csv(file)
            else: df = pd.read_excel(file)
            
            comp_col = df.columns[0] # 默认第一列是化合物名称
            # 1. 精准提取包含指标后缀的列 (如 " : 面积 ")
            metric_cols = [c for c in df.columns if metric_suffix in str(c)]
            if not metric_cols:
                continue # 如果这个文件里没有找到该指标，跳过
                
            sub_df = df[[comp_col] + metric_cols].copy()
            
            # 2. 智能清洗表头
            def clean_col_name(c):
                c = str(c).replace(metric_suffix, "").strip() # 砍掉 " : 面积 "
                c = re.sub(mode_regex, '-', c, flags=re.IGNORECASE) # 替换 "-P-" 为 "-"
                return c
                
            sub_df.rename(columns={c: clean_col_name(c) for c in metric_cols}, inplace=True)
            
            # 3. 计算该文件内每个化合物的“平均响应强度”用于去重竞争
            for c in sub_df.columns[1:]:
                sub_df[c] = pd.to_numeric(sub_df[c], errors='coerce')
            sub_df['__mean_resp__'] = sub_df.iloc[:, 1:].mean(axis=1)
            all_dfs.append(sub_df)
            
        if not all_dfs:
            return None, None, "所有上传的文件中都没有找到您指定的提取指标！请检查后缀拼写（注意空格）。"
            
        # 4. 纵向拼缝与“最高响应”去重 (方案A)
        combined = pd.concat(all_dfs, ignore_index=True)
        # 按照响应强度降序排列，然后去重（保留第一个，即强度最大的那个）
        combined = combined.sort_values('__mean_resp__', ascending=False).drop_duplicates(subset=[combined.columns[0]])
        combined = combined.drop(columns=['__mean_resp__'])
        
        # 5. 矩阵翻转为系统标准格式
        combined.set_index(combined.columns[0], inplace=True)
        df_t = combined.T
        df_t.index.name = 'SampleID'
        df_t = df_t.reset_index()
        df_t['Group'] = 'Unknown'
        df_t['Source_Files'] = 'Manual_Targeted_Merged'
        
        # 构建 meta 信息
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
    if method == 'knn': X_imp = KNNImputer(n_neighbors=5).fit_transform(X)
    elif method == 'min': X_imp = np.where(np.isnan(X), np.nanmin(X, axis=0) * 0.5, X)
    elif method == 'mean': X_imp = np.where(np.isnan(X), np.nanmean(X, axis=0), X)
    else: X_imp = np.nan_to_num(X, nan=0.0)
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
    df_norm[features] = X_norm
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
    df_scaled[features] = X_scaled
    return df_scaled

def data_cleaning_pipeline(df, group_col, miss_th=0.2, impute_m='knn', norm_m='pqn', do_log=True, scale_m='pareto'):
    features = [c for c in df.columns if c not in ['SampleID', group_col, 'Source_Files'] and pd.api.types.is_numeric_dtype(df[c])]
    
    # Missing Value Filtering
    miss_rates = df[features].isnull().mean()
    keep_feats = miss_rates[miss_rates <= miss_th].index.tolist()
    df_proc = df[['SampleID', group_col] + keep_feats].copy()
    
    # Imputation
    df_proc = impute_missing_values(df_proc, keep_feats, method=impute_m)
    
    # Normalization
    df_proc = normalize_data(df_proc, keep_feats, method=norm_m)
    
    # Log Transformation
    if do_log:
        X = df_proc[keep_feats].values
        X = np.log2(X + 1)
        df_proc[keep_feats] = X
        
    # Scaling
    df_proc = scale_data(df_proc, keep_feats, method=scale_m)
    return df_proc, keep_feats

# ==========================================
# 3. OPLS-DA 算法核心
# ==========================================
class OPLS_DA:
    def __init__(self, n_components=1):
        self.n_components = n_components
    
    def fit(self, X, y):
        self.X_ = np.array(X); self.y_ = np.array(y)
        self.pls = PLSRegression(n_components=self.n_components, scale=False)
        self.pls.fit(self.X_, self.y_)
        self.t = self.pls.x_scores_[:, 0]
        self.w = self.pls.x_weights_[:, 0]
        self.p = np.dot(self.X_.T, self.t) / np.dot(self.t.T, self.t)
        
        w_ortho = self.p - (np.dot(self.w.T, self.p) / np.dot(self.w.T, self.w)) * self.w
        w_ortho = w_ortho / np.linalg.norm(w_ortho)
        self.t_ortho = np.dot(self.X_, w_ortho)
        
        self.vip = self._calculate_vip()
        self.p_corr = self._calculate_p_corr()
        return self
        
    def _calculate_vip(self):
        t = self.pls.x_scores_; w = self.pls.x_weights_; q = self.pls.y_loadings_
        p, h = w.shape; vips = np.zeros((p,))
        s = np.diag(t.T @ t @ q.T @ q).reshape(h, -1)
        total_s = np.sum(s)
        for i in range(p):
            weight = np.array([(w[i, j] / np.linalg.norm(w[:, j]))**2 for j in range(h)])
            vips[i] = np.sqrt(p * (s.T @ weight) / total_s)
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
        corrs = []; r2s = []; q2s = []
        original_r2 = self.pls.score(X, y)
        y_pred = self.pls.predict(X)
        original_q2 = 1 - np.sum((y - y_pred.flatten())**2) / np.sum((y - np.mean(y))**2)
        
        for _ in range(n_permutations):
            y_perm = np.random.permutation(y)
            corrs.append(np.abs(np.corrcoef(y, y_perm)[0, 1]))
            pls_perm = PLSRegression(n_components=self.n_components, scale=False)
            pls_perm.fit(X, y_perm)
            r2s.append(pls_perm.score(X, y_perm))
            y_pred_perm = pls_perm.predict(X)
            q2 = 1 - np.sum((y_perm - y_pred_perm.flatten())**2) / np.sum((y_perm - np.mean(y_perm))**2)
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
        _, res_df['FDR'], _, _ = statsmodels.stats.multitest.multipletests(res_df['P_Value'], method='fdr_bh') if 'statsmodels' in globals() else (None, res_df['P_Value'], None, None)
    return res_df
