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
        np.random.seed(42)
        orig_R2Y, orig_Q2 = self.evaluate(X, y)
        r2_perm, q2_perm, correlations = [], [], []
        
        pls = PLSRegression(n_components=1)
        cv_splits = min(7, len(y))
        kf = KFold(n_splits=cv_splits, shuffle=True, random_state=42)
        
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
    if 'confidence_level' not in df.columns: df['confidence_level'] = 'Unknown'
    
    df['name'] = df['name'].fillna("").astype(str)
    mask_annotated = (df['name'] != "") & (df['name'].str.lower() != "nan")
    clean_names = df['name'].str.split(';', expand=True)[0]
    mz_str = df['mz'].map('{:.4f}'.format).astype(str) if 'mz' in df.columns else ""
    rt_str = df['rt'].map('{:.2f}'.format).astype(str) if 'rt' in df.columns else ""
    unannotated_ids = "m/z" + mz_str + "_RT" + rt_str + "_" + clean_tag
    final_ids = np.where(mask_annotated, clean_names + "_" + clean_tag, unannotated_ids)
    final_ids = make_unique(final_ids)

    kegg_col = next((c for c in df.columns if 'KEGG' in str(c).upper()), None)
    if kegg_col is not None:
        kegg_vals = df[kegg_col].fillna('').astype(str).values
        orig_names = [f"{n} | {k}" if str(k).strip() and str(k).lower() not in ['nan', 'none', ''] else str(n) for n, k in zip(df['name'], kegg_vals)]
    else:
        orig_names = df['name']

    meta_df = pd.DataFrame({
        "Metabolite_ID": final_ids, 
        "Original_Name": orig_names, 
        "Clean_Name": np.where(mask_annotated, clean_names, final_ids), 
        "Confidence_Level": df['confidence_level'], 
        "Is_Annotated": mask_annotated, 
        "Source_File": clean_tag
    })
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
    
    valid_cols = data_df.isnull().mean()[data_df.isnull().mean() <= missing_thresh].index
    data_df = data_df[valid_cols]
    
    if data_df.isnull().sum().sum() > 0:
        if impute_method == 'min': data_df = data_df.fillna(data_df.min() * 0.5)
        elif impute_method == 'mean': data_df = data_df.fillna(data_df.mean())
        elif impute_method == 'KNN': data_df = pd.DataFrame(KNNImputer(n_neighbors=5).fit_transform(data_df), columns=data_df.columns, index=data_df.index)
        else: data_df = data_df.fillna(0)

    data_df = data_df.astype(float)

    if norm_method == 'Sum': data_df = data_df.div(data_df.sum(axis=1).replace(0, 1e-9), axis=0) * data_df.sum(axis=1).mean()
    elif norm_method == 'Median': data_df = data_df.div(data_df.median(axis=1).replace(0, 1e-9), axis=0) * data_df.median(axis=1).mean()
    elif norm_method == 'PQN':
        ref = data_df.median(axis=0); ref[ref <= 0] = 1e-6
        data_df = data_df.div(data_df.div(ref, axis=1).median(axis=1).replace(0, 1e-9), axis=0)

    if log_transform:
        data_df = np.log2(np.clip(data_df, 0, None) + 1)
        
    data_df = data_df.replace([np.inf, -np.inf], np.nan).fillna(0)

    if scale_method != 'None':
        mean = data_df.mean()
        std = data_df.std().replace(0, 1e-9) 
        if scale_method == 'Auto': data_df = (data_df - mean) / std
        elif scale_method == 'Pareto': data_df = (data_df - mean) / np.sqrt(std)

    data_df = data_df.replace([np.inf, -np.inf], np.nan).fillna(0)
    data_df = data_df.loc[:, data_df.var() > 1e-9]
    return pd.concat([meta_df, data_df], axis=1), data_df.columns.tolist()

# ==============================================================================
# 🌟 原版严谨通路富集引擎 (完美兼容 MA 官方无表头与带表头格式，只导出纯净名称)
# ==============================================================================
def run_pathway_enrichment(sig_metabolites, background_metabolites, custom_db_source=None):
    db = pd.DataFrame()
    if custom_db_source is not None:
        try:
            if hasattr(custom_db_source, 'name'): 
                db = pd.read_csv(custom_db_source, header=None)
            elif isinstance(custom_db_source, str) and os.path.exists(custom_db_source): 
                db = pd.read_csv(custom_db_source, header=None)
        except Exception as e:
            print(f"读取库失败: {e}")
            return pd.DataFrame(), pd.DataFrame()
            
    if not db.empty and len(db.columns) >= 2:
        db = db.rename(columns={0: 'Pathway', 1: 'Compounds'})
        if str(db.iloc[0]['Pathway']).strip().lower() == 'pathway':
            db = db.iloc[1:].reset_index(drop=True)
    else:
        return pd.DataFrame(), pd.DataFrame() 
    
    def get_synonyms(full_name):
        syns = set()
        for p in str(full_name).split('|'):
            for sub_p in p.split(';'):
                clean_p = re.sub(r'[^a-z0-9]', '', sub_p.lower())
                if clean_p: syns.add(clean_p)
        return syns

    bg_syns_list = []
    for name in background_metabolites:
        bg_syns_list.append((name, get_synonyms(name)))
        
    sig_names = set(sig_metabolites)
    
    all_db_comps = set()
    for _, row in db.iterrows():
        if pd.notna(row['Compounds']):
            for c in str(row['Compounds']).split(';'):
                clean_c = re.sub(r'[^a-z0-9]', '', c.lower())
                if clean_c: all_db_comps.add(clean_c)
                
    mapped_bg_names = set()
    mapped_bg_syns = set()
    for orig_name, syns in bg_syns_list:
        intersect = syns.intersection(all_db_comps)
        if intersect:
            mapped_bg_names.add(orig_name)
            mapped_bg_syns.update(intersect)
            
    mapped_sig_names = mapped_bg_names.intersection(sig_names)
    
    N = len(mapped_bg_names)
    K_drawn = len(mapped_sig_names)
    
    if N == 0 or K_drawn == 0: return pd.DataFrame(), pd.DataFrame()
    
    results = []
    filtered_db_records = [] 

    for _, row in db.iterrows():
        pw = row['Pathway']
        if pd.isna(row['Compounds']): continue
        
        pw_comps = set([re.sub(r'[^a-z0-9]', '', str(c).lower()) for c in str(row['Compounds']).split(';')])
        pw_detectable = pw_comps.intersection(mapped_bg_syns)
        M = len(pw_detectable)
        
        if M == 0: continue
        
        pw_bg_orig_names = set()
        for orig_name in mapped_bg_names:
            syns = next(s for n, s in bg_syns_list if n == orig_name)
            if syns.intersection(pw_detectable):
                pw_bg_orig_names.add(orig_name)
                
        # 🌟 核心终极修复：切掉 "| 编号"，只保留纯净的名称，彻底对齐 MA demo 格式！
        pw_bg_pure_names = set([str(n).split('|')[0].strip() for n in pw_bg_orig_names])
        
        filtered_db_records.append({
            'Pathway': pw,
            'Compounds': "; ".join(list(pw_bg_pure_names))
        })
        
        hits_orig_names = set()
        for orig_name in mapped_sig_names:
            if orig_name in pw_bg_orig_names:
                hits_orig_names.add(orig_name)
                
        k = len(hits_orig_names)
        if k > 0:
            p_val = hypergeom.sf(k - 1, N, M, K_drawn)
            expected = (K_drawn * M) / N
            enrich_factor = k / expected if expected > 0 else 0
            
            # 为了美观，App 界面的 Hits 显示也剥离掉编号，保持纯净
            hits_pure_names = set([str(n).split('|')[0].strip() for n in hits_orig_names])
            
            results.append({
                'Pathway': pw, 
                'Total_in_Pathway': M, 
                'Hits': k,
                'P_Value': p_val, 
                'Enrichment_Factor': enrich_factor,
                'Hit_Metabolites': ", ".join(list(hits_pure_names))
            })
            
    res_df = pd.DataFrame(results)
    filtered_db_df = pd.DataFrame(filtered_db_records)
    
    if not res_df.empty:
        try:
            from statsmodels.stats.multitest import multipletests
            _, fdr, _, _ = multipletests(res_df['P_Value'], method='fdr_bh')
            res_df['FDR'] = fdr
        except: res_df['FDR'] = res_df['P_Value']
        res_df['-Log10_P'] = -np.log10(res_df['P_Value'].astype(float) + 1e-300)
        res_df = res_df.sort_values('P_Value')
        
    return res_df, filtered_db_df

# ==============================================================================
# 【物理隔离流 B：手动靶向宽表专属解析器与 KEGG API 引擎】
# ==============================================================================
def build_kegg_dictionary(dict_files):
    kegg_mapping = {}
    if not dict_files: return kegg_mapping
    for file in dict_files:
        try:
            file.seek(0)
            if file.name.endswith('.csv'):
                try: df = pd.read_csv(file, engine='pyarrow')
                except: file.seek(0); df = pd.read_csv(file, low_memory=False)
            else: df = pd.read_excel(file)
            
            target_cols = ['name', 'metabolite', '化合物名称', 'peak_name']
            name_col = None
            df_cols_lower = [str(c).lower() for c in df.columns]
            for tc in target_cols:
                if tc in df_cols_lower:
                    name_col = df.columns[df_cols_lower.index(tc)]
                    break
            if not name_col: name_col = df.columns[0]
            
            kegg_col = next((c for c in df.columns if 'KEGG' in str(c).upper()), None)
            if kegg_col:
                for _, row in df.iterrows():
                    k = str(row[kegg_col]).strip()
                    if k and k.lower() not in ['nan', 'none', '']:
                        k_clean = k.split(';')[0].strip() 
                        names_str = str(row[name_col])
                        for n_part in names_str.split(';'):
                            n_clean = n_part.strip().lower()
                            if n_clean and n_clean != 'nan':
                                kegg_mapping[n_clean] = k_clean
        except Exception: pass
    return kegg_mapping

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
                    if k and k.lower() not in ['nan', 'none', '']: 
                        local_kegg_mapping[n.lower()] = k
                    
            metric_cols = [c for c in df.columns if metric_suffix in str(c)]
            if not metric_cols: continue 
                
            sub_df = df[[comp_col] + metric_cols].copy()
            sub_df.rename(columns={comp_col: '__Compound__'}, inplace=True)
            
            def clean_col_name(c):
                c = str(c).replace(metric_suffix, "").strip() 
                c = re.sub(mode_regex, '-', c, flags=re.IGNORECASE) 
                return c.replace('--', '-') 
                
            sub_df.rename(columns={c: clean_col_name(c) for c in metric_cols}, inplace=True)
            for c in sub_df.columns[1:]: sub_df[c] = pd.to_numeric(sub_df[c], errors='coerce')
            sub_df['__mean_resp__'] = sub_df.iloc[:, 1:].mean(axis=1)
            all_dfs.append(sub_df)
            
        if not all_dfs: return None, None, "未找到指定的提取指标！"
            
        combined = pd.concat(all_dfs, ignore_index=True)
        combined = combined.sort_values('__mean_resp__', ascending=False).drop_duplicates(subset=['__Compound__'])
        combined = combined.drop(columns=['__mean_resp__'])
        combined.set_index('__Compound__', inplace=True)
        
        orig_names = []
        for n in combined.index:
            n_lower = str(n).strip().lower()
            mapped_kegg = None
            if n_lower in local_kegg_mapping: mapped_kegg = local_kegg_mapping[n_lower]
            elif n_lower in external_kegg_dict: mapped_kegg = external_kegg_dict[n_lower]
            else:
                for n_part in n_lower.split(';'):
                    n_part_clean = n_part.strip()
                    if n_part_clean in external_kegg_dict:
                        mapped_kegg = external_kegg_dict[n_part_clean]
                        break
            if mapped_kegg: orig_names.append(f"{n} | {mapped_kegg}")
            else: orig_names.append(n)
                
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
    except Exception as e: return None, None, f"手动表格解析失败: {str(e)}"
