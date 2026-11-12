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
# SIMCA: OPLS-DA 算法与置换检验 (含正交矩阵剥离 Deflation)
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
        X = np.array(X, dtype=float)
        X = X - np.mean(X, axis=0)
        y = np.array(y, dtype=float).flatten()
        y_centered = y - np.mean(y)
        
        # 1. 初始预测权重 w
        w = np.dot(X.T, y_centered) / (np.dot(y_centered.T, y_centered) + 1e-12)
        w_norm_val = np.linalg.norm(w)
        if w_norm_val > 0: w /= w_norm_val
        
        t = np.dot(X, w) / (np.dot(w.T, w) + 1e-12)
        p = np.dot(X.T, t) / (np.dot(t.T, t) + 1e-12)
        
        # 2. 计算正交权重 w_ortho 与正交得分 t_ortho
        w_ortho = p - (np.dot(w.T, p) / (np.dot(w.T, w) + 1e-12)) * w
        wo_norm_val = np.linalg.norm(w_ortho)
        if wo_norm_val > 0: w_ortho /= wo_norm_val
        
        t_ortho = np.dot(X, w_ortho) / (np.dot(w_ortho.T, w_ortho) + 1e-12)
        p_ortho = np.dot(X.T, t_ortho) / (np.dot(t_ortho.T, t_ortho) + 1e-12)
        
        # 3. 核心 NIPALS OPLS 剥离步骤：从 X 中剔除正交变异，重新提取纯净预测得分 t
        X_deflated = X - np.outer(t_ortho, p_ortho)
        w_pred = np.dot(X_deflated.T, y_centered) / (np.dot(y_centered.T, y_centered) + 1e-12)
        wp_norm_val = np.linalg.norm(w_pred)
        if wp_norm_val > 0: w_pred /= wp_norm_val
        
        t_pred = np.dot(X_deflated, w_pred) / (np.dot(w_pred.T, w_pred) + 1e-12)
        p_pred = np.dot(X_deflated.T, t_pred) / (np.dot(t_pred.T, t_pred) + 1e-12)
        
        self.t = t_pred.flatten()
        self.t_ortho = t_ortho.flatten()
        self.p = p_pred.flatten()
        
        # 4. 计算 S-Plot 的 p(corr) 与 VIP
        p_corr_list = []
        t_std = np.std(self.t)
        for i in range(X.shape[1]):
            col_std = np.std(X_deflated[:, i])
            if col_std > 1e-12 and t_std > 1e-12:
                c = np.corrcoef(X_deflated[:, i], self.t)[0, 1]
                p_corr_list.append(0.0 if np.isnan(c) else c)
            else:
                p_corr_list.append(0.0)
        self.p_corr = np.array(p_corr_list)
        
        w_final = (w_pred / (np.linalg.norm(w_pred) + 1e-12)).flatten()
        self.vip = np.sqrt(len(w_final) * (w_final ** 2))
        return self

    def evaluate(self, X, y, n_splits=7):
        n_samples = len(y)
        cv_splits = min(n_splits, n_samples)
        if cv_splits < 2: return 0, 0
            
        pls = PLSRegression(n_components=min(2, X.shape[1], n_samples - 1))
        pls.fit(X, y)
        y_pred_fit = pls.predict(X).flatten()
        self.R2Y = r2_score(y, y_pred_fit)
        
        kf = KFold(n_splits=cv_splits, shuffle=True, random_state=42)
        y_cv = cross_val_predict(pls, X, y, cv=kf).flatten()
        self.Q2 = r2_score(y, y_cv)
        return self.R2Y, self.Q2

    def permutation_test(self, X, y, n_permutations=100):
        np.random.seed(42)
        orig_R2Y, orig_Q2 = self.evaluate(X, y)
        r2_perm, q2_perm, correlations = [], [], []
        
        n_comp = min(2, X.shape[1], len(y) - 1)
        pls = PLSRegression(n_components=n_comp)
        cv_splits = min(7, len(y))
        kf = KFold(n_splits=cv_splits, shuffle=True, random_state=42)
        
        for _ in range(n_permutations):
            y_shuffled = np.random.permutation(y)
            corr = np.abs(np.corrcoef(y, y_shuffled)[0, 1])
            correlations.append(0.0 if np.isnan(corr) else corr)
            
            pls.fit(X, y_shuffled)
            r2_perm.append(r2_score(y_shuffled, pls.predict(X).flatten()))
            q2_perm.append(r2_score(y_shuffled, cross_val_predict(pls, X, y_shuffled, cv=kf).flatten()))
            
        return np.array(correlations), np.array(r2_perm), np.array(q2_perm), orig_R2Y, orig_Q2

# ====================
# 通用数据清洗管线 (支持同时返回未缩放的 Log2 矩阵供单变量统计使用)
# ====================
def data_cleaning_pipeline(df, group_col, missing_thresh=0.2, impute_method='KNN', norm_method='PQN', log_transform=True, scale_method='Pareto', return_unscaled=False):
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in [group_col, 'SampleID', 'Source_Files']]
    meta_cols = [c for c in df.columns if c not in numeric_cols]
    data_df = df[numeric_cols].copy()
    meta_df = df[meta_cols].copy()
    
    # 1. 缺失率过滤 (将 <= 0 或 NaN 视为质谱未检出缺失值进行统计)
    na_mask = data_df.isnull() | (data_df <= 0)
    valid_cols = na_mask.mean()[na_mask.mean() <= missing_thresh].index
    data_df = data_df[valid_cols].mask(data_df[valid_cols] <= 0, np.nan)
    
    # 2. 缺失值填充
    if data_df.isnull().sum().sum() > 0:
        non_empty_cols = data_df.columns[data_df.notna().any()].tolist()
        data_df = data_df[non_empty_cols]
        if not data_df.empty:
            if impute_method == 'min':
                data_df = data_df.fillna(data_df.min() * 0.5).fillna(1e-6)
            elif impute_method == 'mean':
                data_df = data_df.fillna(data_df.mean()).fillna(1e-6)
            elif impute_method == 'KNN':
                data_df = pd.DataFrame(
                    KNNImputer(n_neighbors=min(5, len(data_df))).fit_transform(data_df),
                    columns=data_df.columns, index=data_df.index
                )
            else:
                data_df = data_df.fillna(1e-6)

    data_df = data_df.astype(float)

    # 3. 样本归一化 (消除进样体积与电离效率误差)
    if norm_method == 'Sum':
        row_sums = data_df.sum(axis=1).replace(0, 1e-9)
        data_df = data_df.div(row_sums, axis=0) * row_sums.mean()
    elif norm_method == 'Median':
        row_meds = data_df.median(axis=1).replace(0, 1e-9)
        data_df = data_df.div(row_meds, axis=0) * row_meds.mean()
    elif norm_method == 'PQN':
        ref = data_df.median(axis=0)
        ref[ref <= 0] = 1e-6
        quotients = data_df.div(ref, axis=1)
        factors = quotients.median(axis=1).replace(0, 1e-9)
        data_df = data_df.div(factors, axis=0)

    # 4. 生成供单变量统计 (Log2_FC 与 T 检验) 使用的纯净未缩放 Log2 矩阵
    unscaled_log2_df = np.log2(np.clip(data_df, 0, None) + 1.0)
    unscaled_log2_df = unscaled_log2_df.replace([np.inf, -np.inf], np.nan).fillna(0)

    if log_transform:
        data_df = unscaled_log2_df.copy()
    else:
        data_df = data_df.replace([np.inf, -np.inf], np.nan).fillna(0)

    # 5. 特征缩放 (供 PCA, OPLS-DA, 热图等使用)
    if scale_method != 'None':
        mean = data_df.mean()
        std = data_df.std().replace(0, 1e-9) 
        if scale_method == 'Auto': data_df = (data_df - mean) / std
        elif scale_method == 'Pareto': data_df = (data_df - mean) / np.sqrt(std)

    data_df = data_df.replace([np.inf, -np.inf], np.nan).fillna(0)
    valid_var_cols = data_df.columns[data_df.var() > 1e-9].tolist()
    data_df = data_df[valid_var_cols]
    unscaled_log2_df = unscaled_log2_df[valid_var_cols]
    
    proc_full = pd.concat([meta_df, data_df], axis=1)
    if return_unscaled:
        unscaled_full = pd.concat([meta_df, unscaled_log2_df], axis=1)
        return proc_full, valid_var_cols, unscaled_full
    return proc_full, valid_var_cols

def align_sample_info(data_df, info_df, sample_col_name=None):
    target_col = sample_col_name if sample_col_name and sample_col_name in info_df.columns else info_df.columns[0]
    info_map = {re.sub(r'[^a-zA-Z0-9]', '', str(r[target_col])).lower(): r for _, r in info_df.iterrows()}
    aligned_data = [info_map.get(re.sub(r'[^a-zA-Z0-9]', '', str(sid)).lower(), pd.Series([np.nan]*len(info_df.columns), index=info_df.columns)) for sid in data_df['SampleID']]
    aligned_df = pd.DataFrame(aligned_data)
    aligned_df.index = data_df.index 
    return aligned_df

def make_unique(series):
    seen = set(); result = []
    for item in series:
        new_item = item; counter = 1
        while new_item in seen:
            new_item = f"{item}_{counter}"; counter += 1
        seen.add(new_item); result.append(new_item)
    return result

def read_file_robust(file_buffer):
    file_buffer.seek(0)
    if file_buffer.name.endswith('.csv'):
        encodings = ['utf-8', 'utf-8-sig', 'gbk', 'latin1']
        for enc in encodings:
            try:
                file_buffer.seek(0)
                df = pd.read_csv(file_buffer, encoding=enc, header=None, low_memory=False)
                return df
            except Exception: continue
        file_buffer.seek(0)
        return pd.read_csv(file_buffer, encoding='utf-8', header=None, errors='ignore')
    else:
        return pd.read_excel(file_buffer, header=None)

def build_kegg_dictionary(dict_files):
    kegg_mapping = {}
    if not dict_files: return kegg_mapping
    for file in dict_files:
        try:
            df = read_file_robust(file)
            if df.empty: continue
            df.columns = df.iloc[0].astype(str).tolist()
            df = df.iloc[1:].reset_index(drop=True)
            
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

# ==============================================================================
# 🚀 引擎 A：MA 极简单表解析器 
# ==============================================================================
def parse_universal_single_table(file_list, external_kegg_dict=None):
    if external_kegg_dict is None: external_kegg_dict = {}
    try:
        combined_records = {}; sample_groups = {}
        for file in file_list:
            df = read_file_robust(file)
            if len(df) < 3: continue 
            
            samples = df.iloc[0, 1:].astype(str).tolist()
            groups = df.iloc[1, 1:].astype(str).tolist()
            for s, g in zip(samples, groups):
                s_clean = s.strip()
                if s_clean and s_clean.lower() != 'nan':
                    sample_groups[s_clean] = g.strip()
                    
            data_df = df.iloc[2:].copy()
            comp_col = data_df.columns[0]
            data_df = data_df.set_index(comp_col)
            data_df.columns = df.iloc[0, 1:].astype(str).str.strip().tolist()
            
            valid_samples = [s for s in data_df.columns if s and s.lower() != 'nan']
            data_df = data_df[valid_samples]
            for c in data_df.columns:
                data_df[c] = pd.to_numeric(data_df[c], errors='coerce').fillna(0)
                
            data_df['__mean_resp__'] = data_df.mean(axis=1)
            for comp_name, row in data_df.iterrows():
                c_name = str(comp_name).strip()
                if not c_name or c_name.lower() == 'nan': continue
                m_val = row['__mean_resp__']
                if c_name not in combined_records or m_val > combined_records[c_name][0]:
                    combined_records[c_name] = (m_val, row.drop('__mean_resp__'))
                    
        if not combined_records: return None, None, "未能解析出有效数据。请确保符合MA单表格式。"
            
        final_data = []; final_comp_names = []
        for c_name, (m_val, row_series) in combined_records.items():
            final_comp_names.append(c_name); final_data.append(row_series)
            
        combined_df = pd.DataFrame(final_data, index=final_comp_names)
        
        orig_names = []; kegg_list = []
        for n in combined_df.index:
            n_lower = str(n).strip().lower(); mapped_kegg = ""
            if n_lower in external_kegg_dict: mapped_kegg = external_kegg_dict[n_lower]
            else:
                for n_part in n_lower.split(';'):
                    if n_part.strip() in external_kegg_dict:
                        mapped_kegg = external_kegg_dict[n_part.strip()]; break
            orig_names.append(f"{n} | {mapped_kegg}" if mapped_kegg else str(n))
            kegg_list.append(mapped_kegg)
                
        combined_df.index = orig_names
        df_t = combined_df.T
        df_t.index.name = 'SampleID'
        df_t = df_t.reset_index()
        df_t['Group'] = df_t['SampleID'].map(sample_groups).fillna('Unknown')
        df_t['Source_Files'] = 'MA_Single'
        
        meta = pd.DataFrame(index=combined_df.index)
        meta['Clean_Name'] = [str(n).split(' | ')[0].split(';')[0].strip() for n in combined_df.index]
        meta['Original_Name'] = combined_df.index
        meta['KEGG_ID'] = kegg_list
        meta['Source_Mode'] = 'MA_Single'
        meta['Is_Annotated'] = True 
        return df_t, meta, None
    except Exception as e: return None, None, f"通用表格解析失败: {str(e)}"

# ==============================================================================
# 🚀 引擎 B：MRM 拟靶向后缀解析器 
# ==============================================================================
def parse_manual_targeted_files(file_list, metric_suffix=" : 面积", mode_regex=r'-(P|N|RP-P|RP-N|HILIC-P|HILIC-N|POS|NEG)-', external_kegg_dict=None):
    if external_kegg_dict is None: external_kegg_dict = {}
    try:
        all_dfs = []; local_kegg_mapping = {} 
        for file in file_list:
            df = read_file_robust(file)
            if df.empty: continue
            df.columns = df.iloc[0].astype(str).tolist()
            df = df.iloc[1:].reset_index(drop=True)
            
            comp_col = df.columns[0] 
            kegg_col = next((c for c in df.columns if 'KEGG' in str(c).upper()), None)
            if kegg_col:
                for _, row in df.iterrows():
                    n = str(row[comp_col]).strip(); k = str(row[kegg_col]).strip()
                    if k and k.lower() not in ['nan', 'none', '']:
                        local_kegg_mapping[n.lower()] = k.split(';')[0].strip()
                    
            metric_cols = [c for c in df.columns if metric_suffix in str(c)]
            if not metric_cols: continue 
                
            sub_df = df[[comp_col] + metric_cols].copy()
            sub_df.rename(columns={comp_col: '__Compound__'}, inplace=True)
            
            def clean_col_name(c):
                c = str(c).replace(metric_suffix, "").strip() 
                c = re.sub(mode_regex, '-', c, flags=re.IGNORECASE) 
                return c.replace('--', '-') 
                
            sub_df.rename(columns={c: clean_col_name(c) for c in metric_cols}, inplace=True)
            for c in sub_df.columns[1:]: sub_df[c] = pd.to_numeric(sub_df[c], errors='coerce').fillna(0)
            sub_df['__mean_resp__'] = sub_df.iloc[:, 1:].mean(axis=1)
            all_dfs.append(sub_df)
            
        if not all_dfs: return None, None, f"未找到后缀为 '{metric_suffix}' 的数据列！请检查表头。"
            
        combined = pd.concat(all_dfs, ignore_index=True)
        combined = combined.sort_values('__mean_resp__', ascending=False).drop_duplicates(subset=['__Compound__'])
        combined = combined.drop(columns=['__mean_resp__'])
        combined.set_index('__Compound__', inplace=True)
        
        orig_names = []; kegg_list = []
        for n in combined.index:
            n_lower = str(n).strip().lower(); mapped_kegg = ""
            if n_lower in local_kegg_mapping: mapped_kegg = local_kegg_mapping[n_lower]
            elif n_lower in external_kegg_dict: mapped_kegg = external_kegg_dict[n_lower]
            else:
                for n_part in n_lower.split(';'):
                    if n_part.strip() in external_kegg_dict:
                        mapped_kegg = external_kegg_dict[n_part.strip()]; break
            orig_names.append(f"{n} | {mapped_kegg}" if mapped_kegg else str(n))
            kegg_list.append(mapped_kegg)
                
        combined.index = orig_names
        df_t = combined.T
        df_t.index.name = 'SampleID'
        df_t = df_t.reset_index()
        df_t['Group'] = 'Unknown' 
        df_t['Source_Files'] = 'MRM_Targeted'
        
        meta = pd.DataFrame(index=combined.index)
        meta['Clean_Name'] = [str(n).split(' | ')[0].split(';')[0].strip() for n in combined.index]
        meta['Original_Name'] = combined.index
        meta['KEGG_ID'] = kegg_list
        meta['Source_Mode'] = 'MRM_Targeted'
        meta['Is_Annotated'] = True 
        return df_t, meta, None
    except Exception as e: return None, None, f"MRM 拟靶向表格解析失败: {str(e)}"

# ==============================================================================
# 🚀 引擎 C：MetDNA 原生管线提取器 (支持自定义模式标签 RP+/RP-/HILIC+/HILIC-)
# ==============================================================================
def rank_confidence(conf_str):
    s = str(conf_str).upper().strip()
    if s in ['NAN', 'NONE', '', 'NULL']: return 99
    digits = re.findall(r'\d+', s)
    if digits: return int(digits[0])
    if 'A' in s: return 1
    if 'B' in s: return 2
    if 'C' in s: return 3
    return 99

def parse_metdna_file(file_buffer, file_name, valid_samples=None, dedup_within_file=True, custom_mode_tag=None):
    try:
        df = read_file_robust(file_buffer)
        if df.empty: return None, None, "表格为空"
        df.columns = df.iloc[0].astype(str).tolist()
        df = df.iloc[1:].reset_index(drop=True)
    except Exception as e: return None, None, f"读取失败: {str(e)}"

    known_meta = {'peak_name', 'mz', 'rt', 'id', 'id_zhulab', 'name', 'formula', 
                  'confidence_level', 'smiles', 'inchikey', 'isotope', 'adduct', 
                  'total_score', 'mz_error', 'rt_error_abs', 'rt_error_rela', 
                  'ms2_score', 'iden_score', 'iden_type', 'peak_group_id', 
                  'base_peak', 'num_peaks', 'cons_formula_pred', 'id_kegg', 
                  'id_hmdb', 'id_metacyc', 'stereo_isomer_id', 'stereo_isomer_name'}
    
    sample_cols = [c for c in df.columns if str(c).strip().lower() not in known_meta and str(c).strip() != '']
    if valid_samples and len(valid_samples) > 0:
        valid_clean = [re.sub(r'[^a-zA-Z0-9]', '', str(s)).lower() for s in valid_samples]
        sample_cols = [c for c in sample_cols if re.sub(r'[^a-zA-Z0-9]', '', str(c)).lower() in valid_clean]
            
    if not sample_cols: 
        return None, None, "未在数据表中找到与 Info 表匹配的样本列名。"

    # 优先使用用户在侧边栏指定的模式名称 (如 RP+, RP-, HILIC+, HILIC-)，彻底杜绝同名文件互相覆盖
    if custom_mode_tag and str(custom_mode_tag).strip():
        clean_tag = str(custom_mode_tag).strip()
    else:
        base_raw = os.path.splitext(os.path.basename(file_name))[0]
        clean_tag = re.sub(r'[^a-zA-Z0-9_\-\.\+]', '_', base_raw)
    
    safe_id_suffix = re.sub(r'[^a-zA-Z0-9_\-]', '_', clean_tag.replace('+', 'POS').replace('-', 'NEG'))
    
    if 'name' not in df.columns: df['name'] = ""
    if 'confidence_level' not in df.columns: df['confidence_level'] = 'Unknown'
    if 'total_score' not in df.columns: df['total_score'] = 0
    if 'peak_name' not in df.columns: 
        mz_vals = df['mz'] if 'mz' in df.columns else pd.Series([""] * len(df))
        rt_vals = df['rt'] if 'rt' in df.columns else df.index.astype(str)
        df['peak_name'] = [f"M{mz}_RT{rt}" for mz, rt in zip(mz_vals, rt_vals)]
    
    final_ids = df['peak_name'].astype(str).str.strip() + "_" + safe_id_suffix
    final_ids = make_unique(final_ids)
    
    df['name'] = df['name'].fillna("").astype(str).str.strip()
    mask_annotated = (df['name'] != "") & (df['name'].str.lower() != "nan")
    clean_names = df['name'].str.split(';', expand=True)[0].str.strip()
    
    kegg_col = next((c for c in df.columns if 'KEGG' in str(c).upper()), None)
    if kegg_col:
        kegg_ids = df[kegg_col].fillna('').astype(str).apply(lambda s: s.split(';')[0].strip() if s.lower() not in ['nan', 'none'] else '')
    else:
        kegg_ids = pd.Series([""] * len(df))

    df_data = df[sample_cols].copy()
    for c in df_data.columns: df_data[c] = pd.to_numeric(df_data[c], errors='coerce').fillna(0)
    mean_areas = df_data.mean(axis=1).values

    meta_df = pd.DataFrame({
        "Peak_Name": final_ids, 
        "Original_Name": np.where(mask_annotated, clean_names, final_ids), 
        "Clean_Name": np.where(mask_annotated, clean_names, final_ids), 
        "Confidence_Level": df['confidence_level'], 
        "Total_Score": pd.to_numeric(df['total_score'], errors='coerce').fillna(0),
        "Mean_Area": mean_areas,
        "Is_Annotated": mask_annotated, 
        "Source_Mode": clean_tag,
        "KEGG_ID": kegg_ids
    })
    meta_df.set_index('Peak_Name', inplace=True)
    df_data.index = meta_df.index

    # 单文件（单模式）内优先对同名已注释化合物执行三级漏斗去重 (剔除加合离子/源内裂解冗余)
    if dedup_within_file:
        meta_df['__rank__'] = [rank_confidence(c) for c in meta_df['Confidence_Level']]
        anno_meta = meta_df[meta_df['Is_Annotated']].sort_values(
            by=['__rank__', 'Total_Score', 'Mean_Area'], ascending=[True, False, False]
        )
        dedup_anno_ids = anno_meta[~anno_meta['Clean_Name'].duplicated(keep='first')].index.tolist()
        unanno_ids = meta_df[~meta_df['Is_Annotated']].index.tolist()
        keep_ids = dedup_anno_ids + unanno_ids
        meta_df = meta_df.loc[keep_ids].drop(columns=['__rank__'])
        df_data = df_data.loc[keep_ids]

    df_transposed = df_data.T
    df_transposed.reset_index(inplace=True)
    df_transposed.rename(columns={'index': 'SampleID'}, inplace=True)
    df_transposed['Source_Files'] = clean_tag
    df_transposed['Group'] = 'Unknown'
    
    return df_transposed, meta_df, None

def merge_multiple_dfs(results_list):
    """跨模式三级漏斗优选合并：将多个模式矩阵按最优特征去重合并为一张唯一代谢物总表"""
    if not results_list: return None, None, "无数据"
    
    best_features = {} 
    sample_source_map = {}
    all_detected_modes = {}
    
    for file_idx, (df, meta, fname) in enumerate(results_list):
        current_tag = str(fname) if fname else (str(df['Source_Files'].iloc[0]) if 'Source_Files' in df.columns and not df.empty else f"Mode_{file_idx+1}")
        for sid in df['SampleID']:
            if sid not in sample_source_map: sample_source_map[sid] = set()
            sample_source_map[sid].add(current_tag)
            
        numeric_df = df.select_dtypes(include=[np.number])
        intensities = numeric_df.mean(axis=0) 
        
        for peak_name in numeric_df.columns:
            if peak_name not in meta.index: continue
            m_row = meta.loc[peak_name]
            
            clean_name = str(m_row['Clean_Name']).strip()
            conf = m_row['Confidence_Level']
            score = float(m_row.get('Total_Score', 0))
            area = float(m_row.get('Mean_Area', intensities.get(peak_name, 0)))
            
            if clean_name not in all_detected_modes: all_detected_modes[clean_name] = []
            if current_tag not in all_detected_modes[clean_name]:
                all_detected_modes[clean_name].append(current_tag)
            
            rank_val = rank_confidence(conf)
            current_tuple = (rank_val, -score, -area)
            
            if clean_name not in best_features:
                best_features[clean_name] = (file_idx, peak_name, current_tuple)
            else:
                best_tuple = best_features[clean_name][2]
                if current_tuple < best_tuple:
                    best_features[clean_name] = (file_idx, peak_name, current_tuple)
    
    files_features_to_keep = {i: [] for i in range(len(results_list))}
    for c_name, (f_idx, p_name, _) in best_features.items(): 
        files_features_to_keep[f_idx].append(p_name)
        
    dfs_to_concat = []
    for i, (df, meta, fname) in enumerate(results_list):
        df_idx = df.set_index('SampleID')
        df_clean = df_idx.drop(columns=['Group', 'Source_Files'], errors='ignore')
        keep_c = [c for c in files_features_to_keep[i] if c in df_clean.columns]
        dfs_to_concat.append(df_clean[keep_c])
        
    full_df = pd.concat(dfs_to_concat, axis=1, join='outer').fillna(0)
    full_df.insert(0, 'Group', 'Unknown')
    full_df.reset_index(inplace=True)
    full_df.rename(columns={'index': 'SampleID'}, inplace=True)
    full_df['Source_Files'] = full_df['SampleID'].apply(lambda sid: "; ".join(sorted(list(sample_source_map.get(sid, set())))))
    
    final_ids = [fid for f_list in files_features_to_keep.values() for fid in f_list]
    
    merged_meta = pd.concat([res[1] for res in results_list])
    merged_meta = merged_meta[~merged_meta.index.duplicated(keep='first')]
    merged_meta = merged_meta.loc[final_ids].copy()
    
    rename_map = {fid: str(merged_meta.loc[fid, 'Clean_Name']) for fid in final_ids}
    full_df.rename(columns=rename_map, inplace=True)
    
    merged_meta.reset_index(inplace=True)
    merged_meta['Metabolite_ID'] = merged_meta['Peak_Name'].map(lambda x: rename_map.get(x, x))
    merged_meta['Detected_Modes'] = merged_meta['Clean_Name'].map(lambda cn: ", ".join(all_detected_modes.get(str(cn).strip(), [])))
    merged_meta.set_index('Metabolite_ID', inplace=True)
    
    return full_df, merged_meta, None

# ==============================================================================
# 🌟 严谨通路富集引擎 (统一唯一 C-number 量纲 + 支持全库/实测双背景切换)
# ==============================================================================
OVERVIEW_PATHWAYS = {
    'metabolic pathways',
    'biosynthesis of secondary metabolites',
    'microbial metabolism in diverse environments',
    'carbon metabolism',
    '2-oxocarboxylic acid metabolism',
    'fatty acid metabolism',
    'biosynthesis of amino acids',
    'biosynthesis of cofactors',
    'degradation of aromatic compounds',
    'overview of biosynthetic pathways'
}

def run_pathway_enrichment(sig_metabolites, background_metabolites, custom_db_source=None, bg_mode="reference"):
    db = pd.DataFrame()
    if custom_db_source is not None:
        try:
            if hasattr(custom_db_source, 'name'):
                custom_db_source.seek(0)
                db = pd.read_csv(custom_db_source, header=None)
            elif isinstance(custom_db_source, str) and os.path.exists(custom_db_source):
                db = pd.read_csv(custom_db_source, header=None)
        except Exception:
            return pd.DataFrame(), pd.DataFrame()
            
    if not db.empty and len(db.columns) >= 2:
        db = db.rename(columns={0: 'Pathway', 1: 'Compounds'})
        if str(db.iloc[0]['Pathway']).strip().lower() == 'pathway':
            db = db.iloc[1:].reset_index(drop=True)
    else:
        return pd.DataFrame(), pd.DataFrame() 

    parsed_pathways = []
    all_db_comps = set()
    for _, row in db.iterrows():
        pw_raw = str(row['Pathway']).strip()
        pw_clean = re.sub(r'\s+-\s+[^-]+$', '', pw_raw).strip()
        if pd.isna(row['Compounds']) or pw_clean.lower() in OVERVIEW_PATHWAYS:
            continue
        pw_comps = set()
        for c in str(row['Compounds']).split(';'):
            c_clean = re.sub(r'[^a-z0-9]', '', c.lower())
            if c_clean:
                pw_comps.add(c_clean)
        if len(pw_comps) >= 2:
            parsed_pathways.append((pw_clean, pw_comps))
            all_db_comps.update(pw_comps)

    if not all_db_comps:
        return pd.DataFrame(), pd.DataFrame()

    def extract_primary_id_and_name(full_str):
        parts = str(full_str).split('|')
        disp_name = parts[0].split(';')[0].strip()
        if len(parts) > 1 and parts[1].strip():
            kegg_c = re.sub(r'[^a-z0-9]', '', parts[1].split(';')[0].lower())
            if kegg_c in all_db_comps:
                return disp_name, kegg_c
        name_clean = re.sub(r'[^a-z0-9]', '', disp_name.lower())
        if name_clean in all_db_comps:
            return disp_name, name_clean
        return disp_name, None

    exp_id_to_name = {}
    for bg_item in background_metabolites:
        d_name, cid = extract_primary_id_and_name(bg_item)
        if cid and cid not in exp_id_to_name:
            exp_id_to_name[cid] = d_name

    sig_id_to_name = {}
    for sig_item in sig_metabolites:
        d_name, cid = extract_primary_id_and_name(sig_item)
        if cid:
            sig_id_to_name[cid] = d_name
            if cid not in exp_id_to_name:
                exp_id_to_name[cid] = d_name

    use_ref_bg = (bg_mode == "reference") or (len(exp_id_to_name) <= len(sig_id_to_name))
    universe_ids = all_db_comps if use_ref_bg else set(exp_id_to_name.keys())
    
    sig_valid_ids = set(sig_id_to_name.keys()).intersection(universe_ids)
    N = len(universe_ids)
    K_drawn = len(sig_valid_ids)
    
    if N == 0 or K_drawn == 0:
        return pd.DataFrame(), pd.DataFrame()
    
    results, filtered_db_records = [], []
    for pw_name, pw_comps in parsed_pathways:
        exp_in_pw = pw_comps.intersection(set(exp_id_to_name.keys()))
        if len(exp_in_pw) >= 1:
            filtered_db_records.append({
                'Pathway': pw_name,
                'Compounds': "; ".join(sorted(set(exp_id_to_name[cid] for cid in exp_in_pw)))
            })

        pw_universe = pw_comps.intersection(universe_ids)
        M = len(pw_universe)
        if M < 2:
            continue
            
        hit_ids = sig_valid_ids.intersection(pw_universe)
        k = len(hit_ids)
        if k == 0:
            continue
            
        p_val = float(hypergeom.sf(k - 1, N, M, K_drawn))
        expected = (K_drawn * M) / float(N)
        enrich_factor = k / expected if expected > 0 else 0.0
        hit_names = sorted(set(sig_id_to_name[cid] for cid in hit_ids))
        
        results.append({
            'Pathway': pw_name,
            'Total_in_Pathway': M,
            'Hits': k,
            'P_Value': np.clip(p_val, 1e-300, 1.0),
            'Enrichment_Factor': enrich_factor,
            'Hit_Metabolites': ", ".join(hit_names)
        })
            
    res_df = pd.DataFrame(results)
    filtered_db_df = pd.DataFrame(filtered_db_records)
    
    if not res_df.empty:
        try:
            from statsmodels.stats.multitest import multipletests
            _, fdr, _, _ = multipletests(res_df['P_Value'], method='fdr_bh')
            res_df['FDR'] = fdr
        except Exception:
            res_df['FDR'] = res_df['P_Value']
            
        res_df['-Log10_P'] = -np.log10(np.clip(res_df['P_Value'].astype(float), 1e-300, 1.0))
        res_df = res_df.sort_values(['P_Value', 'Enrichment_Factor'], ascending=[True, False]).reset_index(drop=True)
        
    return res_df, filtered_db_df
