import pandas as pd
import numpy as np
import re
import os

def parse_metdna_file(file_buffer, file_name, file_type='csv'):
    """
    解析 MetDNA 导出文件 (v3.2)
    """
    try:
        if file_type == 'csv':
            df = pd.read_csv(file_buffer)
        else:
            df = pd.read_excel(file_buffer)
    except Exception as e:
        return None, None, f"读取失败: {str(e)}"

    # 2. 智能识别样本列
    known_meta_cols = [
        'peak_name', 'mz', 'rt', 'id', 'id_zhulab', 'name', 'formula', 
        'confidence_level', 'smiles', 'inchikey', 'isotope', 'adduct', 
        'total_score', 'mz_error', 'rt_error_abs', 'rt_error_rela', 
        'ms2_score', 'iden_score', 'iden_type', 'peak_group_id', 
        'base_peak', 'num_peaks', 'cons_formula_pred', 'id_kegg', 
        'id_hmdb', 'id_metacyc', 'stereo_isomer_id', 'stereo_isomer_name'
    ]
    
    sample_cols = []
    for col in df.columns:
        if col in known_meta_cols: continue
        try:
            numeric_series = pd.to_numeric(df[col], errors='coerce')
            if numeric_series.notna().mean() > 0.5:
                sample_cols.append(col)
        except: pass
            
    if not sample_cols:
        return None, None, "未找到样本数据列。"

    # 3. 构建元数据
    file_tag = os.path.splitext(os.path.basename(file_name))[0]
    file_tag = re.sub(r'[^a-zA-Z0-9]', '_', file_tag)
    
    if 'name' not in df.columns: df['name'] = np.nan
    if 'confidence_level' not in df.columns: df['confidence_level'] = 'Unknown'

    def process_row(row):
        raw_name = str(row['name']).strip() if pd.notna(row['name']) else ""
        is_annotated = (raw_name != "") and (raw_name.lower() != "nan")
        
        if is_annotated:
            clean_name = raw_name.split(';')[0]
            unique_id = f"{clean_name}_{file_tag}"
        else:
            unique_id = f"m/z{row['mz']:.4f}_RT{row['rt']:.2f}_{file_tag}"
            clean_name = unique_id
            
        return {
            "Metabolite_ID": unique_id,
            "Original_Name": raw_name,
            "Clean_Name": clean_name,
            "Confidence_Level": row.get('confidence_level', 'Unknown'),
            "Is_Annotated": is_annotated,
            "Source_File": file_tag
        }

    meta_info = df.apply(process_row, axis=1)
    meta_df = pd.DataFrame(meta_info.tolist())
    meta_df['Metabolite_ID'] = make_unique(meta_df['Metabolite_ID'])
    meta_df.set_index('Metabolite_ID', inplace=True)
    
    # 4. 提取数据
    df_data = df[sample_cols].copy()
    df_data.index = meta_df.index
    df_transposed = df_data.T
    
    # 5. 生成 SampleID
    df_transposed.reset_index(inplace=True)
    df_transposed.rename(columns={'index': 'SampleID'}, inplace=True)
    
    # 默认猜测分组 (会被 sample info 覆盖)
    def guess_group(s):
        s_no = re.sub(r'\d+$', '', str(s))
        return s_no.rstrip('._-') or "Unknown"

    df_transposed.insert(1, 'Group', df_transposed['SampleID'].apply(guess_group))
    
    return df_transposed, meta_df, None

def merge_multiple_dfs(results_list):
    """
    合并多文件逻辑 (智能保留最大峰面积)
    """
    if not results_list: return None, None, "无数据"
    
    # 1. 竞选最佳代谢物
    best_features = {}
    for file_idx, (df, meta, fname) in enumerate(results_list):
        numeric_df = df.select_dtypes(include=[np.number])
        intensities = numeric_df.sum(axis=0)
        
        for feat_id in numeric_df.columns:
            try:
                clean_name = meta.loc[feat_id, 'Clean_Name']
            except KeyError: continue
            
            curr_score = intensities.get(feat_id, 0)
            
            if clean_name not in best_features:
                best_features[clean_name] = (file_idx, feat_id, curr_score)
            else:
                prev_idx, prev_id, prev_score = best_features[clean_name]
                if curr_score > prev_score:
                    best_features[clean_name] = (file_idx, feat_id, curr_score)
    
    # 2. 构建保留列表
    files_features_to_keep = {i: [] for i in range(len(results_list))}
    for c_name, (f_idx, f_id, score) in best_features.items():
        files_features_to_keep[f_idx].append(f_id)
        
    # 3. 拼接
    dfs_to_concat = []
    base_group_series = None
    
    for i, (df, meta, fname) in enumerate(results_list):
        if 'SampleID' in df.columns: df = df.set_index('SampleID')
        
        # 暂时移除 Group，最后再加
        if 'Group' in df.columns:
            if base_group_series is None: base_group_series = df['Group']
            df = df.drop(columns=['Group'])
            
        cols_to_keep = files_features_to_keep[i]
        valid_cols = [c for c in cols_to_keep if c in df.columns]
        dfs_to_concat.append(df[valid_cols])
        
    try:
        full_df = pd.concat(dfs_to_concat, axis=1, join='outer')
    except Exception as e:
        return None, None, f"合并出错: {str(e)}"
    
    full_df.fillna(0, inplace=True)
    
    if base_group_series is not None:
        aligned_group = base_group_series.reindex(full_df.index).fillna('Unknown')
        full_df.insert(0, 'Group', aligned_group)
    else:
        full_df.insert(0, 'Group', 'Unknown')
        
    full_df.reset_index(inplace=True)
    full_df.rename(columns={'index': 'SampleID'}, inplace=True)
    
    # 4. 整理元数据
    final_ids = [fid for f_list in files_features_to_keep.values() for fid in f_list]
    all_meta = pd.concat([res[1] for res in results_list])
    merged_meta = all_meta.loc[final_ids]
    
    return full_df, merged_meta, None

def apply_sample_info(df, info_file):
    """
    应用样本信息表覆盖默认分组
    支持模糊匹配 (忽略 . - _ 和大小写)
    """
    try:
        if info_file.name.endswith('.csv'):
            info_df = pd.read_csv(info_file)
        else:
            info_df = pd.read_excel(info_file)
    except Exception as e:
        return df, f"样本表读取失败: {e}"
        
    # 1. 识别列名
    # 寻找 Sample 和 Group 列
    sample_col = None
    group_col = None
    
    cols_lower = [c.lower() for c in info_df.columns]
    
    # 找 Sample 列
    if 'sample.name' in cols_lower: sample_col = info_df.columns[cols_lower.index('sample.name')]
    elif 'sample' in cols_lower: sample_col = info_df.columns[cols_lower.index('sample')]
    elif 'sampleid' in cols_lower: sample_col = info_df.columns[cols_lower.index('sampleid')]
    
    # 找 Group 列
    if 'group' in cols_lower: group_col = info_df.columns[cols_lower.index('group')]
    elif 'class' in cols_lower: group_col = info_df.columns[cols_lower.index('class')]
    
    if not sample_col or not group_col:
        return df, "在信息表中未找到 Sample 或 Group 列，请检查表头。"
        
    # 2. 构建映射字典 (Normalizer)
    # MetDNA 可能会把 '-' 变成 '.'，这里我们把两者都统一成纯字母数字来匹配
    def normalize_name(s):
        return re.sub(r'[^a-zA-Z0-9]', '', str(s)).lower()
        
    # 构建: normalized_name -> group
    info_map = {}
    for _, row in info_df.iterrows():
        key = normalize_name(row[sample_col])
        info_map[key] = row[group_col]
        
    # 3. 应用映射
    mapped_groups = []
    match_count = 0
    
    for _, row in df.iterrows():
        sid = row['SampleID']
        norm_sid = normalize_name(sid)
        
        if norm_sid in info_map:
            mapped_groups.append(info_map[norm_sid])
            match_count += 1
        else:
            # 没匹配上就保留原来的 (或者标记 Unknown)
            mapped_groups.append(row.get('Group', 'Unknown'))
            
    df['Group'] = mapped_groups
    
    return df, f"成功匹配 {match_count}/{len(df)} 个样本的分组信息。"

def make_unique(series):
    seen = set()
    result = []
    for item in series:
        new_item = item
        counter = 1
        while new_item in seen:
            new_item = f"{item}_{counter}"
            counter += 1
        seen.add(new_item)
        result.append(new_item)
    return result

# --- 清洗管道 (不变) ---
def data_cleaning_pipeline(df, group_col, missing_thresh=0.5, impute_method='min', 
                           norm_method='None', log_transform=True, scale_method='None'):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if group_col in numeric_cols: numeric_cols.remove(group_col)
    meta_cols = [c for c in df.columns if c not in numeric_cols]
    data_df = df[numeric_cols].copy()
    meta_df = df[meta_cols].copy()
    
    missing_ratio = data_df.isnull().mean()
    cols_to_keep = missing_ratio[missing_ratio <= missing_thresh].index
    data_df = data_df[cols_to_keep]
    
    if data_df.isnull().sum().sum() > 0:
        if impute_method == 'min': data_df = data_df.fillna(data_df.min() * 0.5)
        elif impute_method == 'mean': data_df = data_df.fillna(data_df.mean())
        elif impute_method == 'median': data_df = data_df.fillna(data_df.median())
        elif impute_method == 'zero': data_df = data_df.fillna(0)
        data_df = data_df.fillna(0)

    if norm_method == 'Sum':
        data_df = data_df.div(data_df.sum(axis=1), axis=0) * data_df.sum(axis=1).mean()
    elif norm_method == 'Median':
        data_df = data_df.div(data_df.median(axis=1), axis=0) * data_df.median(axis=1).mean()

    if log_transform:
        if not (data_df < 0).any().any(): data_df = np.log2(data_df + 1)

    if scale_method == 'Auto':
        data_df = (data_df - data_df.mean()) / data_df.std()
    elif scale_method == 'Pareto':
        data_df = (data_df - data_df.mean()) / np.sqrt(data_df.std())

    var_mask = data_df.var() > 1e-9
    data_df = data_df.loc[:, var_mask]
    return pd.concat([meta_df, data_df], axis=1), data_df.columns.tolist()