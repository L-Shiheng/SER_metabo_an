# ====================
# 通路富集分析核心引擎 (增强版：支持多同义词/分号匹配)
# ====================
def run_pathway_enrichment(sig_metabolites, background_metabolites, custom_db_source=None):
    """
    优先读取外部数据库 (csv/gmt)，如果没找到则使用极小回退库防止崩溃。
    增强功能：支持通过分号分隔的多种化合物名称同时进行模糊匹配。
    """
    from scipy.stats import hypergeom
    raw_pathways = {}
    
    # 辅助清理函数 (去符号，全小写)
    def clean_met_name(name):
        return re.sub(r'[^a-z0-9]', '', str(name).lower())

    # 1. 尝试从本地或上传文件加载完整库
    if custom_db_source is not None:
        try:
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

    if not raw_pathways:
        raw_pathways = {
            "Please upload custom database (请上传或配置完整通路库)": ["glucose", "citrate", "pyruvate"]
        }

    # 构建处理后的库字典
    processed_pathways = {}
    for pw, mets in raw_pathways.items():
        processed_pathways[pw] = set([clean_met_name(m) for m in mets])

    # --- 核心修复：建立 别名(Synonym) -> 主特征名(Feature) 的反向映射字典 ---
    def build_synonym_to_feature_map(met_list_with_semicolons):
        syn2feat = {}
        for full_name in met_list_with_semicolons:
            if pd.isna(full_name) or str(full_name).strip() == "": continue
            parts = str(full_name).split(';')
            # 将分号前的第一个名字作为绘图和统计的唯一代表 (Feature Name)
            feature_name = parts[0].strip() 
            for p in parts:
                cleaned = clean_met_name(p)
                if cleaned:
                    syn2feat[cleaned] = feature_name
        return syn2feat

    sig_syn2feat = build_synonym_to_feature_map(sig_metabolites)
    bg_syn2feat = build_synonym_to_feature_map(background_metabolites)
    
    # 获取独立特征总数 (去重后的物质种类数)
    sig_features = set(sig_syn2feat.values())
    bg_features = set(bg_syn2feat.values())
    
    N = len(bg_features) if len(bg_features) > 0 else 1000 
    n = len(sig_features)
    results = []
    
    for pathway_name, pw_set in processed_pathways.items():
        # 只要库里的物质在这个 Feature 的任何一个别名中出现，就算命中这个 Feature
        K_features = set([bg_syn2feat[m] for m in pw_set if m in bg_syn2feat])
        K = len(K_features)
        if K == 0: K = len(pw_set) # 如果在背景完全未检出，以全库数量兜底防止除零
            
        k_features = set([sig_syn2feat[m] for m in pw_set if m in sig_syn2feat])
        k = len(k_features)
        
        if k > 0:
            expected = (K / N) * n
            enrichment_factor = k / expected if expected > 0 else 0
            
            p_val = hypergeom.sf(k - 1, N, K, n)
            
            results.append({
                "Pathway": pathway_name,
                "Total_in_Pathway": K,
                "Hits": k,
                "Hit_Metabolites": ", ".join(list(k_features)), # 输出整洁的代表名
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
