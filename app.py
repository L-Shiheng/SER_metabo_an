import streamlit as st
import pandas as pd
import numpy as np
import os
import gc
import datetime
import re
import io
import base64
import requests
import csv 
import traceback
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ==========================================
# 0. 品牌更新与 UI 配置 (MetaFlow Studio)
# ==========================================
st.set_page_config(page_title="MetaFlow Studio", page_icon="🧬", layout="wide")

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'PingFang SC', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

try:
    from data_preprocessing import (
        data_cleaning_pipeline, 
        parse_metdna_file, 
        parse_universal_single_table, 
        parse_manual_targeted_files,
        merge_multiple_dfs, 
        align_sample_info, 
        OPLS_DA, 
        run_pathway_enrichment, 
        build_kegg_dictionary
    )
    from stats_utils import run_pairwise_statistics
    from plot_utils import update_layout_square, get_ellipse_coordinates, plot_nomogram
    from report_generator import generate_offline_html, generate_ai_prompt
except ImportError as e:
    st.error(f"❌ 严重错误：未找到依赖文件。详情: {e}")
    st.stop()

# 恢复 SERRF 模块检测
try:
    from serrf_module import serrf_normalization
except ImportError:
    pass

COLOR_PALETTE = {'Up': '#CD0000', 'Down': '#00008B', 'NS': '#E0E0E0'} 
GROUP_COLORS = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F', '#8491B4', '#91D1C2', '#DC0000', '#7E6148', '#B09C85']

st.markdown("""
<style>
    .block-container {padding-top: 2rem !important; padding-bottom: 3rem !important;}
    h1, h2, h3, div, p {font-family: 'Arial', sans-serif; color: #2c3e50;}
    div[data-testid="stForm"] button {width: 100%; background-color: #ff4b4b; color: white; font-weight: bold;}
    .logo-container {display: flex; align-items: center; margin-bottom: 20px;}
    .logo-text {font-size: 2.5rem; font-weight: 800; color: #2c3e50; margin-left: 15px; letter-spacing: -1px;}
    .logo-badge {background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); color: white; padding: 4px 12px; border-radius: 20px; font-size: 1rem; margin-left: 10px; vertical-align: middle;}
</style>
""", unsafe_allow_html=True)

# 挂载 Logo (如果您本地有图片，可将下面的 emoji 替换为 st.image("logo.png", width=60))
st.markdown("""
<div class="logo-container">
    <span style="font-size: 3rem;">♾️</span>
    <span class="logo-text">MetaFlow Studio</span>
    <span class="logo-badge">Pro</span>
</div>
""", unsafe_allow_html=True)

# ==========================================
# 1. 极简侧边栏数据流控制台 (三大引擎独立切分)
# ==========================================
if 'raw_df' not in st.session_state: st.session_state.raw_df = None
if 'feature_meta' not in st.session_state: st.session_state.feature_meta = None
if 'data_loaded' not in st.session_state: st.session_state.data_loaded = False
if 'qc_report' not in st.session_state: st.session_state.qc_report = {}

with st.sidebar:
    st.header("🛠️ 数据控制台")
    
    data_source = st.radio(
        "选择数据流模式", 
        ["1. MA 标准单表 (自带分组)", "2. 拟靶向 MRM 宽表 (需后缀)", "3. MetDNA 原始宽表"], 
        index=0,
        help="【MA 单表】：第一行样本，第二行分组，无需Info表。\n【MRM 宽表】：仪器原始导出表，列名包含特定后缀。\n【MetDNA】：原始结果表。"
    )
    
    info_df = None; candidate_samples = []; user_sample_col = None; user_group_col = None
    excluded_samples = []
    
    if data_source in ["2. 拟靶向 MRM 宽表 (需后缀)", "3. MetDNA 原始宽表"]:
        st.markdown("#### 1. 上传 Sample Info")
        sample_info_file = st.file_uploader("Info表格 (.csv/.xlsx) [必填]", type=["csv", "xlsx"], key="info")
        if sample_info_file:
            try:
                sample_info_file.seek(0)
                info_df = pd.read_csv(sample_info_file) if sample_info_file.name.endswith('.csv') else pd.read_excel(sample_info_file)
                cols = list(info_df.columns); cols_lower = [c.lower() for c in cols]
                idx_sample = next((cols_lower.index(kw) for kw in ['sample.name', 'sample_name', 'sample', 'name', 'id'] if kw in cols_lower), 0)
                idx_group = next((cols_lower.index(kw) for kw in ['group', 'class', 'type', 'condition'] if kw in cols_lower), 1 if len(cols) > 1 else 0)
                c1, c2 = st.columns(2)
                user_sample_col = c1.selectbox("样本列", cols, index=idx_sample)
                user_group_col = c2.selectbox("分组列", cols, index=idx_group)
                if user_sample_col: candidate_samples = info_df[user_sample_col].astype(str).unique().tolist()
                st.caption(f"✅ 已加载 {len(info_df)} 行")
            except Exception as e: st.error(f"Info 读取失败: {e}")
            
        excluded_samples = st.multiselect("2. 样本剔除 (黑名单)", options=candidate_samples, default=[], help="从分析中彻底移除异常样本。")
    else:
        st.markdown("#### 1. 上传数据矩阵")
        st.caption("格式：Row1 样本名，Row2 分组，Row3起为数据。")
        ex_str = st.text_input("2. 样本剔除 (选填)", help="输入需剔除的样本名，用英文逗号分隔，如: S1,S5")
        if ex_str: excluded_samples = [s.strip() for s in ex_str.split(',') if s.strip()]

    st.markdown("#### 3. KEGG 通路配置")
    species = st.selectbox("物种背景", ["Human (Homo sapiens)", "Mouse (Mus musculus)", "Rat (Rattus norvegicus)", "General (所有物种)"], index=0)
    species_code = {"Human (Homo sapiens)": "hsa", "Mouse (Mus musculus)": "mmu", "Rat (Rattus norvegicus)": "rno", "General (所有物种)": "map"}[species]
    db_filename = f"kegg_{species_code}.csv"
    custom_pathway_file = st.file_uploader("自定义通路库 (.csv)", type=["csv", "gmt"], key="pathway_db")
    
    if st.button(f"🔄 同步 {species_code} 通路库", use_container_width=True) or not os.path.exists(db_filename):
        with st.spinner(f"正在连接 KEGG API 拉取库..."):
            try:
                pw_res = requests.get(f"http://rest.kegg.jp/list/pathway/{species_code}")
                pw_dict = {re.sub(r'^[a-z]+', '', p.split('\t')[0].replace('path:', '')): p.split('\t')[1] for p in pw_res.text.strip().split('\n') if p}
                link_res = requests.get("http://rest.kegg.jp/link/cpd/pathway")
                pw_cpd_map = {}
                for line in link_res.text.strip().split('\n'):
                    if line and line.startswith('path:map'):
                        pw_num, cpd = line.split('\t')[0].replace('path:map', ''), line.split('\t')[1].replace('cpd:', '')
                        if pw_num not in pw_cpd_map: pw_cpd_map[pw_num] = []
                        pw_cpd_map[pw_num].append(cpd)
                pd.DataFrame([{'Pathway': name, 'Compounds': ';'.join(pw_cpd_map[pw_num])} for pw_num, name in pw_dict.items() if pw_num in pw_cpd_map]).to_csv(db_filename, index=False)
                st.toast(f"✅ 库同步成功！")
            except Exception as e: st.error("网络请求失败")

    st.markdown("#### 4. 上传分析数据")
    feature_scope = "全部特征"
    dict_files = None
    
    if data_source == "1. MA 标准单表 (自带分组)":
        dict_files = st.file_uploader("关联 MetDNA 字典 (可选)", type=["csv", "xlsx"], accept_multiple_files=True, key="dict_files")
        uploaded_files = st.file_uploader("上传 MA 格式单表 (支持多选自动合并)", type=["csv", "xlsx"], accept_multiple_files=True, key="data")
    elif data_source == "2. 拟靶向 MRM 宽表 (需后缀)":
        suffix = st.text_input("提取指标后缀", value=" : 面积", help="代码将按此后缀寻找数据列。")
        dict_files = st.file_uploader("关联 MetDNA 字典 (可选)", type=["csv", "xlsx"], accept_multiple_files=True, key="dict_files")
        uploaded_files = st.file_uploader("上传 MRM 宽表 (支持多选最大响应保留)", type=["csv", "xlsx"], accept_multiple_files=True, key="data")
    else:
        feature_scope = st.radio("特征范围", ["仅已注释特征", "全部特征"], index=0)
        uploaded_files = st.file_uploader("上传 MetDNA 结果表", type=["csv", "xlsx"], accept_multiple_files=True, key="data")
        
    start_process = st.container().button("📥 加载数据矩阵", use_container_width=True, type="primary")

# ==========================================
# 2. 核心路由与解析引擎调用
# ==========================================
if start_process:
    if 'analysis_res' in st.session_state: del st.session_state['analysis_res']
        
    if not uploaded_files: 
        st.error("请先上传主数据文件！")
    else:
        progress_bar = st.progress(0); status_text = st.empty()
        
        # 引擎 A：MA 单表
        if data_source == "1. MA 标准单表 (自带分组)":
            with st.spinner("正在启动 MA 矩阵解析引擎..."):
                ext_dict = build_kegg_dictionary(dict_files) if dict_files else {}
                df_t, meta, err = parse_universal_single_table(uploaded_files, external_kegg_dict=ext_dict)
                if err: st.error(err)
                else:
                    st.session_state.raw_df, st.session_state.feature_meta = df_t, meta
                    st.session_state.data_loaded = True
                    st.success("✅ MA 单表合并加载成功！")

        # 引擎 B：MRM 拟靶向
        elif data_source == "2. 拟靶向 MRM 宽表 (需后缀)":
            if info_df is None: st.error("⚠️ 必须先上传 Sample Info 表格以映射分组！")
            else:
                with st.spinner("正在启动 MRM 拟靶向缝合引擎..."):
                    ext_dict = build_kegg_dictionary(dict_files) if dict_files else {}
                    df_t, meta, err = parse_manual_targeted_files(uploaded_files, metric_suffix=suffix, external_kegg_dict=ext_dict)
                    if err: st.error(err)
                    else:
                        info_aligned = align_sample_info(df_t, info_df, sample_col_name=user_sample_col)
                        if user_group_col and user_group_col in info_aligned.columns: 
                            df_t['Group'] = info_aligned[user_group_col].fillna('Unknown').values
                        st.session_state.raw_df, st.session_state.feature_meta = df_t, meta
                        st.session_state.data_loaded = True
                        st.success("✅ 拟靶向多表缝合（最大响应保留）加载成功！")

        # 引擎 C：MetDNA 原始
        else:
            if info_df is None: st.error("⚠️ 必须先上传 Sample Info 表格以映射分组！")
            else:
                with st.spinner("正在启动 MetDNA 解析引擎..."):
                    parsed_results = []
                    for i, file in enumerate(uploaded_files):
                        unique_name = f"{os.path.splitext(file.name)[0]}_{i}{os.path.splitext(file.name)[1]}"
                        df_t, meta, err = parse_metdna_file(file, unique_name)
                        if not err: parsed_results.append((df_t, meta, unique_name))
                        progress_bar.progress((i + 1) / len(uploaded_files))
                    
                    if parsed_results:
                        raw_df, meta, err = merge_multiple_dfs(parsed_results)
                        if err: st.error(err)
                        else:
                            if feature_scope.startswith("仅已注释"):
                                anno_ids = meta[meta['Is_Annotated'] == True].index
                                keep_cols = ['SampleID', 'Group', 'Source_Files'] + [c for c in raw_df.columns if c in anno_ids]
                                raw_df = raw_df[keep_cols]
                                meta = meta.loc[meta.index.isin(raw_df.columns)]
                            info_aligned = align_sample_info(raw_df, info_df, sample_col_name=user_sample_col)
                            if user_group_col and user_group_col in info_aligned.columns: 
                                raw_df['Group'] = info_aligned[user_group_col].fillna('Unknown').values
                            st.session_state.raw_df, st.session_state.feature_meta = raw_df, meta
                            st.session_state.data_loaded = True
                            st.success("✅ MetDNA 原始数据加载成功！")

        # 统一执行剔除逻辑并强制刷新
        if st.session_state.data_loaded:
            df = st.session_state.raw_df
            if excluded_samples:
                ex_fps = set([re.sub(r'[^a-z0-9]', '', str(s).strip().lower()) for s in excluded_samples])
                st.session_state.raw_df = df[~df['SampleID'].astype(str).apply(lambda s: re.sub(r'[^a-z0-9]', '', str(s).strip().lower())).isin(ex_fps)]
            st.rerun()

# ==========================================
# 3. 统计与可视化展示区
# ==========================================
if not st.session_state.data_loaded:
    st.info("👈 请在左侧控制台按对应数据格式上传数据。")
    st.stop()

raw_df = st.session_state.raw_df
st.info(f"数据总览: {len(raw_df)} 样本 x {len(raw_df.columns)-3} 特征")
st.download_button("📥 导出合并矩阵", raw_df.to_csv(index=False).encode('utf-8'), "MetaFlow_Matrix.csv", "text/csv")
st.divider()

with st.form(key='analysis_form'):
    st.markdown("### ⚙️ 分析参数配置")
    non_num = raw_df.select_dtypes(exclude=[np.number]).columns.tolist()
    group_col = st.selectbox("分组列", non_num, index=non_num.index('Group') if 'Group' in non_num else 0)
    
    with st.expander("数据预处理", expanded=False):
        c_p1, c_p2 = st.columns(2)
        miss_th = c_p1.slider("缺失率过滤", 0.0, 1.0, 0.20)
        impute_m = c_p2.selectbox("缺失值填充", ["KNN", "min", "mean", "zero"], index=0)
        c_p3, c_p4 = st.columns(2)
        norm_m = c_p3.selectbox("样本归一化", ["PQN", "Median", "Sum", "None"], index=0)
        scale_m = c_p4.selectbox("特征缩放", ["Pareto", "Auto", "None"], index=0)
        do_log = st.checkbox("Log2 对数转化", value=True)

    with st.expander("图表设置", expanded=False):
        c_t1, c_t2, c_t3 = st.columns(3)
        vip_show_num = c_t1.slider("VIP 展示数", 10, 50, 25)
        nomo_num = c_t2.slider("列线图标志物数", 2, 8, 4)
        pw_show_num = c_t3.slider("网络图通路数", 5, 30, 15)

    cur_grps = sorted(raw_df[group_col].astype(str).unique())
    sel_grps = st.multiselect("选择对比组 (限 2 组)", cur_grps, default=cur_grps[:2] if len(cur_grps)>=2 else cur_grps)
    c1, c2, c3, c4 = st.columns(4)
    case = c1.selectbox("Case 组", list(sel_grps), index=0 if sel_grps else None)
    ctrl = c2.selectbox("Control 组", list(sel_grps), index=1 if len(sel_grps)>1 else 0)
    p_th = c3.number_input("P-value 阈值", value=0.05, step=0.01)
    fc_th = c4.number_input("Log2 FC 阈值", value=0.58, step=0.10)
    submit_button = st.form_submit_button(label='🚀 执行分析')

# ==========================================
# 4. 执行核心分析计算 (加入深度 Bug 拦截器)
# ==========================================
if submit_button:
    if len(sel_grps) != 2: 
        st.error("⚠️ 必须且只能选择 2 个组！")
        st.stop()

    with st.spinner("正在运行分析引擎..."):
        try:
            # === 开始执行可能会报错的代码 ===
            raw_df = st.session_state.raw_df
            meta = st.session_state.feature_meta
            
            df_proc, feats = data_cleaning_pipeline(
                raw_df, group_col, missing_thresh=miss_th, 
                impute_method=impute_m, norm_method=norm_m, 
                log_transform=do_log, scale_method=scale_m
            )
            df_sub = df_proc[df_proc[group_col].isin(sel_grps)].copy()
            if len(df_sub) < 4: raise ValueError(f"所选两组总样本数太少 ({len(df_sub)}个)，无法建模。")
            if len(feats) < 2: raise ValueError("清洗后剩余特征太少，无法建模。请调高缺失率过滤阈值。")
     
            stats_df = run_pairwise_statistics(df_sub, group_col, case, ctrl, feats)
            if meta is not None and 'Clean_Name' in meta.columns and 'Original_Name' in meta.columns: 
                stats_df = stats_df.merge(meta[['Clean_Name', 'Original_Name']], left_on='Metabolite', right_index=True, how='left')
                stats_df['Name'] = stats_df['Clean_Name'].fillna(stats_df['Metabolite'])
                stats_df['Search_Name'] = stats_df['Original_Name'].fillna(stats_df['Metabolite'])
            else:
                stats_df['Name'] = stats_df['Search_Name'] = stats_df['Metabolite']
                
            stats_df['Sig'] = 'NS'
            stats_df.loc[(stats_df['P_Value']<p_th)&(stats_df['Log2_FC']>fc_th), 'Sig']='Up'
            stats_df.loc[(stats_df['P_Value']<p_th)&(stats_df['Log2_FC']<-fc_th), 'Sig']='Down'

            y_binary = np.where(df_sub[group_col] == case, 1, -1)
            X_matrix = df_sub[feats].values
            
            opls = OPLS_DA()
            opls.fit(X_matrix, y_binary)
            corrs, r2_perm, q2_perm, R2Y, Q2 = opls.permutation_test(X_matrix, y_binary, n_permutations=100)
            m_q2, b_q2 = np.polyfit(corrs, q2_perm, 1) if len(corrs)>0 else (0,0)
            m_r2, b_r2 = np.polyfit(corrs, r2_perm, 1) if len(corrs)>0 else (0,0)

            vip_df = pd.DataFrame({'Metabolite': feats, 'VIP': opls.vip, 'p_corr': opls.p_corr})
            stats_df = stats_df.merge(vip_df, on='Metabolite')
            
            stats_df['Is_Biomarker'] = (stats_df['VIP'] > 1.0) & (stats_df['P_Value'] < p_th) & (stats_df['Log2_FC'].abs() > fc_th)
            out_df = stats_df[stats_df['Is_Biomarker']].sort_values('VIP', ascending=False)

            opls_score_df = pd.DataFrame({'t1 (Predictive)': opls.t, 't_ortho (Orthogonal)': opls.t_ortho, 'Group': df_sub[group_col].values})
            fig_opls = px.scatter(opls_score_df, x='t1 (Predictive)', y='t_ortho (Orthogonal)', color='Group', symbol='Group', color_discrete_sequence=GROUP_COLORS)
            for i, g in enumerate(list(sel_grps)):
                sub_grp = opls_score_df[opls_score_df['Group']==g]
                if len(sub_grp)>=3:
                    el_x, el_y = get_ellipse_coordinates(sub_grp['t1 (Predictive)'], sub_grp['t_ortho (Orthogonal)'])
                    if el_x is not None: fig_opls.add_trace(go.Scatter(x=el_x, y=el_y, mode='lines', line=dict(color=GROUP_COLORS[i%len(GROUP_COLORS)], width=2, dash='dash'), showlegend=False, hoverinfo='skip'))
            fig_opls.update_traces(marker=dict(size=14, line=dict(width=1, color='black'), opacity=0.9))
            fig_opls.add_hline(y=0, line_dash="dash", line_color="gray"); fig_opls.add_vline(x=0, line_dash="dash", line_color="gray")
            fig_opls = update_layout_square(fig_opls, "OPLS-DA Score Plot", "t [1]", "to [1]")

            fig_perm = go.Figure()
            fig_perm.add_trace(go.Scatter(x=corrs, y=r2_perm, mode='markers', name='R2', marker=dict(color='green', symbol='circle-open', size=8)))
            fig_perm.add_trace(go.Scatter(x=corrs, y=q2_perm, mode='markers', name='Q2', marker=dict(color='blue', symbol='square-open', size=8)))
            fig_perm.add_trace(go.Scatter(x=[1], y=[R2Y], mode='markers', name='Original R2', marker=dict(color='green', symbol='circle', size=12)))
            fig_perm.add_trace(go.Scatter(x=[1], y=[Q2], mode='markers', name='Original Q2', marker=dict(color='blue', symbol='square', size=12)))
            x_line = np.array([0, 1])
            fig_perm.add_trace(go.Scatter(x=x_line, y=m_r2*x_line + b_r2, mode='lines', name=f'R2 Line (Int: {b_r2:.2f})', line=dict(color='green', dash='dash')))
            fig_perm.add_trace(go.Scatter(x=x_line, y=m_q2*x_line + b_q2, mode='lines', name=f'Q2 Line (Int: {b_q2:.2f})', line=dict(color='blue', dash='dash')))
            fig_perm.update_layout(template="simple_white", width=600, height=600, title={'text': "Permutation Test (n=100)", 'y':0.95, 'x':0.5, 'xanchor': 'center'}, xaxis_title="Correlation", yaxis_title="R2 / Q2")

            splot_df = stats_df.copy()
            splot_df['Color'] = np.where(splot_df['Is_Biomarker'], 'VIP>1 & P<0.05', 'NS')
            fig_splot = px.scatter(splot_df, x='Log2_FC', y='p_corr', color='Color', hover_data=['Name', 'VIP'], color_discrete_map={'VIP>1 & P<0.05': '#CD0000', 'NS': '#E0E0E0'})
            fig_splot.add_hline(y=0.5, line_dash="dash", line_color="gray"); fig_splot.add_hline(y=-0.5, line_dash="dash", line_color="gray")
            fig_splot = update_layout_square(fig_splot, "S-Plot", "Log2 Fold Change", "p(corr)")

            top_vip_df = stats_df.sort_values('VIP', ascending=True).tail(vip_show_num)
            fig_vip = px.bar(top_vip_df, x="VIP", y="Name", orientation='h', color="VIP", color_continuous_scale="RdBu_r")
            fig_vip.add_vline(x=1.0, line_dash="dash", line_color="black")
            fig_vip.update_layout(template="simple_white", width=800, height=700, title={'text': f"Top {vip_show_num} VIP Scores", 'x':0.5, 'xanchor': 'center'}, coloraxis_showscale=False)

            fig_pca = None
            if len(df_proc) >= 3:
                valid_feats_pca = df_proc[feats].var()[df_proc[feats].var() > 1e-9].index.tolist()
                if valid_feats_pca:
                    X_scaled_all = StandardScaler().fit_transform(df_proc[valid_feats_pca])
                    pca_all = PCA(n_components=2).fit(X_scaled_all)
                    pcs_all = pca_all.transform(X_scaled_all)
                    var_all = pca_all.explained_variance_ratio_
                    pca_df_all = pd.DataFrame({'PC1': pcs_all[:,0], 'PC2': pcs_all[:,1], 'Group': df_proc[group_col].values, 'SampleID': df_proc['SampleID']})
                    fig_pca = px.scatter(pca_df_all, x='PC1', y='PC2', color='Group', symbol='Group', hover_data=['SampleID'], color_discrete_sequence=GROUP_COLORS)
                    for i, g in enumerate(sorted(df_proc[group_col].unique())):
                        sub_grp = pca_df_all[pca_df_all['Group'] == g]
                        if len(sub_grp) >= 3:
                            el_x, el_y = get_ellipse_coordinates(sub_grp['PC1'], sub_grp['PC2'])
                            if el_x is not None: fig_pca.add_trace(go.Scatter(x=el_x, y=el_y, mode='lines', line=dict(color=GROUP_COLORS[i % len(GROUP_COLORS)], width=1, dash='dot'), showlegend=False, hoverinfo='skip'))
                    fig_pca.update_traces(marker=dict(size=14, line=dict(width=1, color='black'), opacity=0.9))
                    fig_pca = update_layout_square(fig_pca, "Global PCA Plot", f"PC1 ({var_all[0]:.1%})", f"PC2 ({var_all[1]:.1%})")

            fig_vol = px.scatter(stats_df, x="Log2_FC", y="-Log10_P", color="Sig", color_discrete_map=COLOR_PALETTE, hover_data=['Name', 'VIP'])
            fig_vol.add_hline(y=-np.log10(p_th), line_dash="dash", line_color="gray")
            fig_vol.add_vline(x=fc_th, line_dash="dash", line_color="gray"); fig_vol.add_vline(x=-fc_th, line_dash="dash", line_color="gray")
            fig_vol = update_layout_square(fig_vol, "Volcano Plot", "Log2 Fold Change", "-Log10(P-value)")

            hm_fig, hm_base64 = None, ""
            sig_mets = out_df['Metabolite'].tolist()
            if sig_mets:
                hm_feats = out_df.head(50)['Metabolite'].tolist()
                hm_data = df_sub.set_index(group_col)[hm_feats].T
                hm_data.index = [meta.loc[f, 'Clean_Name'] if (meta is not None and f in meta.index) else f for f in hm_data.index]
                lut = {g: GROUP_COLORS[i%len(GROUP_COLORS)] for i, g in enumerate(df_sub[group_col].unique())}
                try:
                    g = sns.clustermap(hm_data.astype(float), z_score=0, cmap="vlag", center=0, col_colors=df_sub[group_col].map(lut), figsize=(8, 8))
                    g.ax_heatmap.set_xlabel(""); g.ax_heatmap.set_ylabel(""); hm_fig = g.fig
                    buf = io.BytesIO(); g.savefig(buf, format='png', bbox_inches='tight'); buf.seek(0); hm_base64 = base64.b64encode(buf.read()).decode('utf-8')
                except Exception: pass

            fig_nomogram = None
            if len(out_df) >= 2:
                top_n = min(nomo_num, len(out_df))
                try: fig_nomogram = plot_nomogram(df_sub, out_df.head(top_n)['Metabolite'].tolist(), out_df.head(top_n)['Name'].tolist(), group_col, case)
                except: pass

            pathway_df, filtered_db_df, fig_pathway, fig_network = pd.DataFrame(), pd.DataFrame(), None, None
            sig_mets_fullnames = stats_df[stats_df['Is_Biomarker']]['Search_Name'].tolist()
            if sig_mets_fullnames:
                pathway_df, filtered_db_df = run_pathway_enrichment(sig_mets_fullnames, stats_df['Search_Name'].tolist(), custom_db_source=custom_pathway_file if custom_pathway_file else db_filename)
                if not pathway_df.empty:
                    pathway_df['-Log10_P'] = -np.log10(pathway_df['P_Value'].astype(float).clip(lower=1e-10))
                    plot_pw_df = pathway_df[pathway_df['Hits'] > 0].head(pw_show_num)
                    fig_pathway = px.scatter(plot_pw_df, x='Enrichment_Factor', y='-Log10_P', size='Hits', color='P_Value', hover_name='Pathway', hover_data={'Hit_Metabolites': True, 'P_Value': ':.4f'}, color_continuous_scale='Reds_r', size_max=40)
                    fig_pathway.update_traces(marker=dict(line=dict(width=1, color='black'), opacity=0.6))
                    fig_pathway.update_layout(template="simple_white", width=800, height=600, title={'text': "Pathway Enrichment", 'y':0.95, 'x':0.5, 'xanchor': 'center'}, xaxis_title="Enrichment Factor", yaxis_title="-Log10(P)")
                    fig_pathway.add_hline(y=-np.log10(0.05), line_dash="dash", line_color="gray")

                    sig_pws = pathway_df[pathway_df['P_Value'] < 0.05].head(pw_show_num)
                    if not sig_pws.empty:
                        G = nx.Graph()
                        fc_dict = dict(zip(out_df['Search_Name'].apply(lambda x: str(x).split(';')[0].strip()), out_df['Log2_FC']))
                        for _, row in sig_pws.iterrows():
                            pw_name = row['Pathway']
                            G.add_node(pw_name, node_type='pathway', size=max(15, -np.log10(row['P_Value']) * 10))
                            if pd.notna(row['Hit_Metabolites']) and str(row['Hit_Metabolites']).strip() != "":
                                for hit in [m.strip() for m in row['Hit_Metabolites'].split(',')]:
                                    if hit in fc_dict:
                                        G.add_node(hit, node_type='metabolite', size=10, fc=fc_dict[hit], disp_name=hit)
                                        G.add_edge(pw_name, hit)
                        if len(G.nodes) > 0:
                            pos = nx.spring_layout(G, k=0.7, seed=42)
                            edge_trace = go.Scatter(x=[pos[e[0]][0] for e in G.edges()] + [pos[e[1]][0] for e in G.edges()], y=[pos[e[0]][1] for e in G.edges()] + [pos[e[1]][1] for e in G.edges()], mode='lines', line=dict(width=1, color='#888'), hoverinfo='none')
                            node_trace = go.Scatter(x=[pos[n][0] for n in G.nodes()], y=[pos[n][1] for n in G.nodes()], mode='markers+text', text=[G.nodes[n].get('disp_name', n) if G.nodes[n]['node_type']=='metabolite' else '' for n in G.nodes()], textposition="top center", marker=dict(symbol=['square' if G.nodes[n]['node_type']=='pathway' else 'circle' for n in G.nodes()], color=['#FFD700' if G.nodes[n]['node_type']=='pathway' else ('#CD0000' if G.nodes[n]['fc']>0 else '#00008B') for n in G.nodes()], size=[G.nodes[n]['size'] for n in G.nodes()], line_width=1, line_color='black'))
                            fig_network = go.Figure(data=[edge_trace, node_trace])
                            fig_network.update_layout(title={'text': "Mechanism Network", 'y':0.95, 'x':0.5, 'xanchor': 'center'}, showlegend=False, xaxis=dict(showgrid=False, zeroline=False, showticklabels=False), yaxis=dict(showgrid=False, zeroline=False, showticklabels=False), width=900, height=700, plot_bgcolor='white')

            html_report = generate_offline_html(case, ctrl, feats, p_th, fc_th, norm_m, scale_m, R2Y, Q2, b_q2, out_df, pathway_df, fig_opls, fig_perm, fig_splot, fig_vol, fig_pca, hm_base64, fig_nomogram, fig_pathway, fig_network, vip_show_num, pw_show_num, nomo_num)
            prompt_md = generate_ai_prompt(case, ctrl, norm_m, scale_m, R2Y, Q2, b_q2, p_th, fc_th, out_df, pathway_df)

            st.session_state['analysis_res'] = {
                'case': case, 'ctrl': ctrl, 'R2Y': R2Y, 'Q2': Q2, 'b_q2': b_q2,
                'fig_opls': fig_opls, 'fig_perm': fig_perm, 'fig_splot': fig_splot,
                'fig_vip': fig_vip, 'fig_pca': fig_pca, 'fig_vol': fig_vol,
                'hm_fig': hm_fig, 'out_df': out_df, 'pathway_df': pathway_df,
                'filtered_db_df': filtered_db_df, 
                'fig_nomogram': fig_nomogram, 'fig_pathway': fig_pathway, 'fig_network': fig_network,
                'html_report': html_report, 'prompt_md': prompt_md
            }
        except Exception as e:
            error_details = traceback.format_exc()
            st.error(f"❌ 分析计算失败，请检查数据组别与参数。错误摘要: {str(e)}")
            with st.expander("点击查看详细报错日志 (供调试)"): st.code(error_details)

# ==========================================
# 5. UI 展示层 
# ==========================================
if 'analysis_res' in st.session_state:
    res = st.session_state['analysis_res']
    
    st.title("📊 综合代谢组学分析报告")
    st.markdown(f"**对比**: {res['case']} vs {res['ctrl']} &nbsp;&nbsp;|&nbsp;&nbsp; **模型**: R²Y = `{res['R2Y']:.3f}` &nbsp;&nbsp;|&nbsp;&nbsp; Q² = `{res['Q2']:.3f}`")
    if res['b_q2'] < 0.05 and res['Q2'] > 0.5: st.success(f"✅ OPLS-DA 模型优秀且未过拟合！ (Q²截距: {res['b_q2']:.3f})")
    else: st.warning(f"⚠️ 模型可能过拟合，或组间差异不大 (Q²截距: {res['b_q2']:.3f})")

    tabs = st.tabs(["🎯 OPLS-DA", "🔄 置换检验", "🧬 S-Plot", "📊 VIP", "🌐 PCA", "🌋 火山/热图", "📑 清单", "📏 列线图", "🕸️ 通路富集", "🔗 机制网络图", "📄 导出报告与AI助手"])
    
    with tabs[0]: st.plotly_chart(res['fig_opls']) if res['fig_opls'] else st.warning("图表生成失败")
    with tabs[1]: st.plotly_chart(res['fig_perm']) if res['fig_perm'] else st.warning("图表生成失败")
    with tabs[2]: st.plotly_chart(res['fig_splot']) if res['fig_splot'] else st.warning("图表生成失败")
    with tabs[3]: st.plotly_chart(res['fig_vip']) if res['fig_vip'] else st.warning("图表生成失败")
    with tabs[4]: st.plotly_chart(res['fig_pca']) if res['fig_pca'] else st.warning("样本不足以绘制PCA")
    with tabs[5]:
        c1, c2 = st.columns(2)
        with c1: st.plotly_chart(res['fig_vol'], use_container_width=True) if res['fig_vol'] else None
        with c2: st.pyplot(res['hm_fig']) if res['hm_fig'] else st.info("无满足要求的差异代谢物")
    with tabs[6]:
        st.markdown("### 🏆 生物标志物清单")
        st.dataframe(res['out_df'][['Name', 'Log2_FC', 'P_Value', 'FDR', 'VIP', 'p_corr']].style.format({"Log2_FC":"{:.2f}", "P_Value":"{:.3e}", "FDR":"{:.3e}", "VIP":"{:.2f}", "p_corr":"{:.2f}"}).background_gradient(subset=['VIP'], cmap="Reds"), use_container_width=True)
    with tabs[7]:
        if res['fig_nomogram']: st.plotly_chart(res['fig_nomogram'])
        else: st.warning("⚠️ 显著差异代谢物不足 2 个或分组异常，无法构建列线图。")
    with tabs[8]:
        if 'filtered_db_df' in res and not res['filtered_db_df'].empty:
            st.download_button("📥 导出基于本次实验的专属 MA 背景库", res['filtered_db_df'].to_csv(index=False, header=False, quoting=csv.QUOTE_ALL).encode('utf-8'), f"MA_Background_{res['case']}_{res['ctrl']}.csv", "text/csv", type="primary")
        if res['pathway_df'].empty: st.warning("未能匹配到通路。")
        else:
            if res['fig_pathway']: st.plotly_chart(res['fig_pathway'])
            st.dataframe(res['pathway_df'].drop(columns=['-Log10_P'], errors='ignore').style.format({"P_Value":"{:.3e}", "FDR":"{:.3e}", "Enrichment_Factor":"{:.2f}"}).background_gradient(subset=['P_Value'], cmap="Reds_r", vmin=0, vmax=0.05), use_container_width=True)
    with tabs[9]:
        if res['fig_network']: st.plotly_chart(res['fig_network'])
        else: st.info("没有找到通路与代谢物的有效映射。")
    with tabs[10]:
        c_rep1, c_rep2 = st.columns(2)
        with c_rep1: st.download_button("📥 下载完整离线网页报告 (.html)", res['html_report'].encode('utf-8'), f"Report_{res['case']}_{res['ctrl']}.html", "text/html")
        with c_rep2: st.download_button("📥 下载 AI 辅助撰稿 Prompt (.md)", res['prompt_md'].encode('utf-8'), f"Prompt_{res['case']}_{res['ctrl']}.md", "text/markdown")
