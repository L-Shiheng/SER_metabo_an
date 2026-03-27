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
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ==========================================
# 0. 自定义模块导入与配置
# ==========================================
st.set_page_config(page_title="MetaboAnalyst Pro (SIMCA Edition)", page_icon="🧬", layout="wide")

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'PingFang SC', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

try:
    from data_preprocessing import (
        data_cleaning_pipeline, 
        parse_metdna_file, 
        parse_manual_targeted_files, 
        merge_multiple_dfs, 
        align_sample_info, 
        OPLS_DA, 
        run_pathway_enrichment, 
        build_kegg_dictionary
    )
    from stats_utils import run_pairwise_statistics
    from plot_utils import (
        update_layout_square, 
        get_ellipse_coordinates, 
        plot_nomogram
    )
    from report_generator import generate_offline_html, generate_ai_prompt
except ImportError as e:
    st.error(f"❌ 严重错误：未找到依赖文件。请确保同目录下有所需的 .py 文件。详情: {e}")
    st.stop()

try:
    from serrf_module import serrf_normalization
except ImportError:
    st.warning("⚠️ 未找到 `serrf_module.py`，SERRF 校正功能将被禁用。")

COLOR_PALETTE = {'Up': '#CD0000', 'Down': '#00008B', 'NS': '#E0E0E0'} 
GROUP_COLORS = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F', '#8491B4', '#91D1C2', '#DC0000', '#7E6148', '#B09C85']

st.markdown("""
<style>
    .block-container {padding-top: 2rem !important; padding-bottom: 3rem !important;}
    h1, h2, h3, div, p {font-family: 'Arial', sans-serif; color: #2c3e50;}
    div[data-testid="stForm"] button {width: 100%; background-color: #ff4b4b; color: white; font-weight: bold;}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. 状态管理 & 侧边栏
# ==========================================
if 'raw_df' not in st.session_state: st.session_state.raw_df = None
if 'feature_meta' not in st.session_state: st.session_state.feature_meta = None
if 'data_loaded' not in st.session_state: st.session_state.data_loaded = False
if 'qc_report' not in st.session_state: st.session_state.qc_report = {}
if 'all_sample_ids' not in st.session_state: st.session_state.all_sample_ids = []

with st.sidebar:
    st.header("🛠️ 数据控制台")
    st.markdown("#### 1. 上传 Sample Info (必选)")
    sample_info_file = st.file_uploader("Info表格 (.csv/.xlsx)", type=["csv", "xlsx"], key="info")
    info_df = None; candidate_samples = []; user_sample_col = None; user_group_col = None
    
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

    excluded_samples = st.multiselect("2. 样本剔除 (黑名单)", options=candidate_samples, default=[])
    
    use_serrf = st.checkbox("3. 启用 SERRF 批次校正", value=False)
    serrf_ready = False
    if use_serrf:
        if info_df is not None:
            cols = list(info_df.columns); cols_lower = [c.lower() for c in cols]
            idx_order = next((i for i, c in enumerate(cols_lower) if any(x in c for x in ['order', 'run', 'idx', 'seq'])), 0)
            final_type_idx = cols.index(user_group_col) if user_group_col and info_df[user_group_col].astype(str).str.contains('QC', case=False).any() else next((i for i, c in enumerate(cols_lower) if any(x in c for x in ['class', 'type', 'group'])), 0)
            default_qc_label = next((v for v in info_df.iloc[:, final_type_idx].unique().astype(str) if 'qc' in v.lower()), "QC")
            c1, c2, c3 = st.columns(3)
            run_order_col = c1.selectbox("Order", cols, index=idx_order)
            sample_type_col = c2.selectbox("Type", cols, index=final_type_idx)
            qc_label = c3.text_input("QC名", value=default_qc_label)
            serrf_ready = True
        else: st.warning("⚠️ 需上传 Info 表")

    st.markdown("#### 4. 在线通路引擎 (自动调用 KEGG API)")
    species = st.selectbox("选择物种 (强烈影响富集显著性)", ["Human (人类 - 推荐)", "Mouse (小鼠)", "Rat (大鼠)", "General (所有物种)"], index=0)
    species_code = {"Human (人类 - 推荐)": "hsa", "Mouse (小鼠)": "mmu", "Rat (大鼠)": "rno", "General (所有物种)": "map"}[species]
    db_filename = f"kegg_{species_code}.csv"
    
    custom_pathway_file = st.file_uploader("手动上传库 (覆盖在线库)", type=["csv", "gmt"], key="pathway_db")
    
    if st.button(f"🔄 强制同步 {species_code} 最新通路库", use_container_width=True) or not os.path.exists(db_filename):
        with st.spinner(f"正在连接 KEGG API 拉取 {species} 最新专属通路库..."):
            try:
                pw_res = requests.get(f"http://rest.kegg.jp/list/pathway/{species_code}")
                pw_dict = {}
                for line in pw_res.text.strip().split('\n'):
                    if line:
                        parts = line.split('\t')
                        pw_id_num = re.sub(r'^[a-z]+', '', parts[0].replace('path:', ''))
                        pw_dict[pw_id_num] = parts[1]
                
                link_res = requests.get("http://rest.kegg.jp/link/cpd/pathway")
                pw_cpd_map = {}
                for line in link_res.text.strip().split('\n'):
                    if line:
                        parts = line.split('\t')
                        if parts[0].startswith('path:map'):
                            pw_num = parts[0].replace('path:map', '')
                            cpd = parts[1].replace('cpd:', '')
                            if pw_num not in pw_cpd_map:
                                pw_cpd_map[pw_num] = []
                            pw_cpd_map[pw_num].append(cpd)
                
                data = []
                for pw_num, name in pw_dict.items():
                    if pw_num in pw_cpd_map:
                        data.append({'Pathway': name, 'Compounds': ';'.join(pw_cpd_map[pw_num])})
                pd.DataFrame(data).to_csv(db_filename, index=False)
                st.toast(f"✅ {species} 库同步成功！")
            except Exception as e:
                st.error(f"❌ 网络请求失败: {str(e)}")
    
    st.markdown("#### 5. 上传代谢组学数据")
    data_source = st.radio("选择数据源类型 (物理隔离双轨制):", ["MetDNA 原始结果", "手动 MRM 靶向宽表"], index=0)
    
    metric_suffix = " : 面积 "
    mode_regex = r'-(P|N|RP-P|RP-N|HILIC-P|HILIC-N|POS|NEG)-'
    dict_files = None
    
    if data_source == "手动 MRM 靶向宽表":
        st.info("💡 系统将调用专属【宽表字典融合引擎】")
        c1, c2 = st.columns(2)
        metric_suffix = c1.text_input("提取指标", value=" : 面积 ")
        mode_regex = c2.text_input("模式清洗正则", value=mode_regex)
        feature_scope = "全部特征" 
        
        st.markdown("##### 📚 关联 MetDNA 字典 (仅手动表模式可用)")
        dict_files = st.file_uploader("上传 MetDNA 字典表 (支持多选)", type=["csv", "xlsx"], accept_multiple_files=True, key="dict_files")
        
    else:
        st.info("💡 系统将调用纯净的【MetDNA 原始解析引擎】")
        feature_scope = st.radio("特征范围", ["仅已注释特征", "全部特征"], index=0)
        
    uploaded_files = st.file_uploader("上传主分析数据表 (支持多选)", type=["csv", "xlsx"], accept_multiple_files=True, key="data")
    st.markdown("---")
    start_process = st.container().button("📥 开始清洗并加载数据", use_container_width=True, type="primary")

# ==========================================
# 2. 数据处理运行 (隔离路由)
# ==========================================
if start_process:
    st.session_state.qc_report = {}
    # 清空上次分析的缓存，防止数据重叠
    if 'analysis_res' in st.session_state:
        del st.session_state['analysis_res']
        
    if not uploaded_files: st.error("请先上传数据文件！")
    else:
        progress_bar = st.progress(0); status_text = st.empty()
        
        if data_source == "手动 MRM 靶向宽表":
            with st.spinner("正在启动手动宽表融合引擎..."):
                ext_dict = build_kegg_dictionary(dict_files) if dict_files else {}
                if ext_dict: st.success(f"📚 成功构建后台字典：共提取到 {len(ext_dict)} 个独特代谢物的 KEGG 映射！")
                
                df_t, meta, err = parse_manual_targeted_files(uploaded_files, metric_suffix, mode_regex, external_kegg_dict=ext_dict)
                
                if err: 
                    st.error(err)
                else:
                    if excluded_samples:
                        ex_fps = set([re.sub(r'[^a-z0-9]', '', str(s).strip().lower()) for s in excluded_samples])
                        df_t = df_t[~df_t['SampleID'].astype(str).apply(lambda s: re.sub(r'[^a-z0-9]', '', str(s).strip().lower())).isin(ex_fps)]
                    
                    if info_df is not None:
                        info_aligned = align_sample_info(df_t, info_df, sample_col_name=user_sample_col)
                        if user_group_col and user_group_col in info_aligned.columns: df_t['Group'] = info_aligned[user_group_col].fillna('Unknown').values
                        else:
                            g_col = next((c for c in info_aligned.columns if c.lower() in ['group', 'class']), None)
                            if g_col: df_t['Group'] = info_aligned[g_col].fillna('Unknown').values
                    
                    st.session_state.raw_df = df_t
                    st.session_state.feature_meta = meta
                    st.session_state.data_loaded = True
                    st.success("✅ 手动靶向数据融合完成！")
                    st.rerun()

        else:
            with st.spinner("正在启动 MetDNA 原表解析引擎..."):
                parsed_results = []; current_run_samples = set()
                for i, file in enumerate(uploaded_files):
                    status_text.text(f"处理中: {file.name} ...")
                    try:
                        file_type = 'csv' if file.name.endswith('.csv') else 'excel'
                        unique_name = f"{os.path.splitext(file.name)[0]}_{i+1}{os.path.splitext(file.name)[1]}"
                        
                        df_t, meta, err = parse_metdna_file(file, unique_name, file_type=file_type)
                        if err: st.warning(f"{file.name}: {err}"); continue
                        parsed_results.append((df_t, meta, unique_name))
                    except Exception as e: st.error(f"处理 {file.name} 失败: {str(e)}")
                    progress_bar.progress((i + 1) / len(uploaded_files))
    
                if parsed_results:
                    raw_df, feature_meta, err = merge_multiple_dfs(parsed_results)
                    if err:
                        st.error(err)
                    else:
                        if excluded_samples:
                            ex_fingerprints = set([re.sub(r'[^a-z0-9]', '', str(s).strip().lower()) for s in excluded_samples])
                            raw_df = raw_df[~raw_df['SampleID'].astype(str).apply(lambda s: re.sub(r'[^a-z0-9]', '', str(s).strip().lower())).isin(ex_fingerprints)]
                        
                        if feature_scope.startswith("仅已注释"):
                            annotated_ids = feature_meta[feature_meta['Is_Annotated'] == True].index
                            keep_cols = ['SampleID', 'Group', 'Source_Files'] + [c for c in raw_df.columns if c in annotated_ids]
                            raw_df = raw_df[keep_cols]
                            feature_meta = feature_meta.loc[feature_meta.index.isin(raw_df.columns)]
                            
                        if info_df is not None:
                            info_aligned = align_sample_info(raw_df, info_df, sample_col_name=user_sample_col)
                            if user_group_col and user_group_col in info_aligned.columns: 
                                raw_df['Group'] = info_aligned[user_group_col].fillna('Unknown').values
                        
                        st.session_state.raw_df = raw_df
                        st.session_state.feature_meta = feature_meta
                        st.session_state.data_loaded = True
                        st.success("✅ MetDNA 纯净解析完成！")
                        st.rerun()

# ==========================================
# 3. 统计与可视化展示区
# ==========================================
if st.session_state.data_loaded and st.session_state.raw_df is not None:
    raw_df = st.session_state.raw_df
    st.info(f"数据总览: {len(raw_df)} 样本 x {len(raw_df.columns)-3} 特征")
    csv_data = raw_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 导出清洗前合并数据", csv_data, f"Metabo_Raw_{datetime.datetime.now().strftime('%Y%m%d')}.csv", "text/csv")
    st.divider()

    with st.form(key='analysis_form'):
        st.markdown("### ⚙️ 统计与富集分析设置 (SIMCA 金标准)")
        non_num = raw_df.select_dtypes(exclude=[np.number]).columns.tolist()
        group_col = st.selectbox("分组列", non_num, index=non_num.index('Group') if 'Group' in non_num else 0)
        
        with st.expander("数据预处理配置 (点击展开)", expanded=False):
            c_p1, c_p2 = st.columns(2)
            miss_th = c_p1.slider("剔除缺失率 > X", 0.0, 1.0, 0.20)
            impute_m = c_p2.selectbox("缺失值填充", ["KNN", "min", "mean", "zero"], index=0)
            
            c_p3, c_p4 = st.columns(2)
            norm_m = c_p3.selectbox("样本归一化", ["PQN", "Median", "Sum", "None"], index=0)
            scale_m = c_p4.selectbox("特征缩放 (Scaling)", ["Pareto", "Auto", "None"], index=0)
            do_log = st.checkbox("Log2 对数转化 (强烈推荐)", value=True)

        with st.expander("🎨 可视化高级工具栏 (图表参数微调)", expanded=False):
            c_t1, c_t2, c_t3 = st.columns(3)
            vip_show_num = c_t1.slider("VIP 柱状图显示数量", min_value=10, max_value=50, value=25, step=5)
            nomo_num = c_t2.slider("列线图纳入标志物数", min_value=2, max_value=8, value=4)
            pw_show_num = c_t3.slider("网络图通路数", min_value=5, max_value=30, value=15, step=5)

        cur_grps = sorted(raw_df[group_col].astype(str).unique())
        sel_grps = st.multiselect("纳入对比组 (OPLS-DA 需要严格的 2 组对比)", cur_grps, default=cur_grps[:2] if len(cur_grps)>=2 else cur_grps)
        
        c1, c2, c3, c4 = st.columns(4)
        valid = list(sel_grps)
        case = c1.selectbox("Case 组 (实验组)", valid, index=0 if valid else None)
        ctrl = c2.selectbox("Control 组 (对照组)", valid, index=1 if len(valid)>1 else 0)
        p_th = c3.number_input("P-value 阈值", value=0.05, step=0.01)
        fc_th = c4.number_input("Log2 FC 阈值", value=0.58, step=0.10)
        
        submit_button = st.form_submit_button(label='🚀 运行全自动分析 (生成交互图表与报告)')

if not st.session_state.data_loaded:
    st.title("🧬 MetaboAnalyst Pro (SIMCA Edition)"); st.info("👈 请在左侧面板上传并处理数据"); st.stop()

# ==========================================
# 4. 执行核心分析计算 (锁定后台保险箱)
# ==========================================
if submit_button:
    if len(sel_grps) != 2: 
        st.error("⚠️ OPLS-DA 必须且只能选择 2 个组进行对比！")
        st.stop()

    with st.spinner("正在运行核心运算引擎与可视化构建，请稍候..."):
        raw_df = st.session_state.raw_df
        meta = st.session_state.feature_meta
        
        df_proc, feats = data_cleaning_pipeline(
            raw_df, group_col, missing_thresh=miss_th, 
            impute_method=impute_m, norm_method=norm_m, 
            log_transform=do_log, scale_method=scale_m
        )
        df_sub = df_proc[df_proc[group_col].isin(sel_grps)].copy()
 
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

        # 🎯 预先生成所有图表
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
        if len(df_sub) >= 3:
            X_scaled = StandardScaler().fit_transform(df_sub[feats])
            pca = PCA(n_components=2).fit(X_scaled); pcs = pca.transform(X_scaled); var = pca.explained_variance_ratio_
            pca_df = pd.DataFrame({'PC1': pcs[:,0], 'PC2': pcs[:,1], 'Group': df_sub[group_col].values, 'SampleID': df_sub['SampleID']})
            fig_pca = px.scatter(pca_df, x='PC1', y='PC2', color='Group', symbol='Group', hover_data=['SampleID'], color_discrete_sequence=GROUP_COLORS)
            el_x, el_y = get_ellipse_coordinates(pca_df['PC1'], pca_df['PC2'])
            if el_x is not None: fig_pca.add_trace(go.Scatter(x=el_x, y=el_y, mode='lines', line=dict(color='black', width=1, dash='dot'), name='95% Hotelling T2'))
            fig_pca.update_traces(marker=dict(size=14, line=dict(width=1, color='black'), opacity=0.9))
            fig_pca = update_layout_square(fig_pca, "PCA (QC Check)", f"PC1 ({var[0]:.1%})", f"PC2 ({var[1]:.1%})")

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
            col_colors = df_sub[group_col].map(lut)
            try:
                g = sns.clustermap(hm_data.astype(float), z_score=0, cmap="vlag", center=0, col_colors=col_colors, figsize=(8, 8))
                g.ax_heatmap.set_xlabel(""); g.ax_heatmap.set_ylabel("")
                hm_fig = g.fig
                buf = io.BytesIO(); g.savefig(buf, format='png', bbox_inches='tight'); buf.seek(0); hm_base64 = base64.b64encode(buf.read()).decode('utf-8')
            except Exception as e: pass

        fig_nomogram = None
        if len(out_df) >= 2:
            top_n = min(nomo_num, len(out_df))
            nomo_feats = out_df.head(top_n)['Metabolite'].tolist()
            nomo_names = out_df.head(top_n)['Name'].tolist()
            try:
                fig_nomogram = plot_nomogram(df_sub, nomo_feats, nomo_names, group_col, case)
            except: pass

        pathway_df = pd.DataFrame()
        fig_pathway = None
        sig_mets_fullnames = stats_df[stats_df['Is_Biomarker']]['Search_Name'].tolist()
        all_mets_fullnames = stats_df['Search_Name'].tolist()
        if sig_mets_fullnames:
            db_source = custom_pathway_file if custom_pathway_file else db_filename
            pathway_df = run_pathway_enrichment(sig_mets_fullnames, all_mets_fullnames, custom_db_source=db_source)
            if not pathway_df.empty:
                if '-Log10_P' not in pathway_df.columns:
                    pathway_df['-Log10_P'] = -np.log10(pathway_df['P_Value'].astype(float).clip(lower=1e-10))
                plot_pw_df = pathway_df[pathway_df['Hits'] > 0].head(pw_show_num)
                fig_pathway = px.scatter(
                    plot_pw_df, x='Enrichment_Factor', y='-Log10_P', size='Hits', color='P_Value',
                    hover_name='Pathway', hover_data={'Hit_Metabolites': True, 'P_Value': ':.4f', 'Enrichment_Factor': ':.2f'},
                    color_continuous_scale='Reds_r', size_max=40
                )
                fig_pathway.update_traces(marker=dict(line=dict(width=1, color='black'), opacity=0.6))
                fig_pathway.update_layout(
                    template="simple_white", width=800, height=600,
                    title={'text': "Pathway Enrichment Bubble Plot", 'y':0.95, 'x':0.5, 'xanchor': 'center'},
                    xaxis_title="Enrichment Factor", yaxis_title="-Log10(P-value)", coloraxis_colorbar=dict(title="P-value")
                )
                fig_pathway.add_hline(y=-np.log10(0.05), line_dash="dash", line_color="gray")

        fig_network = None
        if not pathway_df.empty and not out_df.empty:
            sig_pws = pathway_df[pathway_df['P_Value'] < 0.05].head(pw_show_num)
            if not sig_pws.empty:
                G = nx.Graph()
                robust_keys = out_df['Search_Name'].apply(lambda x: str(x).split(';')[0].strip())
                fc_dict = dict(zip(robust_keys, out_df['Log2_FC']))
                vip_dict = dict(zip(robust_keys, out_df['VIP']))
                disp_name_dict = dict(zip(robust_keys, out_df['Name']))
                for _, row in sig_pws.iterrows():
                    pw_name = row['Pathway']
                    G.add_node(pw_name, node_type='pathway', size=max(15, -np.log10(row['P_Value']) * 10))
                    hits_str = row['Hit_Metabolites']
                    if pd.notna(hits_str) and str(hits_str).strip() != "":
                        for hit in [m.strip() for m in hits_str.split(',')]:
                            if hit in fc_dict:
                                G.add_node(hit, node_type='metabolite', size=max(10, vip_dict.get(hit, 1.0) * 8), fc=fc_dict[hit], disp_name=disp_name_dict.get(hit, hit))
                                G.add_edge(pw_name, hit)
                if len(G.nodes) > 0:
                    pos = nx.spring_layout(G, k=0.7, iterations=50, seed=42)
                    edge_x, edge_y = [], []
                    for edge in G.edges():
                        x0, y0 = pos[edge[0]]; x1, y1 = pos[edge[1]]
                        edge_x.extend([x0, x1, None]); edge_y.extend([y0, y1, None])
                    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=1, color='#888'), hoverinfo='none', mode='lines')
                    node_x, node_y, node_text, node_color, node_size, node_symbol = [], [], [], [], [], []
                    for node in G.nodes():
                        x, y = pos[node]
                        node_x.append(x); node_y.append(y)
                        node_info = G.nodes[node]
                        if node_info['node_type'] == 'pathway':
                            node_color.append('#FFD700'); node_size.append(node_info['size']); node_symbol.append('square')
                            node_text.append(f"<b>[Pathway]</b><br>{node}")
                        else:
                            fc = node_info['fc']
                            disp_name = node_info.get('disp_name', node).split(' | ')[0]
                            node_color.append('#CD0000' if fc > 0 else '#00008B')
                            node_size.append(node_info['size']); node_symbol.append('circle')
                            node_text.append(f"<b>{disp_name}</b><br>Log2FC: {fc:.2f}")
                    node_trace = go.Scatter(
                        x=node_x, y=node_y, mode='markers+text',
                        hoverinfo='text', text=[G.nodes[n].get('disp_name', n).split(' | ')[0] if G.nodes[n]['node_type']=='metabolite' else '' for n in G.nodes()], 
                        textposition="top center", hovertext=node_text,
                        marker=dict(symbol=node_symbol, showscale=False, color=node_color, size=node_size, line_width=1, line_color='black')
                    )
                    fig_network = go.Figure(data=[edge_trace, node_trace])
                    fig_network.update_layout(
                        title={'text': "Metabolic Reprogramming Mechanism Network", 'y':0.95, 'x':0.5, 'xanchor': 'center'},
                        showlegend=False, hovermode='closest', margin=dict(b=20,l=5,r=5,t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        width=900, height=700, plot_bgcolor='white'
                    )

        html_report = generate_offline_html(
            case, ctrl, feats, p_th, fc_th, norm_m, scale_m, R2Y, Q2, b_q2,
            out_df, pathway_df, fig_opls, fig_perm, fig_splot, fig_vol, fig_pca, 
            hm_base64, fig_nomogram, fig_pathway, fig_network, 
            vip_show_num, pw_show_num, nomo_num
        )
        prompt_md = generate_ai_prompt(case, ctrl, norm_m, scale_m, R2Y, Q2, b_q2, p_th, fc_th, out_df, pathway_df)

        # 📦 结果存入保险箱！
        st.session_state['analysis_res'] = {
            'case': case, 'ctrl': ctrl, 'R2Y': R2Y, 'Q2': Q2, 'b_q2': b_q2,
            'fig_opls': fig_opls, 'fig_perm': fig_perm, 'fig_splot': fig_splot,
            'fig_vip': fig_vip, 'fig_pca': fig_pca, 'fig_vol': fig_vol,
            'hm_fig': hm_fig, 'out_df': out_df, 'pathway_df': pathway_df,
            'fig_nomogram': fig_nomogram, 'fig_pathway': fig_pathway, 'fig_network': fig_network,
            'html_report': html_report, 'prompt_md': prompt_md
        }

# ==========================================
# 5. UI 展示层 (无论怎么点击下载，永远从保险箱中取数据呈现，绝不重置！)
# ==========================================
if 'analysis_res' in st.session_state:
    res = st.session_state['analysis_res']
    
    st.title("📊 综合代谢组学分析报告")
    st.markdown(f"**对比**: {res['case']} vs {res['ctrl']} &nbsp;&nbsp;|&nbsp;&nbsp; **模型**: R²Y = `{res['R2Y']:.3f}` &nbsp;&nbsp;|&nbsp;&nbsp; Q² = `{res['Q2']:.3f}`")
    if res['b_q2'] < 0.05 and res['Q2'] > 0.5: st.success(f"✅ OPLS-DA 模型优秀且未过拟合！ (Q²截距: {res['b_q2']:.3f})")
    else: st.warning(f"⚠️ 模型可能过拟合，或组间差异不大 (Q²截距: {res['b_q2']:.3f})")

    tabs = st.tabs(["🎯 OPLS-DA", "🔄 置换检验", "🧬 S-Plot", "📊 VIP", "🌐 PCA", "🌋 火山/热图", "📑 清单", "📏 列线图", "🕸️ 通路富集", "🔗 机制网络图", "📄 导出报告与AI助手"])
    
    with tabs[0]:
        c1, c2 = st.columns([1, 4])
        with c2: st.plotly_chart(res['fig_opls'])
        
    with tabs[1]:
        c1, c2 = st.columns([1, 4])
        with c2: st.plotly_chart(res['fig_perm'])
        
    with tabs[2]:
        c1, c2 = st.columns([1, 4])
        with c2: st.plotly_chart(res['fig_splot'])
        
    with tabs[3]:
        c1, c2 = st.columns([1, 6])
        with c2: st.plotly_chart(res['fig_vip'])
        
    with tabs[4]:
        c1, c2 = st.columns([1, 4])
        with c2:
            if res['fig_pca'] is not None: st.plotly_chart(res['fig_pca'])
            else: st.warning("样本不足")
            
    with tabs[5]:
        c1, c2 = st.columns(2)
        with c1: st.plotly_chart(res['fig_vol'], use_container_width=True)
        with c2:
            if res['hm_fig'] is not None: st.pyplot(res['hm_fig'])
            elif len(res['out_df']) == 0: st.info("无满足要求的差异代谢物")
            else: st.warning("热图生成失败")
            
    with tabs[6]:
        st.markdown("### 🏆 生物标志物清单")
        disp_cols = ['Name', 'Log2_FC', 'P_Value', 'FDR', 'VIP', 'p_corr']
        st.dataframe(res['out_df'][disp_cols].style.format({"Log2_FC":"{:.2f}", "P_Value":"{:.3e}", "FDR":"{:.3e}", "VIP":"{:.2f}", "p_corr":"{:.2f}"}).background_gradient(subset=['VIP'], cmap="Reds"), use_container_width=True)
        
    with tabs[7]:
        st.markdown("### 📏 临床诊断列线图 (Diagnostic Nomogram)")
        st.caption("基于 Logistic 回归模型构建的列线图。系统自动提取 Top 差异标志物构建风险预测模型。")
        if len(res['out_df']) < 2: st.warning("⚠️ 显著差异代谢物不足 2 个，无法构建列线图回归模型。")
        else:
            c1, c2 = st.columns([1, 6])
            with c2:
                if res['fig_nomogram'] is not None: st.plotly_chart(res['fig_nomogram'])
                else: st.error("构建列线图失败，请检查样本的组别分布。")
                
    with tabs[8]:
        st.markdown("### 🕸️ 代谢通路富集 (万能自适应引擎)")
        c1, c2 = st.columns([1, 6])
        with c2:
            if res['pathway_df'].empty: st.warning("未能匹配到通路，请检查是否选择了正确的物种库，或无显著差异标志物。")
            else:
                if res['fig_pathway'] is not None: st.plotly_chart(res['fig_pathway'])
                st.dataframe(res['pathway_df'].drop(columns=['-Log10_P'], errors='ignore').style.format({"P_Value":"{:.3e}", "FDR":"{:.3e}", "Enrichment_Factor":"{:.2f}"}).background_gradient(subset=['P_Value'], cmap="Reds_r", vmin=0, vmax=0.05), use_container_width=True)
                
    with tabs[9]:
        st.markdown("### 🔗 代谢重编程机制网络 (Pathway-Metabolite Network)")
        st.caption("展示显著富集通路（P < 0.05）与核心标志物的相互关联。")
        if res['pathway_df'].empty or res['out_df'].empty: st.info("需要产生显著通路和差异代谢物才能构建网络。")
        else:
            if res['fig_network'] is not None: st.plotly_chart(res['fig_network'])
            else: st.warning("⚠️ 没有找到通路和代谢物之间的有效映射关联或无显著通路，无法绘图。")
            
    with tabs[10]:
        st.markdown("### 📄 报告生成中心")
        c_rep1, c_rep2 = st.columns(2)
        with c_rep1:
            st.markdown("#### 👨‍🔬 1. 完整可视化报告下载 (HTML)")
            st.download_button("📥 下载完整交互式网页报告 (.html)", res['html_report'].encode('utf-8'), f"Metabolomics_Report_{res['case']}_vs_{res['ctrl']}.html", "text/html", type="primary")
        with c_rep2:
            st.markdown("#### 🤖 2. AI 撰稿专属 Prompt")
            st.text_area("拷贝此文本发送给 AI:", value=res['prompt_md'], height=250)
            st.download_button("📥 下载 Prompt 文件 (.md)", res['prompt_md'].encode('utf-8'), f"AI_Prompt_{res['case']}_vs_{res['ctrl']}.md", "text/markdown")
