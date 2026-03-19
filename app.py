import streamlit as st
import pandas as pd
import numpy as np
import os
import gc
import datetime
import re
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

# ==========================================
# 0. 模块导入与配置
# ==========================================
st.set_page_config(page_title="MetaboAnalyst Pro (SIMCA Edition)", page_icon="🧬", layout="wide")

try:
    from data_preprocessing import data_cleaning_pipeline, parse_metdna_file, merge_multiple_dfs, align_sample_info, OPLS_DA, run_pathway_enrichment
except ImportError as e:
    st.error("❌ 严重错误：未找到 `data_preprocessing.py` 或内部函数。请确保该文件在同一目录下并已更新！")
    st.stop()

try:
    from serrf_module import serrf_normalization
except ImportError:
    st.warning("⚠️ 未找到 `serrf_module.py`，SERRF 批次校正功能将被禁用。")

COLOR_PALETTE = {'Up': '#CD0000', 'Down': '#00008B', 'NS': '#E0E0E0'} 
GROUP_COLORS = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F', '#8491B4', '#91D1C2', '#DC0000', '#7E6148', '#B09C85']

st.markdown("""
<style>
    .block-container {padding-top: 2rem !important; padding-bottom: 3rem !important;}
    h1, h2, h3, div, p {font-family: 'Arial', sans-serif; color: #2c3e50;}
    div[data-testid="stForm"] button {width: 100%; background-color: #ff4b4b; color: white; font-weight: bold;}
    .process-btn button {width: 100%; background-color: #4CAF50 !important; color: white !important; font-weight: bold;}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. 绘图辅助函数
# ==========================================
def update_layout_square(fig, title="", x_title="", y_title=""):
    fig.update_layout(template="simple_white", width=600, height=600, title={'text': title, 'y':0.95, 'x':0.5, 'xanchor': 'center'}, xaxis=dict(title=x_title, showline=True, linewidth=2, mirror=True), yaxis=dict(title=y_title, showline=True, linewidth=2, mirror=True), legend=dict(yanchor="top", y=1, xanchor="left", x=1.15))
    return fig

def get_ellipse_coordinates(x, y, std_mult=2):
    if len(x) < 3: return None, None
    mean_x, mean_y = np.mean(x), np.mean(y)
    cov = np.cov(x, y)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:,order]
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
    width, height = 2 * std_mult * np.sqrt(vals)
    t = np.linspace(0, 2*np.pi, 100)
    ell_x = width/2 * np.cos(t); ell_y = height/2 * np.sin(t)
    rad = np.radians(theta)
    R = np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])
    ell_coords = np.dot(R, np.array([ell_x, ell_y]))
    return ell_coords[0] + mean_x, ell_coords[1] + mean_y

@st.cache_data
def run_pairwise_statistics(df, group_col, case, control, features, equal_var=False):
    g1 = df[df[group_col] == case]; g2 = df[df[group_col] == control]; res = []
    for f in features:
        v1, v2 = g1[f].values, g2[f].values; fc = np.mean(v1) - np.mean(v2)
        try: t, p = stats.ttest_ind(v1, v2, equal_var=equal_var)
        except: p = 1.0
        res.append({'Metabolite': f, 'Log2_FC': fc, 'P_Value': p if not np.isnan(p) else 1.0})
    res_df = pd.DataFrame(res).dropna()
    if not res_df.empty: _, res_df['FDR'], _, _ = multipletests(res_df['P_Value'], method='fdr_bh'); res_df['-Log10_P'] = -np.log10(res_df['P_Value'])
    return res_df

# ==========================================
# 2. 状态管理 & 侧边栏
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

    if not candidate_samples and st.session_state.all_sample_ids: candidate_samples = st.session_state.all_sample_ids

    st.markdown("#### 2. 样本剔除 (黑名单)")
    excluded_samples = st.multiselect("选择要剔除的样本:", options=candidate_samples, default=[])

    st.markdown("#### 3. 数据范围")
    feature_scope = st.radio("特征范围:", ["仅已注释特征 (推荐)", "全部特征"], index=0)

    st.markdown("#### 4. SERRF 校正")
    use_serrf = st.checkbox("启用 SERRF", value=False)
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

    st.markdown("#### 5. 外部通路库 (可选)")
    custom_pathway_file = st.file_uploader("上传 .csv 或 .gmt 格式。如果不传，自动读取仓库内 kegg_pathways.csv", type=["csv", "gmt"], key="pathway_db")

    st.markdown("#### 6. 上传 MetDNA 数据")
    uploaded_files = st.file_uploader("结果文件 (支持多选)", type=["csv", "xlsx"], accept_multiple_files=True, key="data")
    st.markdown("---")
    start_process = st.container().button("📥 开始处理数据", use_container_width=True, type="primary")

# ==========================================
# 3. 数据处理运行
# ==========================================
if start_process:
    st.session_state.qc_report = {}
    if not uploaded_files: st.error("请先上传数据文件！")
    else:
        progress_bar = st.progress(0); status_text = st.empty()
        with st.spinner("正在处理..."):
            parsed_results = []; current_run_samples = set()
            for i, file in enumerate(uploaded_files):
                status_text.text(f"处理中: {file.name} ...")
                try:
                    file.seek(0)
                    file_type = 'csv' if file.name.endswith('.csv') else 'excel'
                    unique_name = f"{os.path.splitext(file.name)[0]}_{i+1}{os.path.splitext(file.name)[1]}"
                    
                    df_t, meta, err = parse_metdna_file(file, unique_name, file_type=file_type)
                    if err: st.warning(f"{file.name}: {err}"); continue
                    
                    if excluded_samples:
                        ex_fingerprints = set([re.sub(r'[^a-z0-9]', '', str(s).strip().lower()) for s in excluded_samples])
                        df_t = df_t[~df_t['SampleID'].astype(str).apply(lambda s: re.sub(r'[^a-z0-9]', '', str(s).strip().lower())).isin(ex_fingerprints)]
                    
                    current_run_samples.update(df_t['SampleID'].astype(str).tolist())

                    if feature_scope.startswith("仅已注释"):
                        annotated_ids = meta[meta['Is_Annotated'] == True].index
                        df_t = df_t[['SampleID', 'Group', 'Source_Files'] + [c for c in df_t.columns if c in annotated_ids]]
                        meta = meta.loc[meta.index.isin(df_t.columns)]
                        
                    info_aligned = None
                    if info_df is not None:
                        info_aligned = align_sample_info(df_t, info_df, sample_col_name=user_sample_col)
                        if user_group_col and user_group_col in info_aligned.columns: df_t['Group'] = info_aligned[user_group_col].fillna(df_t['Group']).values
                        else:
                            g_col = next((c for c in info_aligned.columns if c.lower() in ['group', 'class']), None)
                            if g_col: df_t['Group'] = info_aligned[g_col].fillna(df_t['Group']).values
                    
                    if use_serrf and serrf_ready and info_aligned is not None:
                        if info_aligned[run_order_col].notna().sum() == 0: st.session_state.qc_report[unique_name] = {"Status": "Failed (No Match)"}
                        else:
                            corrected_data, serrf_stats = serrf_normalization(df_t.select_dtypes(include=[np.number]), info_aligned, run_order_col, sample_type_col, qc_label)
                            if corrected_data is not None:
                                for c in corrected_data.columns: df_t[c] = corrected_data[c].values
                                st.session_state.qc_report[unique_name] = {"Status": "Skipped (Worse)" if serrf_stats['RSD_After'] > serrf_stats['RSD_Before'] else "Success", "RSD_Before": serrf_stats['RSD_Before'], "RSD_After": serrf_stats['RSD_After']}
                            else: st.error(f"❌ {file.name}: SERRF失败")

                    parsed_results.append((df_t, meta, unique_name))
                    del df_t, meta, info_aligned; gc.collect()
                except Exception as e: st.error(f"处理 {file.name} 失败: {str(e)}")
                progress_bar.progress((i + 1) / len(uploaded_files))

            if parsed_results:
                st.session_state.all_sample_ids = sorted(list(set(st.session_state.all_sample_ids) | current_run_samples))
                if len(parsed_results) == 1: st.session_state.raw_df, st.session_state.feature_meta = parsed_results[0][0], parsed_results[0][1]
                else: st.session_state.raw_df, st.session_state.feature_meta, _ = merge_multiple_dfs(parsed_results)
                st.session_state.data_loaded = True
                st.success("✅ 处理完成！")
                st.rerun() 

# ==========================================
# 4. 统计与可视化展示区
# ==========================================
if st.session_state.data_loaded and st.session_state.raw_df is not None:
    raw_df = st.session_state.raw_df
    st.info(f"数据总览: {len(raw_df)} 样本 x {len(raw_df.columns)-3} 特征")
    csv_data = raw_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 导出清洗前合并数据", csv_data, f"Metabo_{datetime.datetime.now().strftime('%Y%m%d')}.csv", "text/csv")
    st.divider()

    with st.form(key='analysis_form'):
        st.markdown("### ⚙️ 统计与富集分析设置")
        non_num = raw_df.select_dtypes(exclude=[np.number]).columns.tolist()
        group_col = st.selectbox("分组列", non_num, index=non_num.index('Group') if 'Group' in non_num else 0)
        
        with st.expander("数据预处理配置 (SIMCA 推荐 Pareto Scaling)", expanded=False):
            filter_option = st.radio("分析范围:", ["全部特征", "仅已注释特征"], index=0)
            miss_th = st.slider("剔除缺失率 > X", 0.0, 1.0, 0.5)
            impute_m = st.selectbox("缺失值填充", ["min (推荐)", "KNN", "mean", "zero"], index=0).split()[0]
            norm_m = st.selectbox("样本归一化", ["None", "PQN", "Sum", "Median"], index=1)
            do_log = st.checkbox("Log2 对数转化", value=True)
            scale_m = st.selectbox("特征缩放 (Scaling)", ["Pareto (SIMCA 默认)", "Auto (Z-score)", "None"], index=0).split()[0]

        cur_grps = sorted(raw_df[group_col].astype(str).unique())
        sel_grps = st.multiselect("纳入对比组 (OPLS-DA 需要严格的 2 组对比)", cur_grps, default=cur_grps[:2] if len(cur_grps)>=2 else cur_grps)
        c1, c2, c3, c4 = st.columns(4)
        valid = list(sel_grps)
        case = c1.selectbox("Case 组 (实验组)", valid, index=0 if valid else None)
        ctrl = c2.selectbox("Control 组 (对照组)", valid, index=1 if len(valid)>1 else 0)
        p_th = c3.number_input("P-value 阈值", 0.05)
        fc_th = c4.number_input("Log2 FC 阈值", 1.0)
        submit_button = st.form_submit_button(label='🚀 运行分析 (含通路富集)')

if not st.session_state.data_loaded:
    st.title("🧬 MetaboAnalyst Pro (SIMCA Edition)"); st.info("👈 请在左侧面板上传并处理数据"); st.stop()

if not submit_button:
    st.title("✅ 数据准备就绪"); st.dataframe(st.session_state.raw_df.head(50)); st.stop()

if submit_button:
    if len(sel_grps) != 2: st.error("⚠️ OPLS-DA 必须且只能选择 2 个组进行对比！"); st.stop()
    with st.spinner("正在运行 OPLS-DA 置换检验和通路富集..."):
        raw_df = st.session_state.raw_df; meta = st.session_state.feature_meta
        df_proc, feats = data_cleaning_pipeline(raw_df, group_col, miss_th, impute_m, norm_m, do_log, scale_m)
        
        if filter_option == "仅已注释特征":
            anno_ids = meta[meta['Is_Annotated']==True].index.tolist() if meta is not None else []
            feats = [f for f in feats if f in anno_ids]
        
        df_sub = df_proc[df_proc[group_col].isin(sel_grps)].copy()
        
        stats_df = run_pairwise_statistics(df_sub, group_col, case, ctrl, feats)
        if meta is not None: stats_df = stats_df.merge(meta[['Clean_Name']], left_on='Metabolite', right_index=True, how='left')
        stats_df['Name'] = stats_df['Clean_Name'] if 'Clean_Name' in stats_df.columns else stats_df['Metabolite']
        stats_df['Sig'] = 'NS'
        stats_df.loc[(stats_df['P_Value']<p_th)&(stats_df['Log2_FC']>fc_th), 'Sig']='Up'
        stats_df.loc[(stats_df['P_Value']<p_th)&(stats_df['Log2_FC']<-fc_th), 'Sig']='Down'

        y_binary = np.where(df_sub[group_col] == case, 1, -1)
        X_matrix = df_sub[feats].values
        
        opls = OPLS_DA().fit(X_matrix, y_binary)
        corrs, r2_perm, q2_perm, R2Y, Q2 = opls.permutation_test(X_matrix, y_binary, n_permutations=100)
        
        m_q2, b_q2 = np.polyfit(corrs, q2_perm, 1) if len(corrs)>0 else (0,0)
        m_r2, b_r2 = np.polyfit(corrs, r2_perm, 1) if len(corrs)>0 else (0,0)

        vip_df = pd.DataFrame({'Metabolite': feats, 'VIP': opls.vip, 'p_corr': opls.p_corr})
        stats_df = stats_df.merge(vip_df, on='Metabolite')
        stats_df['Is_Biomarker'] = (stats_df['VIP'] > 1.0) & (stats_df['P_Value'] < p_th)

        st.title("📊 综合代谢组学分析报告")
        st.markdown(f"**对比**: {case} vs {ctrl} &nbsp;&nbsp;|&nbsp;&nbsp; **模型**: R²Y = `{R2Y:.3f}` &nbsp;&nbsp;|&nbsp;&nbsp; Q² = `{Q2:.3f}`")
        if b_q2 < 0.05 and Q2 > 0.5: st.success(f"✅ OPLS-DA 模型优秀且未过拟合！ (Q²截距: {b_q2:.3f})")
        else: st.warning(f"⚠️ 模型可能过拟合，或组间差异不大 (Q²截距: {b_q2:.3f})")

        tabs = st.tabs(["🎯 OPLS-DA", "🔄 置换检验", "🧬 S-Plot", "📊 VIP", "🌐 PCA", "🌋 火山/热图", "📑 清单", "🕸️ 通路富集"])
        
        with tabs[0]:
            c1, c2 = st.columns([1, 4])
            with c2:
                opls_score_df = pd.DataFrame({'t1 (Predictive)': opls.t, 't_ortho (Orthogonal)': opls.t_ortho, 'Group': df_sub[group_col].values})
                fig_opls = px.scatter(opls_score_df, x='t1 (Predictive)', y='t_ortho (Orthogonal)', color='Group', symbol='Group', color_discrete_sequence=GROUP_COLORS)
                for i, g in enumerate(list(sel_grps)):
                    sub = opls_score_df[opls_score_df['Group']==g]
                    if len(sub)>=3:
                        el_x, el_y = get_ellipse_coordinates(sub['t1 (Predictive)'], sub['t_ortho (Orthogonal)'])
                        if el_x is not None: fig_opls.add_trace(go.Scatter(x=el_x, y=el_y, mode='lines', line=dict(color=GROUP_COLORS[i%len(GROUP_COLORS)], width=2, dash='dash'), showlegend=False, hoverinfo='skip'))
                fig_opls.update_traces(marker=dict(size=14, line=dict(width=1, color='black'), opacity=0.9))
                fig_opls.add_hline(y=0, line_dash="dash", line_color="gray"); fig_opls.add_vline(x=0, line_dash="dash", line_color="gray")
                st.plotly_chart(update_layout_square(fig_opls, "OPLS-DA Score Plot", "t [1]", "to [1]"))

        with tabs[1]:
            c1, c2 = st.columns([1, 4])
            with c2:
                fig_perm = go.Figure()
                fig_perm.add_trace(go.Scatter(x=corrs, y=r2_perm, mode='markers', name='R2', marker=dict(color='green', symbol='circle-open', size=8)))
                fig_perm.add_trace(go.Scatter(x=corrs, y=q2_perm, mode='markers', name='Q2', marker=dict(color='blue', symbol='square-open', size=8)))
                fig_perm.add_trace(go.Scatter(x=[1], y=[R2Y], mode='markers', name='Original R2', marker=dict(color='green', symbol='circle', size=12)))
                fig_perm.add_trace(go.Scatter(x=[1], y=[Q2], mode='markers', name='Original Q2', marker=dict(color='blue', symbol='square', size=12)))
                x_line = np.array([0, 1])
                fig_perm.add_trace(go.Scatter(x=x_line, y=m_r2*x_line + b_r2, mode='lines', name=f'R2 Line (Int: {b_r2:.2f})', line=dict(color='green', dash='dash')))
                fig_perm.add_trace(go.Scatter(x=x_line, y=m_q2*x_line + b_q2, mode='lines', name=f'Q2 Line (Int: {b_q2:.2f})', line=dict(color='blue', dash='dash')))
                fig_perm.update_layout(template="simple_white", width=600, height=600, title={'text': "Permutation Test (n=100)", 'y':0.95, 'x':0.5, 'xanchor': 'center'}, xaxis_title="Correlation", yaxis_title="R2 / Q2")
                st.plotly_chart(fig_perm)

        with tabs[2]:
            c1, c2 = st.columns([1, 4])
            with c2:
                splot_df = stats_df.copy()
                splot_df['Color'] = np.where(splot_df['Is_Biomarker'], 'VIP>1 & P<0.05', 'NS')
                fig_splot = px.scatter(splot_df, x='Log2_FC', y='p_corr', color='Color', hover_data=['Name', 'VIP'], color_discrete_map={'VIP>1 & P<0.05': '#CD0000', 'NS': '#E0E0E0'})
                fig_splot.add_hline(y=0.5, line_dash="dash", line_color="gray"); fig_splot.add_hline(y=-0.5, line_dash="dash", line_color="gray")
                st.plotly_chart(update_layout_square(fig_splot, "S-Plot", "Log2 Fold Change", "p(corr)"))

        with tabs[3]:
            c1, c2 = st.columns([1, 6])
            with c2:
                top_vip = stats_df.sort_values('VIP', ascending=True).tail(25)
                fig_vip = px.bar(top_vip, x="VIP", y="Name", orientation='h', color="VIP", color_continuous_scale="RdBu_r")
                fig_vip.add_vline(x=1.0, line_dash="dash", line_color="black")
                fig_vip.update_layout(template="simple_white", width=800, height=700, title={'text': "Top 25 VIP Scores", 'x':0.5, 'xanchor': 'center'}, coloraxis_showscale=False)
                st.plotly_chart(fig_vip)

        with tabs[4]:
            c1, c2 = st.columns([1, 4])
            with c2:
                if len(df_sub)<3: st.warning("样本不足")
                else:
                    X_scaled = StandardScaler().fit_transform(df_sub[feats])
                    pca = PCA(n_components=2).fit(X_scaled); pcs = pca.transform(X_scaled); var = pca.explained_variance_ratio_
                    pca_df = pd.DataFrame({'PC1': pcs[:,0], 'PC2': pcs[:,1], 'Group': df_sub[group_col].values, 'SampleID': df_sub['SampleID']})
                    fig_pca = px.scatter(pca_df, x='PC1', y='PC2', color='Group', symbol='Group', hover_data=['SampleID'], color_discrete_sequence=GROUP_COLORS)
                    el_x, el_y = get_ellipse_coordinates(pca_df['PC1'], pca_df['PC2'])
                    if el_x is not None: fig_pca.add_trace(go.Scatter(x=el_x, y=el_y, mode='lines', line=dict(color='black', width=1, dash='dot'), name='95% Hotelling T2'))
                    fig_pca.update_traces(marker=dict(size=14, line=dict(width=1, color='black'), opacity=0.9))
                    st.plotly_chart(update_layout_square(fig_pca, "PCA (QC Check)", f"PC1 ({var[0]:.1%})", f"PC2 ({var[1]:.1%})"))

        with tabs[5]:
            c1, c2 = st.columns(2)
            with c1:
                fig_vol = px.scatter(stats_df, x="Log2_FC", y="-Log10_P", color="Sig", color_discrete_map=COLOR_PALETTE, hover_data=['Name', 'VIP'])
                fig_vol.add_hline(y=-np.log10(p_th), line_dash="dash", line_color="gray")
                fig_vol.add_vline(x=fc_th, line_dash="dash", line_color="gray"); fig_vol.add_vline(x=-fc_th, line_dash="dash", line_color="gray")
                st.plotly_chart(update_layout_square(fig_vol, "Volcano Plot", "Log2 Fold Change", "-Log10(P-value)"), use_container_width=True)
            with c2:
                sig_mets = stats_df[stats_df['Is_Biomarker']]['Metabolite'].tolist()
                if not sig_mets: st.info("无满足要求的差异代谢物")
                else:
                    hm_feats = stats_df.sort_values('VIP', ascending=False).head(50)['Metabolite'].tolist()
                    hm_data = df_sub.set_index(group_col)[hm_feats].T
                    hm_data.index = [meta.loc[f, 'Clean_Name'] if (meta is not None and f in meta.index) else f for f in hm_data.index]
                    lut = {g: GROUP_COLORS[i%len(GROUP_COLORS)] for i, g in enumerate(df_sub[group_col].unique())}; col_colors = df_sub[group_col].map(lut)
                    try:
                        g = sns.clustermap(hm_data.astype(float), z_score=0, cmap="vlag", center=0, col_colors=col_colors, figsize=(8, 8))
                        g.ax_heatmap.set_xlabel(""); g.ax_heatmap.set_ylabel("")
                        st.pyplot(g.fig)
                    except: st.warning("热图生成失败")

        with tabs[6]:
            st.markdown("### 🏆 生物标志物清单 (VIP > 1 且 P < 0.05)")
            disp_cols = ['Name', 'Log2_FC', 'P_Value', 'FDR', 'VIP', 'p_corr']
            out_df = stats_df[stats_df['Is_Biomarker']].sort_values('VIP', ascending=False)[disp_cols]
            st.dataframe(out_df.style.format({"Log2_FC":"{:.2f}", "P_Value":"{:.3e}", "FDR":"{:.3e}", "VIP":"{:.2f}", "p_corr":"{:.2f}"}).background_gradient(subset=['VIP'], cmap="Reds"), use_container_width=True)
            st.download_button("📥 导出差异清单 (CSV)", out_df.to_csv(index=False).encode('utf-8'), "Biomarkers.csv", "text/csv")

        # === 核心：调用真实富集引擎 ===
        with tabs[7]:
            st.markdown("### 🕸️ 外部 KEGG 代谢通路富集气泡图")
            st.caption("优先读取您上传的，或 GitHub 中的 `kegg_pathways.csv`（长表两列）。右上角气泡越大越红说明富集越显著。")
            
            c1, c2 = st.columns([1, 6])
            with c2:
                sig_mets_names = stats_df[stats_df['Is_Biomarker']]['Name'].tolist()
                all_mets_names = stats_df['Name'].tolist()
                
                if not sig_mets_names:
                    st.info("⚠️ 无显著差异标志物，无法进行通路富集。")
                else:
                    with st.spinner("正在读取庞大的本地 KEGG 数据库..."):
                        
                        # 核心逻辑：如果用户传了文件就用用户传的，如果没传，自动找仓库根目录里的 kegg_pathways.csv 
                        db_source = custom_pathway_file if custom_pathway_file else "kegg_pathways.csv"
                        pathway_df = run_pathway_enrichment(sig_mets_names, all_mets_names, custom_db_source=db_source)
                        
                        if pathway_df.empty:
                            st.warning("未能匹配到通路。如果使用了外部 CSV 库，请确保包含 `Pathway` 和 `Metabolite` 两列。")
                        else:
                            plot_pw_df = pathway_df[pathway_df['Hits'] > 0].head(15)
                            
                            fig_pathway = px.scatter(
                                plot_pw_df, x='Enrichment_Factor', y='-Log10_P', size='Hits', color='P_Value',
                                hover_name='Pathway', hover_data={'Hit_Metabolites': True, 'P_Value': ':.4f', 'Enrichment_Factor': ':.2f'},
                                color_continuous_scale='Reds_r', size_max=40
                            )
                            fig_pathway.update_layout(
                                template="simple_white", width=800, height=600,
                                title={'text': "Pathway Enrichment Bubble Plot", 'y':0.95, 'x':0.5, 'xanchor': 'center'},
                                xaxis_title="Enrichment Factor (富集倍数)", yaxis_title="-Log10(P-value)",
                                coloraxis_colorbar=dict(title="P-value")
                            )
                            fig_pathway.add_hline(y=-np.log10(0.05), line_dash="dash", line_color="gray", annotation_text="P=0.05")
                            
                            st.plotly_chart(fig_pathway)
                            
                            st.markdown("#### 📖 详细命中统计")
                            st.dataframe(
                                pathway_df.style.format({"P_Value":"{:.3e}", "FDR":"{:.3e}", "Enrichment_Factor":"{:.2f}"}).background_gradient(subset=['P_Value'], cmap="Reds_r", vmin=0, vmax=0.05),
                                use_container_width=True
                            )
