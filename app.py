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
from statsmodels.stats.multitest import multipletests

# ==========================================
# 0. 模块导入与错误捕获
# ==========================================
st.set_page_config(page_title="MetaboAnalyst Pro (SIMCA Edition)", page_icon="🧬", layout="wide")

try:
    from data_preprocessing import data_cleaning_pipeline, parse_metdna_file, merge_multiple_dfs, align_sample_info, OPLS_DA
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
# 1. 辅助函数
# ==========================================
def update_layout_square(fig, title="", x_title="", y_title=""):
    fig.update_layout(template="simple_white", width=600, height=600, title={'text': title, 'y':0.95, 'x':0.5, 'xanchor': 'center'}, xaxis=dict(title=x_title, showline=True, linewidth=2, mirror=True), yaxis=dict(title=y_title, showline=True, linewidth=2, mirror=True), legend=dict(yanchor="top", y=1, xanchor="left", x=1.15))
    return fig

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

    st.markdown("#### 5. 上传 MetDNA 数据")
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
# 4. 统计与可视化展示区 (SIMCA OPLS-DA)
# ==========================================
if st.session_state.data_loaded and st.session_state.raw_df is not None:
    raw_df = st.session_state.raw_df
    st.info(f"数据总览: {len(raw_df)} 样本 x {len(raw_df.columns)-3} 特征")
    csv_data = raw_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 导出清洗前合并数据", csv_data, f"Metabo_{datetime.datetime.now().strftime('%Y%m%d')}.csv", "text/csv")
    st.divider()

    with st.form(key='analysis_form'):
        st.markdown("### ⚙️ SIMCA 统计分析设置")
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
        submit_button = st.form_submit_button(label='🚀 运行 SIMCA 风格分析')

if not st.session_state.data_loaded:
    st.title("🧬 MetaboAnalyst Pro (SIMCA Edition)"); st.info("👈 请在左侧面板上传并处理数据"); st.stop()

if not submit_button:
    st.title("✅ 数据准备就绪"); st.dataframe(st.session_state.raw_df.head(50)); st.stop()

if submit_button:
    if len(sel_grps) != 2: st.error("⚠️ OPLS-DA 必须且只能选择 2 个组进行对比！"); st.stop()
    with st.spinner("正在运行 OPLS-DA 和统计计算..."):
        raw_df = st.session_state.raw_df; meta = st.session_state.feature_meta
        df_proc, feats = data_cleaning_pipeline(raw_df, group_col, miss_th, impute_m, norm_m, do_log, scale_m)
        
        if filter_option == "仅已注释特征":
            anno_ids = meta[meta['Is_Annotated']==True].index.tolist() if meta is not None else []
            feats = [f for f in feats if f in anno_ids]
        
        df_sub = df_proc[df_proc[group_col].isin(sel_grps)].copy()
        
        # 1. 常规统计 (T-test, FC)
        stats_df = run_pairwise_statistics(df_sub, group_col, case, ctrl, feats)
        if meta is not None: stats_df = stats_df.merge(meta[['Clean_Name']], left_on='Metabolite', right_index=True, how='left')
        stats_df['Name'] = stats_df['Clean_Name'] if 'Clean_Name' in stats_df.columns else stats_df['Metabolite']
        stats_df['Sig'] = 'NS'
        stats_df.loc[(stats_df['P_Value']<p_th)&(stats_df['Log2_FC']>fc_th), 'Sig']='Up'
        stats_df.loc[(stats_df['P_Value']<p_th)&(stats_df['Log2_FC']<-fc_th), 'Sig']='Down'

        # 2. 运行 SIMCA OPLS-DA
        y_binary = np.where(df_sub[group_col] == case, 1, -1)
        X_matrix = df_sub[feats].values
        
        opls = OPLS_DA().fit(X_matrix, y_binary)
        R2Y, Q2 = opls.evaluate(X_matrix, y_binary)
        
        # 提取 VIP 和 p_corr
        vip_df = pd.DataFrame({'Metabolite': feats, 'VIP': opls.vip, 'p_corr': opls.p_corr})
        stats_df = stats_df.merge(vip_df, on='Metabolite')
        stats_df['Is_Biomarker'] = (stats_df['VIP'] > 1.0) & (stats_df['P_Value'] < p_th)

        st.title("📊 SIMCA 风格多维统计报告")
        st.markdown(f"**对比**: {case} vs {ctrl} &nbsp;&nbsp;|&nbsp;&nbsp; **模型评估**: R²Y = `{R2Y:.3f}` &nbsp;&nbsp;|&nbsp;&nbsp; Q² = `{Q2:.3f}`")
        if Q2 > 0.5: st.success("✅ 该 OPLS-DA 模型预测能力良好 (Q² > 0.5)")
        else: st.warning("⚠️ 该模型预测能力较弱，组间差异可能不明显或存在噪音干扰 (Q² < 0.5)")

        tabs = st.tabs(["🎯 OPLS-DA 纯净得分图", "🧬 S-Plot 标志物图", "🌋 火山图", "🔥 热图", "📑 差异清单"])
        
        with tabs[0]:
            c1, c2 = st.columns([1, 4])
            with c2:
                opls_score_df = pd.DataFrame({'t1 (Predictive)': opls.t, 't_ortho (Orthogonal)': opls.t_ortho, 'Group': df_sub[group_col].values})
                fig_opls = px.scatter(opls_score_df, x='t1 (Predictive)', y='t_ortho (Orthogonal)', color='Group', symbol='Group', color_discrete_sequence=GROUP_COLORS)
                fig_opls.update_traces(marker=dict(size=14, line=dict(width=1, color='black'), opacity=0.9))
                fig_opls.add_hline(y=0, line_dash="dash", line_color="gray"); fig_opls.add_vline(x=0, line_dash="dash", line_color="gray")
                st.plotly_chart(update_layout_square(fig_opls, "OPLS-DA Score Plot", "t [1] (组间预测差异)", "to [1] (组内正交噪音)"))

        with tabs[1]:
            c1, c2 = st.columns([1, 4])
            with c2:
                splot_df = stats_df.copy()
                splot_df['Color'] = np.where(splot_df['Is_Biomarker'], 'VIP>1 & P<0.05', 'NS')
                fig_splot = px.scatter(splot_df, x='Log2_FC', y='p_corr', color='Color', hover_data=['Name', 'VIP'], color_discrete_map={'VIP>1 & P<0.05': '#CD0000', 'NS': '#E0E0E0'})
                fig_splot.add_hline(y=0.5, line_dash="dash"); fig_splot.add_hline(y=-0.5, line_dash="dash")
                st.plotly_chart(update_layout_square(fig_splot, "S-Plot (p(corr) vs FC)", "Log2 Fold Change", "p(corr) (模型相关性系数)"))

        with tabs[2]:
            c1, c2 = st.columns([1, 4])
            with c2:
                fig_vol = px.scatter(stats_df, x="Log2_FC", y="-Log10_P", color="Sig", color_discrete_map=COLOR_PALETTE, hover_data=['Name', 'VIP'])
                fig_vol.add_hline(y=-np.log10(p_th), line_dash="dash", line_color="gray")
                fig_vol.add_vline(x=fc_th, line_dash="dash", line_color="gray")
                fig_vol.add_vline(x=-fc_th, line_dash="dash", line_color="gray")
                st.plotly_chart(update_layout_square(fig_vol, "Volcano Plot", "Log2 Fold Change", "-Log10(P-value)"))

        with tabs[3]:
            sig_mets = stats_df[stats_df['Is_Biomarker']]['Metabolite'].tolist()
            if not sig_mets: st.info("没有找到满足要求 (VIP>1 且 P<0.05) 的代谢物。")
            else:
                hm_feats = stats_df.sort_values('VIP', ascending=False).head(50)['Metabolite'].tolist()
                hm_data = df_sub.set_index(group_col)[hm_feats].T
                hm_data.index = [meta.loc[f, 'Clean_Name'] if (meta is not None and f in meta.index) else f for f in hm_data.index]
                lut = {g: GROUP_COLORS[i%len(GROUP_COLORS)] for i, g in enumerate(df_sub[group_col].unique())}
                col_colors = df_sub[group_col].map(lut)
                try:
                    g = sns.clustermap(hm_data.astype(float), z_score=0, cmap="vlag", center=0, col_colors=col_colors, figsize=(10, 12))
                    g.ax_heatmap.set_xlabel(""); g.ax_heatmap.set_ylabel("")
                    st.pyplot(g.fig)
                except: st.warning("数据中存在方差为 0 的行，热图生成失败。")

        with tabs[4]:
            st.markdown("### 🏆 候选生物标志物清单 (依据: VIP > 1 且 P < 0.05)")
            disp_cols = ['Name', 'Log2_FC', 'P_Value', 'FDR', 'VIP', 'p_corr']
            out_df = stats_df[stats_df['Is_Biomarker']].sort_values('VIP', ascending=False)[disp_cols]
            st.dataframe(out_df.style.format({"Log2_FC":"{:.2f}", "P_Value":"{:.3e}", "FDR":"{:.3e}", "VIP":"{:.2f}", "p_corr":"{:.2f}"}).background_gradient(subset=['VIP'], cmap="Reds"), use_container_width=True)
            st.download_button("📥 导出差异清单 (CSV)", out_df.to_csv(index=False).encode('utf-8'), "Biomarkers.csv", "text/csv")
