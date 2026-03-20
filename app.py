import streamlit as st
import pandas as pd
import numpy as np
import os
import gc
import datetime
import re
import io
import base64
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

# ==========================================
# 0. 模块导入与配置
# ==========================================
st.set_page_config(page_title="MetaboAnalyst Pro (SIMCA Edition)", page_icon=" 🧬 ", layout="wide")

try:
    from data_preprocessing import data_cleaning_pipeline, parse_metdna_file, merge_multiple_dfs, align_sample_info, OPLS_DA, run_pathway_enrichment
except ImportError as e:
    st.error(" ❌  严重错误：未找到 `data_preprocessing.py` 或内部函数。请确保该文件在同一目录下并已更新！")
    st.stop()

try:
    from serrf_module import serrf_normalization
except ImportError:
    st.warning(" ⚠️  未找到 `serrf_module.py`，SERRF 批次校正功能将被禁用。")

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
    st.header(" 🛠️  数据控制台")
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
            st.caption(f" ✅  已加载 {len(info_df)} 行")
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
        else: st.warning(" ⚠️  需上传 Info 表")

    st.markdown("#### 5. 外部通路库 (可选)")
    
    custom_pathway_file = st.file_uploader("不传则自动读取仓库 kegg_pathways.csv", type=["csv", "gmt"], key="pathway_db")

    st.markdown("#### 6. 上传 MetDNA 数据")
    uploaded_files = st.file_uploader("结果文件 (支持多选)", type=["csv", "xlsx"], accept_multiple_files=True, key="data")
    st.markdown("---")
    start_process = st.container().button(" 📥  开始处理数据", use_container_width=True, type="primary")

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
                            else: st.error(f" ❌  {file.name}: SERRF失败")

                    parsed_results.append((df_t, meta, unique_name))
       
                    del df_t, meta, info_aligned; gc.collect()
                except Exception as e: st.error(f"处理 {file.name} 失败: {str(e)}")
                progress_bar.progress((i + 1) / len(uploaded_files))

            if parsed_results:
                st.session_state.all_sample_ids = sorted(list(set(st.session_state.all_sample_ids) | current_run_samples))
                if len(parsed_results) == 1: st.session_state.raw_df, st.session_state.feature_meta = parsed_results[0][0], parsed_results[0][1]
                else: st.session_state.raw_df, st.session_state.feature_meta, _ = merge_multiple_dfs(parsed_results)
                st.session_state.data_loaded = True
                st.success(" ✅  处理完成！")
                st.rerun() 

# ==========================================
# 4. 统计与可视化展示区
# ==========================================
if st.session_state.data_loaded and st.session_state.raw_df is not None:
    raw_df = st.session_state.raw_df
    st.info(f"数据总览: {len(raw_df)} 样本 x {len(raw_df.columns)-3} 特征")
    csv_data = raw_df.to_csv(index=False).encode('utf-8')
    st.download_button(" 📥  导出清洗前合并数据", csv_data, f"Metabo_{datetime.datetime.now().strftime('%Y%m%d')}.csv", "text/csv")
    st.divider()

    with st.form(key='analysis_form'):
        st.markdown("###  ⚙️  统计与富集分析设置 (已设定为行业金标准)")
        non_num = raw_df.select_dtypes(exclude=[np.number]).columns.tolist()
        group_col = st.selectbox("分组列", non_num, index=non_num.index('Group') if 'Group' in non_num else 0)
        
        with st.expander("数据预处理配置 (SIMCA 标准工作流)", expanded=False):
            filter_option = st.radio("分析范围:", ["全部特征", "仅已注释特征"], index=0)
            c_p1, c_p2 = st.columns(2)
            miss_th = c_p1.slider("剔除缺失率 > X", 0.0, 1.0, 0.20)
            impute_m = c_p2.selectbox("缺失值填充", ["KNN (推荐)", "min", "mean", "zero"], index=0).split()[0]
            
            c_p3, c_p4 = st.columns(2)
            norm_m = c_p3.selectbox("样本归一化", ["PQN (推荐)", "Median", "Sum", "None"], index=0).split()[0]
        
            scale_m = c_p4.selectbox("特征缩放 (Scaling)", ["Pareto (SIMCA 默认)", "Auto (Z-score)", "None"], index=0).split()[0]
            do_log = st.checkbox("Log2 对数转化 (强烈推荐)", value=True)

        cur_grps = sorted(raw_df[group_col].astype(str).unique())
        sel_grps = st.multiselect("纳入对比组 (OPLS-DA 需要严格的 2 组对比)", cur_grps, default=cur_grps[:2] if len(cur_grps)>=2 else cur_grps)
        
        c1, c2, c3, c4 = st.columns(4)
        valid = list(sel_grps)
        
        case = c1.selectbox("Case 组 (实验组)", valid, index=0 if valid else None)
        ctrl = c2.selectbox("Control 组 (对照组)", valid, index=1 if len(valid)>1 else 0)
        p_th = c3.number_input("P-value 阈值", value=0.05, step=0.01)
        fc_th = c4.number_input("Log2 FC 阈值", value=0.58, step=0.10, help="0.58 对应 1.5 倍差异；1.0 对应 2.0 倍差异")
        
        submit_button = st.form_submit_button(label=' 🚀  运行全自动分析 (生成交互图表与离线报告)')

if not st.session_state.data_loaded:
    st.title(" 🧬  MetaboAnalyst Pro (SIMCA Edition)"); st.info(" 👈  请在左侧面板上传并处理数据"); st.stop()

if not submit_button:
    st.title(" ✅  数据准备就绪"); st.dataframe(st.session_state.raw_df.head(50)); st.stop()

if submit_button:
    if len(sel_grps) != 2: st.error(" ⚠️  OPLS-DA 必须且只能选择 2 个组进行对比！"); st.stop()
    
    pathway_df = pd.DataFrame() 
    hm_base64 = ""
    fig_opls = fig_perm = fig_splot = fig_vip = fig_pca = fig_vol = fig_pathway = fig_network = None

    with st.spinner("正在运行分析与网络构建..."):
        raw_df = st.session_state.raw_df; meta = st.session_state.feature_meta
        df_proc, feats = data_cleaning_pipeline(raw_df, group_col, miss_th, impute_m, norm_m, do_log, scale_m)
        
        if filter_option == "仅已注释特征":
            anno_ids = meta[meta['Is_Annotated']==True].index.tolist() if meta is not None else []
            feats = [f for f in feats if f in anno_ids]
        
        df_sub = df_proc[df_proc[group_col].isin(sel_grps)].copy()
 
        
        stats_df = run_pairwise_statistics(df_sub, group_col, case, ctrl, feats)
        if meta is not None and 'Clean_Name' in meta.columns and 'Original_Name' in meta.columns: 
            stats_df = stats_df.merge(meta[['Clean_Name', 'Original_Name']], left_on='Metabolite', right_index=True, how='left')
            stats_df['Name'] = stats_df['Clean_Name'].fillna(stats_df['Metabolite'])
            stats_df['Search_Name'] = stats_df['Original_Name'].fillna(stats_df['Metabolite'])
        else:
    
            stats_df['Name'] = stats_df['Metabolite']
            stats_df['Search_Name'] = stats_df['Metabolite']
            
        # BUG FIX: 增加 Log2FC 的门槛筛选，确保 Biomarker 纯净度
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
        
        # 核心修复点应用在这里
        stats_df['Is_Biomarker'] = (stats_df['VIP'] > 1.0) & (stats_df['P_Value'] < p_th) & (stats_df['Log2_FC'].abs() > fc_th)
        out_df = stats_df[stats_df['Is_Biomarker']].sort_values('VIP', ascending=False)

        st.title(" 📊  综合代谢组学分析报告")
        st.markdown(f"**对比**: {case} vs {ctrl} &nbsp;&nbsp;|&nbsp;&nbsp; **模型**: R²Y = `{R2Y:.3f}` &nbsp;&nbsp;|&nbsp;&nbsp; Q² = `{Q2:.3f}`")
        if b_q2 < 0.05 and Q2 > 0.5: st.success(f" ✅  OPLS-DA 模型优秀且未过拟合！ (Q²截距: {b_q2:.3f})")
        else: st.warning(f" ⚠️  模型可能过拟合，或组间差异不大 (Q²截距: {b_q2:.3f})")

        # 增加网络图 Tab
        tabs = st.tabs([" 🎯  OPLS-DA", " 🔄  置换检验", " 🧬  S-Plot", " 📊  VIP", " 🌐  PCA", " 🌋  火山/热图", " 📑  清单", " 🕸️  通路富集", " 🔗  机制网络图", " 📄  导出报告与AI助手"])
        
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
                fig_opls = update_layout_square(fig_opls, "OPLS-DA Score Plot", "t [1]", "to [1]")
                st.plotly_chart(fig_opls)

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
                fig_splot = update_layout_square(fig_splot, "S-Plot", "Log2 Fold Change", "p(corr)")
                st.plotly_chart(fig_splot)

        with tabs[3]:
            c1, c2 = st.columns([1, 6])
            with c2:
                top_vip_df = stats_df.sort_values('VIP', ascending=True).tail(25)
  
                fig_vip = px.bar(top_vip_df, x="VIP", y="Name", orientation='h', color="VIP", color_continuous_scale="RdBu_r")
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
                    fig_pca = update_layout_square(fig_pca, "PCA (QC Check)", f"PC1 ({var[0]:.1%})", f"PC2 ({var[1]:.1%})")
                    st.plotly_chart(fig_pca)

        with tabs[5]:
            c1, c2 = st.columns(2)
            with c1:
                fig_vol = px.scatter(stats_df, x="Log2_FC", y="-Log10_P", color="Sig", color_discrete_map=COLOR_PALETTE, hover_data=['Name', 'VIP'])
                fig_vol.add_hline(y=-np.log10(p_th), line_dash="dash", line_color="gray")
                fig_vol.add_vline(x=fc_th, line_dash="dash", line_color="gray"); fig_vol.add_vline(x=-fc_th, line_dash="dash", line_color="gray")
                fig_vol = update_layout_square(fig_vol, "Volcano Plot", "Log2 Fold Change", "-Log10(P-value)")
                st.plotly_chart(fig_vol, use_container_width=True)
            with c2:
                sig_mets = out_df['Metabolite'].tolist()
                if not sig_mets: st.info("无满足要求的差异代谢物")
       
                else:
                    hm_feats = out_df.head(50)['Metabolite'].tolist()
                    hm_data = df_sub.set_index(group_col)[hm_feats].T
                    hm_data.index = [meta.loc[f, 'Clean_Name'] if (meta is not None and f in meta.index) else f for f in hm_data.index]
         
                    lut = {g: GROUP_COLORS[i%len(GROUP_COLORS)] for i, g in enumerate(df_sub[group_col].unique())}; col_colors = df_sub[group_col].map(lut)
                    try:
                        g = sns.clustermap(hm_data.astype(float), z_score=0, cmap="vlag", center=0, col_colors=col_colors, figsize=(8, 8))
                        g.ax_heatmap.set_xlabel(""); g.ax_heatmap.set_ylabel("")
                        st.pyplot(g.fig)
                        
                        buf = io.BytesIO()
                        g.savefig(buf, format='png', bbox_inches='tight')
                        buf.seek(0)
                        hm_base64 = base64.b64encode(buf.read()).decode('utf-8')
                    except Exception as e: st.warning(f"热图生成失败: {e}")

        with tabs[6]:
            st.markdown("###  🏆  生物标志物清单")
            disp_cols = ['Name', 'Log2_FC', 'P_Value', 'FDR', 'VIP', 'p_corr']
            st.dataframe(out_df[disp_cols].style.format({"Log2_FC":"{:.2f}", "P_Value":"{:.3e}", "FDR":"{:.3e}", "VIP":"{:.2f}", "p_corr":"{:.2f}"}).background_gradient(subset=['VIP'], cmap="Reds"), use_container_width=True)

        with tabs[7]:
            st.markdown("###  🕸️  KEGG 代谢通路富集")
            c1, c2 = st.columns([1, 6])
            with c2:
        
                sig_mets_fullnames = stats_df[stats_df['Is_Biomarker']]['Search_Name'].tolist()
                all_mets_fullnames = stats_df['Search_Name'].tolist()
                
                if not sig_mets_fullnames: st.info(" ⚠️  无显著差异标志物，无法进行通路富集。")
                else:
                  
                    with st.spinner("正在映射数据库..."):
                        db_source = custom_pathway_file if custom_pathway_file else "kegg_pathways.csv"
                        pathway_df = run_pathway_enrichment(sig_mets_fullnames, all_mets_fullnames, custom_db_source=db_source)
                        
               
                        if pathway_df.empty: st.warning("未能匹配到通路。")
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
                                xaxis_title="Enrichment Factor", yaxis_title="-Log10(P-value)",
                                
                                coloraxis_colorbar=dict(title="P-value")
                            )
                            fig_pathway.add_hline(y=-np.log10(0.05), line_dash="dash", line_color="gray")
                            st.plotly_chart(fig_pathway)
              
                            st.dataframe(pathway_df.style.format({"P_Value":"{:.3e}", "FDR":"{:.3e}", "Enrichment_Factor":"{:.2f}"}).background_gradient(subset=['P_Value'], cmap="Reds_r", vmin=0, vmax=0.05), use_container_width=True)

        # ===============================================
        # 终极新增：代谢重编程机制网络图 (Network)
        # ===============================================
        with tabs[8]:
            st.markdown("###  🔗  代谢重编程机制网络 (Pathway-Metabolite Network)")
            st.caption("展示显著富集通路（P < 0.05）与核心标志物的相互关联。方块代表通路，圆点代表代谢物（红色上调，蓝色下调）。节点越大代表富集度或 VIP 越高。")
         
            
            if pathway_df.empty or out_df.empty:
                st.info("需要产生显著的富集通路和差异代谢物后，才能构建重编程网络。")
            else:
                sig_pws = pathway_df[pathway_df['P_Value'] < 0.05]
                if sig_pws.empty:
                 
                    st.info("当前组别对比下没有 P < 0.05 的显著富集通路，无法绘制网络。")
                else:
                    # 1. 构建 NetworkX 图
                    G = nx.Graph()
                    
           
                    # 提取显著标志物的折叠字典 (Name -> Log2FC)
                    fc_dict = dict(zip(out_df['Name'], out_df['Log2_FC']))
                    vip_dict = dict(zip(out_df['Name'], out_df['VIP']))
                    
                    # 2. 添加节点和边
                    for _, row in sig_pws.iterrows():
                        pw_name = row['Pathway']
                        # 添加通路节点 (权重为 -log10P 加上基础大小)
                     
                        G.add_node(pw_name, node_type='pathway', size=max(15, -np.log10(row['P_Value']) * 10))
                        
                        hits_str = row['Hit_Metabolites']
                        if pd.notna(hits_str) and str(hits_str).strip() != "":
             
                            hits = [m.strip() for m in hits_str.split(',')]
                            for hit in hits:
                                if hit in fc_dict:
             
                                    # 添加代谢物节点
                                    G.add_node(hit, node_type='metabolite', size=max(10, vip_dict.get(hit, 1.0) * 8), fc=fc_dict[hit])
                                 
                                    # 添加连线
                                    G.add_edge(pw_name, hit)

                    if len(G.nodes) > 0:
                        # 3. 计算布局 (Fruchterman-Reingold force-directed algorithm)
       
                        pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
                        
                        edge_x, edge_y = [], []
                        for edge in G.edges():
                            x0, y0 = pos[edge[0]]
                            x1, y1 = pos[edge[1]]
                            edge_x.extend([x0, x1, None])
       
                            edge_y.extend([y0, y1, None])

                        edge_trace = go.Scatter(
                            x=edge_x, y=edge_y, line=dict(width=1, color='#888'),
                    
                            hoverinfo='none', mode='lines'
                        )

                        node_x, node_y, node_text, node_color, node_size, node_symbol = [], [], [], [], [], []
                        for node in G.nodes():
    
                            x, y = pos[node]
                            node_x.append(x)
                            node_y.append(y)
                 
                            
                            node_info = G.nodes[node]
                            if node_info['node_type'] == 'pathway':
                            
                                node_color.append('#FFD700') # 金色代表通路
                                node_size.append(node_info['size'])
                                node_symbol.append('square')
                              
                                node_text.append(f"<b>[Pathway]</b> {node}")
                            else:
                                fc = node_info['fc']
                                color = '#CD0000' if fc > 0 else '#00008B' # 红色上调，蓝色下调
                                node_color.append(color)
                                node_size.append(node_info['size'])
                             
                                node_symbol.append('circle')
                                node_text.append(f"<b>{node}</b><br>Log2FC: {fc:.2f}")

                        node_trace = go.Scatter(
                            x=node_x, y=node_y, mode='markers+text',
        
                            hoverinfo='text', text=[n if G.nodes[n]['node_type']=='pathway' else '' for n in G.nodes()], 
                            textposition="top center",
                            hovertext=node_text,
             
                            marker=dict(
                                symbol=node_symbol, showscale=False,
                                color=node_color, size=node_size,
                   
                                line_width=1, line_color='black'
                            )
                        )

                        fig_network = go.Figure(data=[edge_trace, node_trace])
                        
                        # -------------------------
                        # 核心修复点：将 Layout 严格的 kwargs 替换为 update_layout
                        # -------------------------
                        fig_network.update_layout(
                            title='<br>Metabolic Reprogramming Mechanism Network',
                            title_font_size=16,
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20,l=5,r=5,t=40),
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            width=900, 
                            height=700, 
                            plot_bgcolor='white'
                        )
                        st.plotly_chart(fig_network)
                        st.info(" 💡  **绘图指南**：黄色的方块代表代谢通路，红蓝色的圆圈代表标志物。这个拓扑网络是您在 BioRender 或 Illustrator 中手绘高分 SCI 文章机制示意图的完美草稿。")

        # ===============================================
        # 离线 HTML 报告与 AI 提示词
        # ===============================================
        with tabs[9]:
            st.markdown("###  📄  报告生成中心")
       
            st.caption("一键生成面向专家的可视化汇总报告，以及喂给 AI 的文章起草 Prompt。")
            
            c_rep1, c_rep2 = st.columns(2)
            
            with c_rep1:
                st.markdown("####  👨 ‍ 🔬  1. 完整可视化报告下载 (HTML)")
                st.write("打包了本次分析的所有参数、数据表格和交互式图表。本文件为 100% 离线版，无需网络即可使用任意浏览器秒开，支持缩放与截取出版级图片。")
                
                js_added = [False] 
                def get_html_plot(fig):
                    if fig is not None:
                      
                        if not js_added[0]:
                            js_added[0] = True
                            return fig.to_html(full_html=False, include_plotlyjs=True)
                        else:
            
                            return fig.to_html(full_html=False, include_plotlyjs=False)
                    return "<p style='color:red;'>未生成该图表</p>"
                
                html_report = f"""
                <!DOCTYPE html>
         
                <html>
                <head>
                    <meta charset="utf-8">
                    <title>代谢组学综合分析报告 | {case} vs {ctrl}</title>
                    <style>
                        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 40px auto; max-width: 1100px; color: #333; line-height: 1.6; background-color: #f4f7f6; }}
                        .container {{ background-color: #fff; padding: 40px; border-radius: 10px; box-shadow: 0 4px 10px rgba(0,0,0,0.1); }}
                        h1 {{ color: #2c3e50; text-align: center; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
                        h2 {{ color: #2980b9; margin-top: 40px; border-left: 4px solid #2980b9; padding-left: 10px; }}
                        table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; font-size: 14px; text-align: left; }}
                        th, td {{ border: 1px solid #ddd; padding: 10px; }}
                        th {{ background-color: #f8f9fa; color: #2c3e50; }}
                        tr:nth-child(even) {{ background-color: #f9f9f9; }}
                        .plot-box {{ margin: 30px 0; padding: 15px; border: 1px solid #eee; border-radius: 8px; background: #fafafa; text-align: center; }}
                        .metric-container {{ display: flex; justify-content: space-around; background: #eef2f5; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
                        .metric {{ text-align: center; }}
                        .metric-title {{ font-size: 14px; color: #7f8c8d; text-transform: uppercase; }}
                        .metric-value {{ font-size: 28px; font-weight: bold; color: #e74c3c; }}
                    </style>
                </head>
                <body>
                <div class="container">
                    <h1>代谢组学综合分析报告 (SIMCA 规范)</h1>
     
               
                    <h2>1. 实验项目与参数</h2>
                    <ul>
                        <li><b>对比组别:</b> <code>{case}</code> (实验组) vs <code>{ctrl}</code> (对照组)</li>
                        <li><b>数据规模:</b> 鉴定出 {len(feats)} 个特征</li>
                        
                        <li><b>筛选标准:</b> VIP > 1.0, P-value < {p_th}, |Log2 FC| > {fc_th}</li>
                        <li><b>预处理策略:</b> {norm_m} + {scale_m} Scaling</li>
                    </ul>
                    
                    <h2>2. OPLS-DA 模型评估</h2>
                    <div class="metric-container">
                        <div class="metric"><div class="metric-title">R²Y (模型解释率)</div><div class="metric-value">{R2Y:.3f}</div></div>
                        <div class="metric"><div class="metric-title">Q² (模型预测率)</div><div class="metric-value">{Q2:.3f}</div></div>
                      
                        <div class="metric"><div class="metric-title">Q² 置换检验截距</div><div class="metric-value">{b_q2:.3f}</div></div>
                    </div>
                    <p style="text-align: center;"><b>结论:</b> {"该 OPLS-DA 模型分离度极佳且未发生过拟合，预测结果高度可靠。" if (b_q2 < 0.05 and Q2 > 0.5) else "模型分离度一般或存在轻微过拟合，提示两组间代谢差异可能不显著。"}</p>
                    <h2>3. 核心差异代谢物清单 (Top 25 Biomarkers)</h2>
                    {out_df[['Name', 'Log2_FC', 'P_Value', 'VIP', 'p_corr']].head(25).to_html(index=False, float_format="%.3f")}
                    
                    <h2>4. KEGG 代谢通路富集分析 (Top 15)</h2>
                    {pathway_df[['Pathway', 'Total_in_Pathway', 'Hits', 'Enrichment_Factor', 'P_Value']].head(15).to_html(index=False, float_format="%.4f") if not pathway_df.empty else "<p>未进行通路富集分析或无显著命中。</p>"}
                    
                    <h2>5. 统计与多维可视化图表</h2>
                    <p><i>注：本报告为纯离线交互版。图表支持鼠标悬停、框选缩放，点击图表右上角相机图标  📷  即可下载透明底色高清图片。</i></p>
                    
                    <div class="plot-box"><h3>(1) OPLS-DA 得分图</h3>{get_html_plot(fig_opls)}</div>
                    <div class="plot-box"><h3>(2) 置换检验 (Permutation Test)</h3>{get_html_plot(fig_perm)}</div>
         
                    <div class="plot-box"><h3>(3) S-Plot</h3>{get_html_plot(fig_splot)}</div>
                    <div class="plot-box"><h3>(4) 火山图</h3>{get_html_plot(fig_vol)}</div>
                    <div class="plot-box"><h3>(5) PCA 宏观质控得分图</h3>{get_html_plot(fig_pca)}</div>
                """
                if hm_base64:
         
                    html_report += f'<div class="plot-box"><h3>(6) Top 50 差异代谢物聚类热图</h3><img src="data:image/png;base64,{hm_base64}" style="max-width:100%; border:1px solid #ccc;"/></div>'
                if 'fig_pathway' in locals() and fig_pathway is not None:
                    html_report += f'<div class="plot-box"><h3>(7) KEGG 通路富集气泡图</h3>{get_html_plot(fig_pathway)}</div>'
                if 'fig_network' in locals() and fig_network is not None:
                    html_report += f'<div class="plot-box"><h3>(8) 代谢重编程机制网络图</h3>{get_html_plot(fig_network)}</div>'
 
                html_report += """
                </div>
                </body>
                </html>
                """
                
   
                st.download_button(" 📥  下载完整交互式网页报告 (.html)", html_report.encode('utf-8'), f"Metabolomics_Report_{case}_vs_{ctrl}.html", "text/html", type="primary")
            with c_rep2:
                st.markdown("####  🤖  2. AI 撰稿专属 Prompt")
                st.write("直接点击右下方拷贝按钮，或下载为 `.md` 发给 ChatGPT / Claude，让它立刻帮您写出 SCI 级别的 Results 和 Discussion 段落。")
                
                num_up = len(out_df[out_df['Log2_FC'] > 0]); num_down = len(out_df[out_df['Log2_FC'] < 0])
                top_mets_str = out_df[['Name', 'Log2_FC', 'P_Value', 'VIP']].head(15).to_markdown(index=False) if not out_df.empty else "无显著差异物"
                pw_str = "无显著富集通路"
                if not pathway_df.empty:
                    sig_pws = pathway_df[pathway_df['P_Value'] < 0.05].head(10)
          
                    if not sig_pws.empty: pw_str = sig_pws[['Pathway', 'Hits', 'P_Value']].to_markdown(index=False)
                
                prompt_md = f"""请作为一名资深的生物信息学和代谢组学专家，根据以下我提供的代谢组学数据分析结果，帮我撰写一篇英文科研论文的 **Results（结果）** 和 **Discussion（讨论）** 部分。
###  🔬  1. 实验参数与模型质控
- **对比组别**: {case} (Case) vs {ctrl} (Control)
- **预处理与缩放**: {norm_m} + {scale_m} Scaling
- **OPLS-DA 模型评估**: R²Y = {R2Y:.3f}, Q² = {Q2:.3f}, 置换检验 Q² 截距 = {b_q2:.3f} (模型{"稳健且未过拟合" if (b_q2<0.05 and Q2>0.5) else "预测能力一般"})。
###  🧬  2. 差异生物标志物 (Biomarkers)
- 筛选阈值: VIP > 1.0 且 P-value < {p_th}。
- 整体情况: 共找到 {num_up + num_down} 个标志物，其中 {num_up} 个在 {case} 组中显著上调，{num_down} 个显著下调。
- **Top 15 核心标志物清单**:
{top_mets_str}
###  🕸️  3. KEGG 代谢通路富集 (P < 0.05)
{pw_str}
---
###  📝  撰写要求：
1. **Results 部分**：总结 OPLS-DA 模型的分离情况和置换检验结果；描述差异代谢物的整体分布；客观描述上述显著富集的 KEGG 通路。
2. **Discussion 部分**：结合上述查出的 Top 标志物和关键通路，查阅最新生化医学文献，深入探讨 {case} 组相对于 {ctrl} 组发生这些代谢网络改变的生理/病理机制，以及潜在的临床指导意义。
"""
                st.text_area("拷贝此文本发送给 AI:", value=prompt_md, height=250)
                st.download_button(" 📥  下载 Prompt 文件 (.md)", prompt_md.encode('utf-8'), f"AI_Prompt_{case}_vs_{ctrl}.md", "text/markdown")
