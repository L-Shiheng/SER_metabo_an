import streamlit as st
import pandas as pd
import numpy as np
import os
import gc
import datetime
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

# ==========================================
# 0. 基础配置
# ==========================================
st.set_page_config(page_title="MetaboAnalyst Pro", page_icon="🧬", layout="wide")

COLOR_PALETTE = {'Up': '#CD0000', 'Down': '#00008B', 'NS': '#E0E0E0'} 
GROUP_COLORS = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F', '#8491B4', '#91D1C2', '#DC0000', '#7E6148', '#B09C85']

st.markdown("""
<style>
    .block-container {padding-top: 2rem !important; padding-bottom: 3rem !important;}
    h1, h2, h3, div, p {font-family: 'Arial', sans-serif; color: #2c3e50;}
    button[data-baseweb="tab"] {font-weight: bold;}
    div[data-testid="stForm"] button {
        width: 100%; background-color: #ff4b4b; color: white; font-weight: bold; border: none; padding: 0.5rem;
    }
    .process-btn button {
        width: 100%; background-color: #4CAF50 !important; color: white !important; font-weight: bold; border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

try:
    from data_preprocessing import data_cleaning_pipeline, parse_metdna_file, merge_multiple_dfs, apply_sample_info, align_sample_info
except ImportError:
    st.error("❌ 严重错误：未找到 'data_preprocessing.py'。")
    st.stop()
try:
    from serrf_module import serrf_normalization
except ImportError:
    pass

# ==========================================
# 1. 科学计算与绘图
# ==========================================
def update_layout_square(fig, title="", x_title="", y_title="", width=600, height=600):
    fig.update_layout(
        template="simple_white", width=width, height=height,
        title={'text': title, 'y':0.95, 'x':0.5, 'xanchor': 'center'},
        xaxis=dict(title=x_title, showline=True, linewidth=2, mirror=True),
        yaxis=dict(title=y_title, showline=True, linewidth=2, mirror=True),
        legend=dict(yanchor="top", y=1, xanchor="left", x=1.15),
        margin=dict(l=80, r=180, t=80, b=80)
    )
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
    ell_x = width/2 * np.cos(t)
    ell_y = height/2 * np.sin(t)
    rad = np.radians(theta)
    R = np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])
    ell_coords = np.dot(R, np.array([ell_x, ell_y]))
    return ell_coords[0] + mean_x, ell_coords[1] + mean_y

def calculate_vips(model):
    t = model.x_scores_; w = model.x_weights_; q = model.y_loadings_
    p, h = w.shape; vips = np.zeros((p,))
    s = np.diag(t.T @ t @ q.T @ q); total_s = np.sum(s)
    for i in range(p):
        weight = np.array([(w[i, j] / np.linalg.norm(w[:, j]))**2 for j in range(h)])
        vips[i] = np.sqrt(p * (s @ weight) / total_s)
    return vips

@st.cache_data
def run_pairwise_statistics(df, group_col, case, control, features, equal_var=False):
    g1 = df[df[group_col] == case]
    g2 = df[df[group_col] == control]
    res = []
    for f in features:
        v1, v2 = g1[f].values, g2[f].values
        fc = np.mean(v1) - np.mean(v2) 
        try: t, p = stats.ttest_ind(v1, v2, equal_var=equal_var)
        except: p = 1.0
        if np.isnan(p): p = 1.0
        res.append({'Metabolite': f, 'Log2_FC': fc, 'P_Value': p})
    res_df = pd.DataFrame(res).dropna()
    if not res_df.empty:
        _, p_corr, _, _ = multipletests(res_df['P_Value'], method='fdr_bh')
        res_df['FDR'] = p_corr
        res_df['-Log10_P'] = -np.log10(res_df['P_Value'])
    else: res_df['FDR'] = 1.0; res_df['-Log10_P'] = 0
    return res_df

# ==========================================
# 2. Session State
# ==========================================
if 'raw_df' not in st.session_state: st.session_state.raw_df = None
if 'feature_meta' not in st.session_state: st.session_state.feature_meta = None
if 'data_loaded' not in st.session_state: st.session_state.data_loaded = False
if 'qc_report' not in st.session_state: st.session_state.qc_report = {}

# ==========================================
# 3. 侧边栏
# ==========================================
with st.sidebar:
    st.header("🛠️ 数据控制台")
    
    # 1. Sample Info
    st.markdown("#### 1. 上传 Sample Info (SERRF 必选)")
    sample_info_file = st.file_uploader("Sample Info (.csv/.xlsx)", type=["csv", "xlsx"], key="info")
    info_df = None
    if sample_info_file:
        try:
            if sample_info_file.name.endswith('.csv'): info_df = pd.read_csv(sample_info_file)
            else: info_df = pd.read_excel(sample_info_file)
            st.caption(f"✅ 已加载 {len(info_df)} 行样本信息")
        except: st.error("文件读取失败")

    # 2. Data Scope (Global Setting) - 逻辑修正：放到 SERRF 前面
    st.markdown("#### 2. 数据处理范围")
    # 这是一个全局设置，决定了后面所有操作的特征基数
    feature_scope = st.radio("加载特征范围:", ["仅已注释特征 (推荐)", "全部特征"], index=0, 
                           help="【仅已注释】：仅加载有名字的特征，速度快，适合发现生物学意义。\n【全部特征】：加载所有信号（包含未知物），适合全面探索。")

    # 3. SERRF Setting
    st.markdown("#### 3. SERRF 批次校正")
    use_serrf = st.checkbox("启用 SERRF 校正", value=False)
    serrf_ready = False
    
    if use_serrf:
        if info_df is not None:
            # Auto-Detect Columns
            cols = list(info_df.columns)
            cols_lower = [c.lower() for c in cols]
            
            order_candidates = [i for i, c in enumerate(cols_lower) if any(x in c for x in ['order', 'run', 'idx', 'seq'])]
            idx_order = order_candidates[0] if order_candidates else 0
            
            type_candidates = [i for i, c in enumerate(cols_lower) if any(x in c for x in ['class', 'type', 'group'])]
            final_type_idx = type_candidates[0] if type_candidates else 0
            
            # Find QC label
            found_qc_col = False
            for idx in type_candidates:
                if info_df[cols[idx]].astype(str).str.contains('qc', case=False).any():
                    final_type_idx = idx; found_qc_col = True; break
            
            default_qc_label = "QC"
            if found_qc_col:
                type_vals = info_df.iloc[:, final_type_idx].unique().astype(str)
                qc_match = next((v for v in type_vals if 'qc' in v.lower()), "QC")
                default_qc_label = qc_match

            c1, c2, c3 = st.columns(3)
            run_order_col = c1.selectbox("Order列", cols, index=idx_order)
            sample_type_col = c2.selectbox("Type列", cols, index=final_type_idx)
            qc_label = c3.text_input("QC标签", value=default_qc_label)
            serrf_ready = True
        else:
            st.warning("⚠️ 需上传 Info 表才能启用校正")

    # 4. Upload Data
    st.markdown("#### 4. 上传 MetDNA 数据")
    uploaded_files = st.file_uploader("MetDNA文件 (支持多选)", type=["csv", "xlsx"], accept_multiple_files=True, key="data")
    st.markdown("---")
    
    # 5. Button
    process_container = st.container()
    process_container.markdown('<div class="process-btn">', unsafe_allow_html=True)
    start_process = process_container.button("📥 开始处理数据 (Load & Process)")
    process_container.markdown('</div>', unsafe_allow_html=True)

    # ====================
    # 处理逻辑
    # ====================
    if start_process:
        st.session_state.qc_report = {}
        if not uploaded_files:
            st.error("请先上传数据文件！")
        else:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            with st.spinner("正在启动高性能处理引擎 (Vectorized Parsing & SERRF)..."):
                parsed_results = []
                
                for i, file in enumerate(uploaded_files):
                    status_text.text(f"正在处理 ({i+1}/{len(uploaded_files)}): {file.name} ...")
                    try:
                        file.seek(0)
                        file_type = 'csv' if file.name.endswith('.csv') else 'excel'
                        unique_name = f"{os.path.splitext(file.name)[0]}_{i+1}{os.path.splitext(file.name)[1]}"
                        
                        df_t, meta, err = parse_metdna_file(file, unique_name, file_type=file_type)
                        if err: st.warning(f"{file.name}: {err}"); continue
                        
                        # --- 逻辑修正：无论是否 SERRF，都先根据 scope 过滤 ---
                        if feature_scope.startswith("仅已注释"):
                            annotated_ids = meta[meta['Is_Annotated'] == True].index
                            cols_to_keep = ['SampleID', 'Group'] + [c for c in df_t.columns if c in annotated_ids]
                            df_t = df_t[cols_to_keep]
                            meta = meta.loc[meta.index.isin(df_t.columns)]
                            
                        # 对齐 Info
                        info_aligned = None
                        if info_df is not None:
                            info_aligned = align_sample_info(df_t, info_df)
                            g_col = next((c for c in info_aligned.columns if c.lower() in ['group', 'class']), None)
                            if g_col: df_t['Group'] = info_aligned[g_col].fillna(df_t['Group']).values
                        
                        # SERRF (仅在勾选且Info存在时执行)
                        if use_serrf and serrf_ready and info_aligned is not None:
                            n_matched = info_aligned[run_order_col].notna().sum()
                            if n_matched == 0:
                                st.error(f"❌ {file.name}: 样本名匹配失败，跳过校正。")
                                st.session_state.qc_report[unique_name] = {"Status": "Failed (No Match)"}
                            else:
                                if run_order_col in info_aligned.columns and sample_type_col in info_aligned.columns:
                                    num_cols = df_t.select_dtypes(include=[np.number]).columns.tolist()
                                    df_numeric = df_t[num_cols]
                                    
                                    corrected_data, serrf_stats = serrf_normalization(
                                        df_numeric, info_aligned, run_order_col, sample_type_col, qc_label
                                    )
                                    
                                    if corrected_data is not None:
                                        # 智能回滚
                                        rsd_before = serrf_stats['RSD_Before']
                                        rsd_after = serrf_stats['RSD_After']
                                        
                                        if rsd_after > rsd_before:
                                            st.session_state.qc_report[unique_name] = {
                                                "Status": "Skipped (Worse)", "RSD_Before": rsd_before, "RSD_After": rsd_after
                                            }
                                        else:
                                            for c in corrected_data.columns: df_t[c] = corrected_data[c].values
                                            st.session_state.qc_report[unique_name] = {
                                                "Status": "Success", "RSD_Before": rsd_before, "RSD_After": rsd_after
                                            }
                                    else:
                                        st.error(f"❌ {file.name}: SERRF 失败")
                                else:
                                    st.warning(f"{file.name}: 缺少列")

                        parsed_results.append((df_t, meta, unique_name))
                        del df_t, meta, info_aligned
                        gc.collect()

                    except Exception as e:
                        st.error(f"处理 {file.name} 失败: {e}")
                    
                    progress_bar.progress((i + 1) / len(uploaded_files))

                if parsed_results:
                    if len(parsed_results) == 1:
                        st.session_state.raw_df = parsed_results[0][0]
                        st.session_state.feature_meta = parsed_results[0][1]
                    else:
                        m_df, m_meta, m_err = merge_multiple_dfs(parsed_results)
                        if m_err: st.error(m_err)
                        else:
                            st.session_state.raw_df = m_df
                            st.session_state.feature_meta = m_meta
                    
                    st.session_state.data_loaded = True
                    st.success("✅ 处理完成！")
                    st.rerun() 
                else:
                    st.error("加载失败")

    # --- Export ---
    if st.session_state.data_loaded and st.session_state.raw_df is not None:
        raw_df = st.session_state.raw_df
        st.info(f"数据概览: {len(raw_df)} 样本 x {len(raw_df.columns)-2} 特征")
        
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        csv_data = raw_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 导出合并数据", csv_data, f"Metabo_Processed_{ts}.csv", "text/csv")
        st.divider()

        with st.form(key='analysis_form'):
            st.markdown("### ⚙️ 统计分析参数")
            non_num = raw_df.select_dtypes(exclude=[np.number]).columns.tolist()
            default_grp_idx = non_num.index('Group') if 'Group' in non_num else 0
            group_col = st.selectbox("分组列", non_num, index=default_grp_idx)
            
            # 这里的 filter_option 是用于统计时的二次过滤，保持存在
            filter_option = st.radio("统计分析范围:", ["全部特征", "仅已注释特征"], index=0)
            
            with st.expander("数据清洗与归一化 (高级)", expanded=False):
                miss_th = st.slider("剔除缺失率 > X", 0.0, 1.0, 0.5, 0.1)
                
                impute_m_display = st.selectbox("填充方法", ["min (推荐)", "KNN (高精度但慢)", "mean", "zero"], index=0)
                if "min" in impute_m_display: impute_m = "min"
                elif "KNN" in impute_m_display: impute_m = "KNN"
                elif "mean" in impute_m_display: impute_m = "mean"
                else: impute_m = "zero"
                
                norm_m = st.selectbox("样本归一化", ["None", "PQN", "Sum", "Median"], index=1)
                do_log = st.checkbox("Log2 转化", value=True)
                scale_m = st.selectbox("特征缩放", ["None", "Auto", "Pareto"], index=2)

            current_groups = sorted(raw_df[group_col].astype(str).unique())
            st.markdown("### 组别对比")
            selected_groups = st.multiselect("纳入组:", current_groups, default=current_groups[:2] if len(current_groups)>=2 else current_groups)
            c1, c2 = st.columns(2)
            valid_grps_list = list(selected_groups)
            case_grp = c1.selectbox("Exp (Case)", valid_grps_list, index=0 if valid_grps_list else None)
            ctrl_grp = c2.selectbox("Ctrl (Ref)", valid_grps_list, index=1 if len(valid_grps_list)>1 else 0)
            c3, c4 = st.columns(2)
            p_th = c3.number_input("P-value", 0.05, format="%.3f")
            fc_th = c4.number_input("Log2 FC", 1.0)
            use_equal_var = st.checkbox("Student's t-test (Equal Var)", value=True)
            enable_jitter = st.checkbox("火山图抖动", value=True)
            st.markdown("---")
            submit_button = st.form_submit_button(label='🚀 运行统计分析 (Run Stats)')

# ==========================================
# 4. 主面板
# ==========================================
if not st.session_state.data_loaded:
    st.title("🧬 MetaboAnalyst Pro")
    st.info("👈 请在左侧上传数据并点击 **“开始处理数据”** 按钮。")
    st.stop()

if not submit_button:
    st.title("✅ 数据准备就绪")
    if st.session_state.qc_report:
        st.subheader("🔍 SERRF 校正效果评估")
        cols = st.columns(len(st.session_state.qc_report))
        for idx, (fname, report) in enumerate(st.session_state.qc_report.items()):
            with cols[idx % 3]:
                if report['Status'] == 'Success':
                    st.success(f"📄 {fname}")
                    delta = report['RSD_After'] - report['RSD_Before']
                    st.metric("QC RSD", f"{report['RSD_After']:.1f}%", f"{delta:.1f}%", delta_color="inverse")
                elif report['Status'] == 'Skipped (Worse)':
                    st.warning(f"📄 {fname}")
                    delta = report['RSD_After'] - report['RSD_Before']
                    st.metric("QC RSD (回滚)", f"{report['RSD_Before']:.1f}%", f"变差 (+{delta:.1f}%)", delta_color="off")
                else: st.error(f"📄 {fname}: {report['Status']}")
    st.markdown("---")
    st.subheader("原始数据预览")
    st.dataframe(st.session_state.raw_df.head(50))
    st.stop()

if submit_button:
    if len(selected_groups) < 2: st.error("请至少选择 2 个组！"); st.stop()
    
    with st.spinner("正在进行统计分析与绘图 (WebGL加速中)..."):
        raw_df = st.session_state.raw_df
        feature_meta = st.session_state.feature_meta
        
        df_proc, feats = data_cleaning_pipeline(
            raw_df, group_col, missing_thresh=miss_th, impute_method=impute_m, 
            norm_method=norm_m, log_transform=do_log, scale_method=scale_m
        )

        if filter_option == "仅已注释特征":
            if feature_meta is not None:
                annotated_feats = feature_meta[feature_meta['Is_Annotated'] == True].index.tolist()
                feats = [f for f in feats if f in annotated_feats]
                if not feats: st.error("过滤后无特征！"); st.stop()
            else: st.warning("非 MetDNA 数据，无法过滤。")
        
        df_sub = df_proc[df_proc[group_col].isin(selected_groups)].copy()

        if case_grp != ctrl_grp:
            res_stats = run_pairwise_statistics(df_sub, group_col, case_grp, ctrl_grp, feats, equal_var=use_equal_var)
            if feature_meta is not None:
                res_stats = res_stats.merge(feature_meta[['Confidence_Level', 'Clean_Name']], left_on='Metabolite', right_index=True, how='left')
                res_stats['Confidence_Level'] = res_stats['Confidence_Level'].fillna('Unknown')
            else: res_stats['Confidence_Level'] = 'N/A'
            res_stats['Sig'] = 'NS'
            res_stats.loc[(res_stats['P_Value'] < p_th) & (res_stats['Log2_FC'] > fc_th), 'Sig'] = 'Up'
            res_stats.loc[(res_stats['P_Value'] < p_th) & (res_stats['Log2_FC'] < -fc_th), 'Sig'] = 'Down'
            sig_metabolites = res_stats[res_stats['Sig'] != 'NS']['Metabolite'].tolist()
        else: res_stats = pd.DataFrame(); sig_metabolites = []

        st.title("📊 代谢组学分析报告")
        st.caption(f"对比: {case_grp} vs {ctrl_grp} | 特征数: {len(feats)} | Scaling: {scale_m}")

        # QC Check
        qc_mask = df_sub[group_col].astype(str).str.contains('QC', case=False)
        if qc_mask.sum() >= 2:
             with st.expander("🔍 当前数据质量控制 (QC Check)", expanded=True):
                 qc_data = df_sub.loc[qc_mask, feats]
                 qc_rsd = (qc_data.std() / qc_data.mean()) * 100
                 median_rsd = qc_rsd.median()
                 c1, c2 = st.columns([1, 3])
                 c1.metric("QC Median RSD", f"{median_rsd:.1f}%")
                 fig_rsd = px.histogram(qc_rsd, nbins=50, title="QC RSD Distribution", width=600, height=300)
                 fig_rsd.add_vline(x=20, line_dash="dash", line_color="green")
                 c2.plotly_chart(fig_rsd, use_container_width=True)

        tabs = st.tabs(["📊 PCA", "🎯 PLS-DA", "⭐ VIP", "🌋 火山图", "🔥 热图", "📑 详情"])

        with tabs[0]:
            c1, c2 = st.columns([1, 2])
            with c2:
                if len(df_sub) < 3: st.warning("样本不足")
                else:
                    X = StandardScaler().fit_transform(df_sub[feats])
                    pca = PCA(n_components=2).fit(X); pcs = pca.transform(X); var = pca.explained_variance_ratio_
                    fig_pca = px.scatter(x=pcs[:,0], y=pcs[:,1], color=df_sub[group_col], symbol=df_sub[group_col], color_discrete_sequence=GROUP_COLORS, width=600, height=600, render_mode='webgl')
                    fig_pca.update_traces(marker=dict(size=14, line=dict(width=1, color='black'), opacity=0.9))
                    update_layout_square(fig_pca, "PCA Score Plot", f"PC1 ({var[0]:.1%})", f"PC2 ({var[1]:.1%})")
                    st.plotly_chart(fig_pca, use_container_width=False)

        with tabs[1]:
            c1, c2 = st.columns([1, 2])
            with c2:
                if len(df_sub) < 3: st.warning("样本不足")
                else:
                    X_pls = df_sub[feats].values; y_labels = pd.factorize(df_sub[group_col])[0]
                    pls_model = PLSRegression(n_components=2).fit(X_pls, y_labels)
                    plot_df = pd.DataFrame({'C1': pls_model.x_scores_[:,0], 'C2': pls_model.x_scores_[:,1], 'Group': df_sub[group_col].values})
                    fig_pls = px.scatter(plot_df, x='C1', y='C2', color='Group', symbol='Group', color_discrete_sequence=GROUP_COLORS, width=600, height=600, render_mode='webgl')
                    for i, grp in enumerate(selected_groups):
                        sub_g = plot_df[plot_df['Group'] == grp]
                        if len(sub_g) >= 3:
                            ell_x, ell_y = get_ellipse_coordinates(sub_g['C1'], sub_g['C2'])
                            if ell_x is not None: fig_pls.add_trace(go.Scatter(x=ell_x, y=ell_y, mode='lines', line=dict(color=GROUP_COLORS[i%len(GROUP_COLORS)], width=2, dash='dash'), showlegend=False, hoverinfo='skip'))
                    fig_pls.update_traces(marker=dict(size=14, line=dict(width=1.5, color='black'), opacity=1.0))
                    update_layout_square(fig_pls, "PLS-DA Score Plot", "Component 1", "Component 2")
                    st.plotly_chart(fig_pls, use_container_width=False)

        with tabs[2]:
            if 'pls_model' in locals():
                vip_scores = calculate_vips(pls_model); vip_df = pd.DataFrame({'Metabolite': feats, 'VIP': vip_scores})
                if feature_meta is not None: vip_df = vip_df.merge(feature_meta[['Clean_Name']], left_on='Metabolite', right_index=True, how='left'); vip_df['Display_Name'] = vip_df['Clean_Name'].fillna(vip_df['Metabolite'])
                else: vip_df['Display_Name'] = vip_df['Metabolite']
                top_vip = vip_df.sort_values('VIP', ascending=True).tail(25)
                c1, c2 = st.columns([1, 6])
                with c2:
                    fig_vip = px.bar(top_vip, x="VIP", y="Display_Name", orientation='h', color="VIP", color_continuous_scale="RdBu_r", width=800, height=700)
                    fig_vip.add_vline(x=1.0, line_dash="dash", line_color="black")
                    fig_vip.update_layout(template="simple_white", width=800, height=700, title={'text': "VIP Scores", 'x':0.5, 'xanchor': 'center'}, coloraxis_showscale=False)
                    st.plotly_chart(fig_vip, use_container_width=False)

        with tabs[3]:
            c1, c2 = st.columns([1, 2])
            with c2:
                plot_df = res_stats.copy()
                fig_vol = px.scatter(plot_df, x="Log2_FC", y="-Log10_P", color="Sig", color_discrete_map=COLOR_PALETTE, hover_data={"Metabolite":True}, width=600, height=600, render_mode='webgl')
                fig_vol.add_hline(y=-np.log10(p_th), line_dash="dash", line_color="black"); fig_vol.add_vline(x=fc_th, line_dash="dash", line_color="black"); fig_vol.add_vline(x=-fc_th, line_dash="dash", line_color="black")
                update_layout_square(fig_vol, "Volcano Plot", "Log2 Fold Change", "-Log10(P-value)")
                st.plotly_chart(fig_vol, use_container_width=False)

        with tabs[4]:
            if not sig_metabolites: st.info("无显著差异物")
            else:
                c1, c2 = st.columns([1, 6])
                with c2:
                    top_n = 50; top_feats = res_stats.sort_values('P_Value').head(top_n)['Metabolite'].tolist(); hm_data = df_sub.set_index(group_col)[top_feats].T
                    sample_groups = df_sub[group_col]; lut = {grp: GROUP_COLORS[i % len(GROUP_COLORS)] for i, grp in enumerate(sample_groups.unique())}; col_colors = sample_groups.map(lut)
                    if feature_meta is not None: hm_data.index = [feature_meta.loc[f, 'Clean_Name'] if f in feature_meta.index else f for f in hm_data.index]
                    try:
                        g = sns.clustermap(hm_data.astype(float), z_score=0, cmap="vlag", center=0, col_colors=col_colors, figsize=(12, 14), dendrogram_ratio=(.1, .1), cbar_pos=(0.35, 0.96, 0.3, 0.02), cbar_kws={'orientation': 'horizontal'})
                        g.ax_heatmap.set_ylabel(""); g.ax_heatmap.set_xlabel("")
                        g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xmajorticklabels(), rotation=90, fontsize=9)
                        st.pyplot(g.fig)
                    except: st.error("绘图错误")

        with tabs[5]:
            c1, c2 = st.columns([1.5, 1])
            with c1:
                st.subheader("统计表")
                if not res_stats.empty:
                    display_df = res_stats.sort_values("P_Value").copy()
                    if 'Clean_Name' in display_df.columns: display_df['Name'] = display_df['Clean_Name'].fillna(display_df['Metabolite'])
                    else: display_df['Name'] = display_df['Metabolite']
                    st.dataframe(display_df[[c for c in ["Name", "Log2_FC", "P_Value", "FDR", "Confidence_Level"] if c in display_df]].style.format({"Log2_FC": "{:.2f}", "P_Value": "{:.2e}", "FDR": "{:.2e}"}).background_gradient(subset=['P_Value'], cmap="Reds_r", vmin=0, vmax=0.05), use_container_width=True, height=600)
            with c2:
                st.subheader("箱线图")
                # 箱线图逻辑修复
                c_box1, c_box2 = st.columns(2)
                show_points = c_box1.checkbox("显示散点", value=True)
                box_width = c_box2.slider("箱体宽度", 0.1, 1.0, 0.5)
                
                feat_options = sorted(feats); def_ix = feat_options.index(sig_metabolites[0]) if sig_metabolites else 0; target_feat = st.selectbox("选择代谢物", feat_options, index=def_ix)
                if target_feat:
                    box_df = df_sub[[group_col, target_feat]].copy()
                    # 关键修改：points=all 或 outliers
                    points_arg = "all" if show_points else "outliers"
                    fig_box = px.box(box_df, x=group_col, y=target_feat, color=group_col, color_discrete_sequence=GROUP_COLORS, points=points_arg, width=500, height=500)
                    # 关键修改：pointpos=0 让点居中
                    fig_box.update_traces(width=box_width, marker=dict(size=6, opacity=0.7, line=dict(width=1, color='black')), jitter=0.5, pointpos=0)
                    update_layout_square(fig_box, target_feat, "Group", "Log2 Intensity", width=500, height=500)
                    st.plotly_chart(fig_box, use_container_width=False)
