import streamlit as st
import pandas as pd
import numpy as np
import os
import gc  # 内存回收
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
# 配置与初始化
# ==========================================
st.set_page_config(page_title="MetaboAnalyst Pro", page_icon="🧬", layout="wide")

# 配置常量
COLOR_PALETTE = {'Up': '#CD0000', 'Down': '#00008B', 'NS': '#E0E0E0'} 
GROUP_COLORS = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F', '#8491B4', '#91D1C2', '#DC0000', '#7E6148', '#B09C85']

st.markdown("""
<style>
    .block-container {padding-top: 2rem !important; padding-bottom: 3rem !important;}
    h1, h2, h3, div, p {font-family: 'Arial', sans-serif; color: #2c3e50;}
    /* Tab 样式 */
    button[data-baseweb="tab"] {
        font-size: 16px; font-weight: bold; padding: 10px 15px;
        background-color: white; border-radius: 5px 5px 0 0;
    }
    /* 按钮颜色区分 */
    div[data-testid="stForm"] button {
        width: 100%; background-color: #ff4b4b; color: white; font-weight: bold; border: none; padding: 0.5rem;
    }
    .process-btn button {
        width: 100%; background-color: #4CAF50 !important; color: white !important; font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# 导入模块
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
# 辅助函数
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
# Session State
# ==========================================
if 'raw_df' not in st.session_state: st.session_state.raw_df = None
if 'feature_meta' not in st.session_state: st.session_state.feature_meta = None
if 'data_loaded' not in st.session_state: st.session_state.data_loaded = False
if 'qc_report' not in st.session_state: st.session_state.qc_report = {}

# ==========================================
# 侧边栏：配置与上传
# ==========================================
with st.sidebar:
    st.header("🛠️ 数据控制台")
    
    # --- Sample Info ---
    st.markdown("#### 1. 上传 Sample Info (必选 for SERRF)")
    sample_info_file = st.file_uploader("Sample Info (.csv/.xlsx)", type=["csv", "xlsx"], key="info")
    info_df = None
    if sample_info_file:
        try:
            if sample_info_file.name.endswith('.csv'): info_df = pd.read_csv(sample_info_file)
            else: info_df = pd.read_excel(sample_info_file)
            st.caption(f"✅ Info 表已就绪 ({len(info_df)} 行)")
        except: pass

    # --- SERRF Settings ---
    st.markdown("#### 2. SERRF 设置")
    use_serrf = st.checkbox("启用 SERRF 校正", value=False)
    serrf_ready = False
    
    if use_serrf:
        if info_df is not None:
            # 作用域选择 (关键提速点)
            serrf_scope = st.radio("校正范围:", ["仅已注释特征 (推荐)", "全部特征"], index=0, 
                                   help="仅校正有名字的特征，未注释的特征将被丢弃。")
            
            c1, c2, c3 = st.columns(3)
            cols = list(info_df.columns)
            idx_order = next((i for i, c in enumerate(cols) if 'order' in c.lower()), 0)
            idx_class = next((i for i, c in enumerate(cols) if 'class' in c.lower() or 'type' in c.lower()), 0)
            
            run_order_col = c1.selectbox("Order列", cols, index=idx_order)
            sample_type_col = c2.selectbox("Type列", cols, index=idx_class)
            qc_label = c3.text_input("QC标签", value="QC")
            serrf_ready = True
        else:
            st.warning("⚠️ 请先上传 Sample Info")

    # --- Data Upload ---
    st.markdown("#### 3. 上传 MetDNA 数据")
    uploaded_files = st.file_uploader("MetDNA文件 (支持多选)", type=["csv", "xlsx"], accept_multiple_files=True, key="data")
    
    st.markdown("---")
    
    # --- Process Button ---
    process_container = st.container()
    process_container.markdown('<div class="process-btn">', unsafe_allow_html=True)
    start_process = process_container.button("📥 开始处理数据 (Load & Process)")
    process_container.markdown('</div>', unsafe_allow_html=True)
    
    # --- Processing Logic ---
    if start_process:
        st.session_state.qc_report = {}
        if not uploaded_files:
            st.error("请先上传数据文件！")
        else:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            with st.spinner("正在执行多文件解析与 SERRF 校正..."):
                parsed_results = []
                
                for i, file in enumerate(uploaded_files):
                    status_text.text(f"正在处理: {file.name} ...")
                    try:
                        file.seek(0)
                        file_type = 'csv' if file.name.endswith('.csv') else 'excel'
                        unique_name = f"{os.path.splitext(file.name)[0]}_{i+1}{os.path.splitext(file.name)[1]}"
                        
                        df_t, meta, err = parse_metdna_file(file, unique_name, file_type=file_type)
                        if err: 
                            st.warning(f"{file.name}: {err}"); continue
                        
                        # A. 优化: SERRF 前置过滤 (极速模式)
                        if use_serrf and serrf_ready and serrf_scope.startswith("仅已注释"):
                            annotated_ids = meta[meta['Is_Annotated'] == True].index
                            cols_to_keep = ['SampleID', 'Group'] + [c for c in df_t.columns if c in annotated_ids]
                            df_t = df_t[cols_to_keep]
                            meta = meta.loc[meta.index.isin(df_t.columns)]
                            
                        # B. 对齐 Info
                        info_aligned = None
                        if info_df is not None:
                            info_aligned = align_sample_info(df_t, info_df)
                            g_col = next((c for c in info_aligned.columns if c.lower() in ['group', 'class']), None)
                            if g_col: df_t['Group'] = info_aligned[g_col].fillna(df_t['Group']).values
                        
                        # C. 执行 SERRF
                        if use_serrf and serrf_ready and info_aligned is not None:
                            n_matched = info_aligned[run_order_col].notna().sum()
                            if n_matched == 0:
                                st.error(f"❌ {file.name}: 样本名匹配失败！")
                                st.session_state.qc_report[unique_name] = {"Status": "Failed (No Match)"}
                            else:
                                if run_order_col in info_aligned.columns and sample_type_col in info_aligned.columns:
                                    num_cols = df_t.select_dtypes(include=[np.number]).columns.tolist()
                                    df_numeric = df_t[num_cols]
                                    
                                    # 校正
                                    corrected_data, serrf_stats = serrf_normalization(
                                        df_numeric, info_aligned, run_order_col, sample_type_col, qc_label
                                    )
                                    
                                    if corrected_data is not None:
                                        # 智能回滚逻辑
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
                                        st.error(f"❌ {file.name}: SERRF 计算失败")
                                else:
                                    st.warning(f"{file.name}: 缺少列")

                        parsed_results.append((df_t, meta, unique_name))
                        
                        # 内存回收
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
                    st.success("✅ 数据加载完成！")
                    st.rerun() 
                else:
                    st.error("没有加载任何文件")

    # --- Data Export ---
    if st.session_state.data_loaded and st.session_state.raw_df is not None:
        raw_df = st.session_state.raw_df
        st.info(f"数据: {len(raw_df)} 样本 x {len(raw_df.columns)-2} 特征")
        
        # 带时间戳的文件名
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        csv_data = raw_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 导出合并数据", csv_data, f"Metabo_Processed_{ts}.csv", "text/csv")
        st.divider()

        # --- Analysis Form ---
        with st.form(key='analysis_form'):
            st.markdown("### ⚙️ 统计分析参数")
            non_num = raw_df.select_dtypes(exclude=[np.number]).columns.tolist()
            default_grp_idx = non_num.index('Group') if 'Group' in non_num else 0
            group_col = st.selectbox("分组列", non_num, index=default_grp_idx)
            
            filter_option = st.radio("统计范围:", ["全部特征", "仅已注释特征"], index=0)
            
            with st.expander("数据清洗与归一化 (高级)", expanded=False):
                miss_th = st.slider("剔除缺失率 > X", 0.0, 1.0, 0.5, 0.1)
                impute_m = st.selectbox("填充方法", ["min", "mean", "zero"], index=0)
                # 新增 PQN 选项
                norm_m = st.selectbox("样本归一化", ["None", "PQN", "Sum", "Median"], index=1, help="PQN (Probabilistic Quotient Normalization) 推荐用于尿液/血液样本")
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

    # --- 免责声明 ---
    st.markdown("---")
    with st.expander("⚠️ 免责声明与版权"):
        st.caption("1. 本工具仅供科研参考 (RUO)。\n2. 数据仅在内存临时处理，不保存。\n3. 基于 MIT 协议开源。")

# ==========================================
# 主面板
# ==========================================
if not st.session_state.data_loaded:
    st.title("🧬 MetaboAnalyst Pro")
    st.info("👈 请在左侧上传数据并点击 **“开始处理数据”** 按钮。")
    st.markdown("### 快速指南\n1. 上传 Sample Info (含 Order, Type)。\n2. 上传 MetDNA 结果文件。\n3. 勾选 SERRF 并点击处理。\n4. 数据加载后设置统计参数。")
    st.stop()

if not submit_button:
    st.title("✅ 数据准备就绪")
    
    # SERRF 报告
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
                    st.metric("QC RSD (已回滚)", f"{report['RSD_Before']:.1f}%", f"校正后变差 (+{delta:.1f}%)", delta_color="off")
                else:
                    st.error(f"📄 {fname}: {report['Status']}")
    st.markdown("---")
    st.markdown("👈 请在左侧配置参数并点击 **“运行统计分析”**。")
    st.subheader("原始数据预览")
    st.dataframe(st.session_state.raw_df.head(50))
    st.stop()

if submit_button:
    if len(selected_groups) < 2: st.error("请至少选择 2 个组！"); st.stop()
    
    with st.spinner("正在进行统计分析与绘图 (WebGL加速中)..."):
        raw_df = st.session_state.raw_df
        feature_meta = st.session_state.feature_meta
        
        # 清洗
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

        # 统计
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

        # 可视化
        st.title("📊 代谢组学分析报告")
        st.caption(f"对比: {case_grp} vs {ctrl_grp} | 特征数: {len(feats)} | Scaling: {scale_m} | Norm: {norm_m}")
        
        # QC 检查
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
                    pca = PCA(n_components=2).fit(X)
                    pcs = pca.transform(X)
                    var = pca.explained_variance_ratio_
                    # WebGL 加速
                    fig_pca = px.scatter(x=pcs[:,0], y=pcs[:,1], color=df_sub[group_col], symbol=df_sub[group_col],
                                         color_discrete_sequence=GROUP_COLORS, width=600, height=600, render_mode='webgl')
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
                    fig_pls = px.scatter(plot_df, x='C1', y='C2', color='Group', symbol='Group',
                                         color_discrete_sequence=GROUP_COLORS, width=600, height=600, render_mode='webgl')
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
                # WebGL 加速火山图
                fig_vol = px.scatter(plot_df, x="Log2_FC", y="-Log10_P", color="Sig", color_discrete_map=COLOR_PALETTE,
                                     hover_data={"Metabolite":True}, width=600, height=600, render_mode='webgl')
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
                if not res_stats.empty:
                    display_df = res_stats.sort_values("P_Value").copy()
                    if 'Clean_Name' in display_df.columns: display_df['Name'] = display_df['Clean_Name'].fillna(display_df['Metabolite'])
                    else: display_df['Name'] = display_df['Metabolite']
                    st.dataframe(display_df.style.format({"Log2_FC": "{:.2f}", "P_Value": "{:.2e}", "FDR": "{:.2e}"}).background_gradient(subset=['P_Value'], cmap="Reds_r", vmin=0, vmax=0.05), use_container_width=True, height=600)
            with c2:
                feat_options = sorted(feats); def_ix = feat_options.index(sig_metabolites[0]) if sig_metabolites else 0; target_feat = st.selectbox("选择代谢物", feat_options, index=def_ix)
                if target_feat:
                    box_df = df_sub[[group_col, target_feat]].copy()
                    fig_box = px.box(box_df, x=group_col, y=target_feat, color=group_col, color_discrete_sequence=GROUP_COLORS, points="all", width=500, height=500)
                    update_layout_square(fig_box, target_feat, "Group", "Log2 Intensity", width=500, height=500)
                    st.plotly_chart(fig_box, use_container_width=False)
