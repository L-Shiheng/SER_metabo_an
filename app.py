import streamlit as st
import pandas as pd
import numpy as np
import os
import gc
import datetime
import re  # <--- 修复：顶层导入re
import traceback
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

# ... (绘图函数保持不变) ...
def update_layout_square(fig, title="", x_title="", y_title="", width=600, height=600):
    fig.update_layout(template="simple_white", width=width, height=height, title={'text': title, 'y':0.95, 'x':0.5, 'xanchor': 'center'}, xaxis=dict(title=x_title, showline=True, linewidth=2, mirror=True), yaxis=dict(title=y_title, showline=True, linewidth=2, mirror=True), legend=dict(yanchor="top", y=1, xanchor="left", x=1.15), margin=dict(l=80, r=180, t=80, b=80))
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
    t = model.x_scores_; w = model.x_weights_; q = model.y_loadings_; p, h = w.shape; vips = np.zeros((p,))
    s = np.diag(t.T @ t @ q.T @ q); total_s = np.sum(s)
    for i in range(p):
        weight = np.array([(w[i, j] / np.linalg.norm(w[:, j]))**2 for j in range(h)])
        vips[i] = np.sqrt(p * (s @ weight) / total_s)
    return vips
@st.cache_data
def run_pairwise_statistics(df, group_col, case, control, features, equal_var=False):
    g1 = df[df[group_col] == case]; g2 = df[df[group_col] == control]; res = []
    for f in features:
        v1, v2 = g1[f].values, g2[f].values; fc = np.mean(v1) - np.mean(v2)
        try: t, p = stats.ttest_ind(v1, v2, equal_var=equal_var)
        except: p = 1.0
        if np.isnan(p): p = 1.0
        res.append({'Metabolite': f, 'Log2_FC': fc, 'P_Value': p})
    res_df = pd.DataFrame(res).dropna()
    if not res_df.empty: _, p_corr, _, _ = multipletests(res_df['P_Value'], method='fdr_bh'); res_df['FDR'] = p_corr; res_df['-Log10_P'] = -np.log10(res_df['P_Value'])
    else: res_df['FDR'] = 1.0; res_df['-Log10_P'] = 0
    return res_df

# Session State
if 'raw_df' not in st.session_state: st.session_state.raw_df = None
if 'feature_meta' not in st.session_state: st.session_state.feature_meta = None
if 'data_loaded' not in st.session_state: st.session_state.data_loaded = False
if 'qc_report' not in st.session_state: st.session_state.qc_report = {}
if 'all_sample_ids' not in st.session_state: st.session_state.all_sample_ids = []

# ==========================================
# 3. 侧边栏 (UI 升级)
# ==========================================
with st.sidebar:
    st.header("🛠️ 数据控制台")
    
    # 1. Info (必须)
    st.markdown("#### 1. 上传 Sample Info (必选)")
    sample_info_file = st.file_uploader("Sample Info (.csv/.xlsx)", type=["csv", "xlsx"], key="info")
    info_df = None
    candidate_samples = []
    
    user_sample_col = None
    user_group_col = None
    
    if sample_info_file:
        try:
            if sample_info_file.name.endswith('.csv'): info_df = pd.read_csv(sample_info_file)
            else: info_df = pd.read_excel(sample_info_file)
            
            # --- 核心修复：基础列名映射 ---
            cols = list(info_df.columns)
            cols_lower = [c.lower() for c in cols]
            
            idx_sample = 0
            for kw in ['sample.name', 'sample_name', 'sample', 'name', 'id']:
                if kw in cols_lower: idx_sample = cols_lower.index(kw); break
            
            # 默认选第2列作为 Group
            idx_group = 1 if len(cols) > 1 else 0
            for kw in ['group', 'class', 'type', 'condition']:
                if kw in cols_lower: idx_group = cols_lower.index(kw); break
            
            # 让用户确认 (解决 2 列文件识别问题)
            c_base1, c_base2 = st.columns(2)
            user_sample_col = c_base1.selectbox("样本名列", cols, index=idx_sample)
            user_group_col = c_base2.selectbox("分组列", cols, index=idx_group)

            if user_sample_col:
                candidate_samples = info_df[user_sample_col].astype(str).unique().tolist()
                
            st.caption(f"✅ 已加载 {len(info_df)} 行")
            
        except Exception as e: st.error(f"Info 读取失败: {e}")

    if not candidate_samples and st.session_state.all_sample_ids:
        candidate_samples = st.session_state.all_sample_ids

    # 2. 剔除
    st.markdown("#### 2. 样本管理 (剔除异常点)")
    excluded_samples = st.multiselect("选择要剔除的样本:", options=candidate_samples, default=[])
    if excluded_samples: st.warning(f"⚠️ 将剔除 {len(excluded_samples)} 个样本")

    # 3. Scope
    st.markdown("#### 3. 数据处理范围")
    feature_scope = st.radio("加载特征范围:", ["仅已注释特征 (推荐)", "全部特征"], index=0)

    # 4. SERRF (可选)
    st.markdown("#### 4. SERRF 批次校正")
    use_serrf = st.checkbox("启用 SERRF 校正", value=False)
    serrf_ready = False
    
    if use_serrf:
        if info_df is not None:
            cols = list(info_df.columns); cols_lower = [c.lower() for c in cols]
            idx_order = next((i for i, c in enumerate(cols_lower) if any(x in c for x in ['order', 'run', 'idx', 'seq'])), 0)
            type_cands = [i for i, c in enumerate(cols_lower) if any(x in c for x in ['class', 'type', 'group'])]
            # 优先用 Group 列
            final_type_idx = type_cands[0] if type_cands else 0
            if user_group_col in cols: final_type_idx = cols.index(user_group_col)
            
            default_qc_label = "QC"
            try:
                vals = info_df.iloc[:, final_type_idx].unique().astype(str)
                default_qc_label = next((v for v in vals if 'qc' in v.lower()), "QC")
            except: pass

            c1, c2, c3 = st.columns(3)
            run_order_col = c1.selectbox("Order列", cols, index=idx_order)
            sample_type_col = c2.selectbox("Type列", cols, index=final_type_idx)
            qc_label = c3.text_input("QC标签", value=default_qc_label)
            serrf_ready = True
        else:
            st.warning("⚠️ SERRF 必须上传 Info 表")

    # 5. Upload
    st.markdown("#### 5. 上传 MetDNA 数据")
    uploaded_files = st.file_uploader("MetDNA文件 (支持多选)", type=["csv", "xlsx"], accept_multiple_files=True, key="data")
    st.markdown("---")
    
    # 6. Button
    process_container = st.container()
    process_container.markdown('<div class="process-btn">', unsafe_allow_html=True)
    start_process = process_container.button("📥 开始处理数据 (Load & Process)")
    process_container.markdown('</div>', unsafe_allow_html=True)

    # 7. Logic
    if start_process:
        st.session_state.qc_report = {}
        if not uploaded_files:
            st.error("请先上传数据文件！")
        else:
            progress_bar = st.progress(0); status_text = st.empty()
            with st.spinner("正在启动高性能处理引擎..."):
                parsed_results = []
                current_run_samples = set()
                
                for i, file in enumerate(uploaded_files):
                    status_text.text(f"正在处理 ({i+1}/{len(uploaded_files)}): {file.name} ...")
                    try:
                        file.seek(0)
                        file_type = 'csv' if file.name.endswith('.csv') else 'excel'
                        unique_name = f"{os.path.splitext(file.name)[0]}_{i+1}{os.path.splitext(file.name)[1]}"
                        
                        df_t, meta, err = parse_metdna_file(file, unique_name, file_type=file_type)
                        if err: st.warning(f"{file.name}: {err}"); continue
                        
                        # === 剔除逻辑 ===
                        if excluded_samples:
                            n_before = len(df_t)
                            def normalize_str(s): return re.sub(r'[^a-z0-9]', '', str(s).strip().lower())
                            ex_norm = set([normalize_str(s) for s in excluded_samples])
                            df_t['id_clean'] = df_t['SampleID'].astype(str).apply(normalize_str)
                            mask_remove = df_t['id_clean'].isin(ex_norm)
                            df_t = df_t[~mask_remove].drop(columns=['id_clean'])
                            n_after = len(df_t)
                            if n_before > n_after: st.success(f"✅ {unique_name}: 已剔除 {n_before - n_after} 个样本")
                        
                        current_run_samples.update(df_t['SampleID'].astype(str).tolist())

                        # Filter Scope
                        if feature_scope.startswith("仅已注释"):
                            annotated_ids = meta[meta['Is_Annotated'] == True].index
                            cols_to_keep = ['SampleID', 'Group', 'Source_Files'] + [c for c in df_t.columns if c in annotated_ids]
                            cols_to_keep = [c for c in cols_to_keep if c in df_t.columns] 
                            df_t = df_t[cols_to_keep]
                            meta = meta.loc[meta.index.isin(df_t.columns)]
                            
                        # === 关键：应用分组 (无论是否SERRF) ===
                        info_aligned = None
                        if info_df is not None and user_sample_col:
                            # 1. 强制对齐
                            info_aligned = align_sample_info(df_t, info_df, sample_col_name=user_sample_col)
                            
                            # 2. 强制覆盖 Group (如果匹配到了)
                            if user_group_col and user_group_col in info_aligned.columns:
                                group_vals = info_aligned[user_group_col].fillna(df_t['Group']).values
                                df_t['Group'] = group_vals
                                
                                # 反馈匹配情况
                                n_matched = (~info_aligned[user_group_col].isna()).sum()
                                st.caption(f"✅ Info匹配: {n_matched}/{len(df_t)} 样本")
                        
                        # SERRF
                        if use_serrf and serrf_ready and info_aligned is not None:
                            n_matched = info_aligned[run_order_col].notna().sum()
                            if n_matched == 0:
                                st.error(f"❌ {file.name}: SERRF 匹配失败 (Order列)"); st.session_state.qc_report[unique_name] = {"Status": "Failed (No Match)"}
                            else:
                                if run_order_col in info_aligned.columns and sample_type_col in info_aligned.columns:
                                    num_cols = df_t.select_dtypes(include=[np.number]).columns.tolist()
                                    df_numeric = df_t[num_cols]
                                    corrected_data, serrf_stats = serrf_normalization(df_numeric, info_aligned, run_order_col, sample_type_col, qc_label)
                                    if corrected_data is not None:
                                        if serrf_stats['RSD_After'] > serrf_stats['RSD_Before']:
                                            st.session_state.qc_report[unique_name] = {"Status": "Skipped (Worse)", "RSD_Before": serrf_stats['RSD_Before'], "RSD_After": serrf_stats['RSD_After']}
                                        else:
                                            for c in corrected_data.columns: df_t[c] = corrected_data[c].values
                                            st.session_state.qc_report[unique_name] = {"Status": "Success", "RSD_Before": serrf_stats['RSD_Before'], "RSD_After": serrf_stats['RSD_After']}
                                    else: st.error(f"❌ {file.name}: SERRF失败")
                                else: st.warning(f"{file.name}: 缺少列")

                        parsed_results.append((df_t, meta, unique_name))
                        del df_t, meta, info_aligned; gc.collect()

                    except Exception as e:
                        st.error(f"处理 {file.name} 失败: {str(e)}")
                        st.text(traceback.format_exc())
                    progress_bar.progress((i + 1) / len(uploaded_files))

                if parsed_results:
                    if current_run_samples:
                        combined = set(st.session_state.all_sample_ids) | current_run_samples
                        st.session_state.all_sample_ids = sorted(list(combined))

                    if len(parsed_results) == 1:
                        st.session_state.raw_df = parsed_results[0][0]
                        st.session_state.feature_meta = parsed_results[0][1]
                    else:
                        m_df, m_meta, m_err = merge_multiple_dfs(parsed_results)
                        if m_err: st.error(m_err)
                        else: st.session_state.raw_df = m_df; st.session_state.feature_meta = m_meta
                    
                    st.session_state.data_loaded = True
                    st.success("✅ 处理完成！")
                    st.rerun() 
                else: st.error("加载失败")

    # Export & Analysis
    if st.session_state.data_loaded and st.session_state.raw_df is not None:
        raw_df = st.session_state.raw_df
        st.info(f"数据: {len(raw_df)} 样本 x {len(raw_df.columns)-3} 特征")
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        csv_data = raw_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 导出数据", csv_data, f"Metabo_{ts}.csv", "text/csv")
        st.divider()

        with st.form(key='analysis_form'):
            st.markdown("### ⚙️ 统计分析")
            non_num = raw_df.select_dtypes(exclude=[np.number]).columns.tolist()
            def_grp = non_num.index('Group') if 'Group' in non_num else 0
            group_col = st.selectbox("分组列", non_num, index=def_grp)
            
            filter_option = st.radio("分析范围:", ["全部特征", "仅已注释特征"], index=0)
            
            with st.expander("数据清洗", expanded=False):
                miss_th = st.slider("剔除缺失 > X", 0.0, 1.0, 0.5)
                impute_m_disp = st.selectbox("填充", ["min (推荐)", "KNN", "mean", "zero"], index=0)
                impute_m = "KNN" if "KNN" in impute_m_disp else ("mean" if "mean" in impute_m_disp else ("zero" if "zero" in impute_m_disp else "min"))
                norm_m = st.selectbox("归一化", ["None", "PQN", "Sum", "Median"], index=1)
                do_log = st.checkbox("Log2", value=True); scale_m = st.selectbox("缩放", ["None", "Auto", "Pareto"], index=2)

            cur_grps = sorted(raw_df[group_col].astype(str).unique())
            sel_grps = st.multiselect("纳入组", cur_grps, default=cur_grps[:2] if len(cur_grps)>=2 else cur_grps)
            c1, c2 = st.columns(2)
            valid = list(sel_grps)
            case = c1.selectbox("Case", valid, index=0 if valid else None)
            ctrl = c2.selectbox("Control", valid, index=1 if len(valid)>1 else 0)
            c3, c4 = st.columns(2)
            p_th = c3.number_input("P-value", 0.05); fc_th = c4.number_input("Log2 FC", 1.0)
            equal_var = st.checkbox("Equal Var", value=True); jitter = st.checkbox("Jitter", value=True)
            submit_button = st.form_submit_button(label='🚀 运行')

# Main
if not st.session_state.data_loaded:
    st.title("🧬 MetaboAnalyst Pro"); st.info("👈 请在左侧上传数据"); st.stop()

if not submit_button:
    st.title("✅ 就绪"); st.dataframe(st.session_state.raw_df.head(50)); st.stop()

if submit_button:
    if len(sel_grps)<2: st.error("请选2组"); st.stop()
    with st.spinner("计算中..."):
        raw_df = st.session_state.raw_df; meta = st.session_state.feature_meta
        df_proc, feats = data_cleaning_pipeline(raw_df, group_col, miss_th, impute_m, norm_m, do_log, scale_m)
        if filter_option == "仅已注释特征":
            if meta is not None:
                anno_ids = meta[meta['Is_Annotated']==True].index.tolist()
                feats = [f for f in feats if f in anno_ids]
                if not feats: st.error("无特征"); st.stop()
            else: st.warning("无Meta")
        
        df_sub = df_proc[df_proc[group_col].isin(sel_grps)].copy()
        
        if case != ctrl:
            stats_df = run_pairwise_statistics(df_sub, group_col, case, ctrl, feats, equal_var)
            if meta is not None: stats_df = stats_df.merge(meta[['Confidence_Level', 'Clean_Name']], left_on='Metabolite', right_index=True, how='left')
            stats_df['Sig'] = 'NS'
            stats_df.loc[(stats_df['P_Value']<p_th)&(stats_df['Log2_FC']>fc_th), 'Sig']='Up'
            stats_df.loc[(stats_df['P_Value']<p_th)&(stats_df['Log2_FC']<-fc_th), 'Sig']='Down'
            sig_mets = stats_df[stats_df['Sig']!='NS']['Metabolite'].tolist()
        else: stats_df = pd.DataFrame(); sig_mets = []

        st.title("📊 分析报告"); st.caption(f"{case} vs {ctrl} | N={len(feats)}")
        
        # Tabs
        tabs = st.tabs(["📊 PCA", "🎯 PLS-DA", "⭐ VIP", "🌋 Volcano", "🔥 Heatmap", "📑 Data"])
        with tabs[0]:
            c1, c2 = st.columns([1, 2])
            with c2:
                if len(df_sub)<3: st.warning("样本不足")
                else:
                    X = StandardScaler().fit_transform(df_sub[feats])
                    pca = PCA(n_components=2).fit(X); pcs = pca.transform(X); var = pca.explained_variance_ratio_
                    hover_cols = ["SampleID"]; 
                    if "Source_Files" in df_sub.columns: hover_cols.append("Source_Files")
                    else: df_sub["Source_Files"] = "Unknown"; hover_cols.append("Source_Files")
                    fig_pca = px.scatter(df_sub, x=pcs[:,0], y=pcs[:,1], color=group_col, symbol=group_col, color_discrete_sequence=GROUP_COLORS, width=600, height=600, render_mode='webgl', hover_data=hover_cols)
                    fig_pca.update_traces(marker=dict(size=14, line=dict(width=1, color='black'), opacity=0.9))
                    update_layout_square(fig_pca, "PCA", f"PC1 ({var[0]:.1%})", f"PC2 ({var[1]:.1%})")
                    st.plotly_chart(fig_pca, use_container_width=False)
        
        with tabs[1]:
            c1, c2 = st.columns([1, 2])
            with c2:
                if len(df_sub)<3: st.warning("样本不足")
                else:
                    X_pls = df_sub[feats].values; y_lbl = pd.factorize(df_sub[group_col])[0]
                    pls = PLSRegression(n_components=2).fit(X_pls, y_lbl)
                    plot_df = pd.DataFrame({'C1': pls.x_scores_[:,0], 'C2': pls.x_scores_[:,1], 'Group': df_sub[group_col].values})
                    fig_pls = px.scatter(plot_df, x='C1', y='C2', color='Group', symbol='Group', color_discrete_sequence=GROUP_COLORS, width=600, height=600, render_mode='webgl')
                    for i, g in enumerate(list(sel_grps)):
                        sub = plot_df[plot_df['Group']==g]
                        if len(sub)>=3:
                            el_x, el_y = get_ellipse_coordinates(sub['C1'], sub['C2'])
                            if el_x is not None: fig_pls.add_trace(go.Scatter(x=el_x, y=el_y, mode='lines', line=dict(color=GROUP_COLORS[i%len(GROUP_COLORS)], width=2, dash='dash'), showlegend=False, hoverinfo='skip'))
                    fig_pls.update_traces(marker=dict(size=14, line=dict(width=1.5, color='black'), opacity=1.0))
                    update_layout_square(fig_pls, "PLS-DA", "C1", "C2")
                    st.plotly_chart(fig_pls, use_container_width=False)
        
        with tabs[2]:
            if 'pls' in locals():
                vips = calculate_vips(pls); vip_df = pd.DataFrame({'Metabolite': feats, 'VIP': vips})
                if meta is not None: vip_df = vip_df.merge(meta[['Clean_Name']], left_on='Metabolite', right_index=True, how='left'); vip_df['Display_Name'] = vip_df['Clean_Name'].fillna(vip_df['Metabolite'])
                else: vip_df['Display_Name'] = vip_df['Metabolite']
                top = vip_df.sort_values('VIP', ascending=True).tail(25)
                fig_vip = px.bar(top, x="VIP", y="Display_Name", orientation='h', color="VIP", color_continuous_scale="RdBu_r")
                fig_vip.add_vline(x=1.0, line_dash="dash"); fig_vip.update_layout(template="simple_white", height=700, coloraxis_showscale=False)
                st.plotly_chart(fig_vip, use_container_width=False)

        with tabs[3]:
            fig_vol = px.scatter(stats_df, x="Log2_FC", y="-Log10_P", color="Sig", color_discrete_map=COLOR_PALETTE, hover_data={"Metabolite":True}, width=600, height=600, render_mode='webgl')
            fig_vol.add_hline(y=-np.log10(p_th), line_dash="dash"); fig_vol.add_vline(x=fc_th, line_dash="dash"); fig_vol.add_vline(x=-fc_th, line_dash="dash")
            update_layout_square(fig_vol, "Volcano", "Log2 FC", "-Log10 P")
            st.plotly_chart(fig_vol, use_container_width=False)

        with tabs[4]:
            if not sig_mets: st.info("无差异")
            else:
                top_n = stats_df.sort_values('P_Value').head(50)['Metabolite'].tolist(); hm = df_sub.set_index(group_col)[top_n].T
                lut = {g: GROUP_COLORS[i%len(GROUP_COLORS)] for i, g in enumerate(df_sub[group_col].unique())}; cols = df_sub[group_col].map(lut)
                if meta is not None: hm.index = [meta.loc[f, 'Clean_Name'] if f in meta.index else f for f in hm.index]
                try:
                    g = sns.clustermap(hm.astype(float), z_score=0, cmap="vlag", center=0, col_colors=cols, figsize=(10, 12))
                    g.ax_heatmap.set_xlabel(""); g.ax_heatmap.set_ylabel("")
                    st.pyplot(g.fig)
                except: st.error("绘图错误")

        with tabs[5]:
            c1, c2 = st.columns([1.5, 1])
            with c1:
                st.dataframe(stats_df.sort_values("P_Value").style.format({"Log2_FC":"{:.2f}", "P_Value":"{:.2e}"}).background_gradient(subset=['P_Value'], cmap="Reds_r", vmax=0.05), height=600)
            with c2:
                show_pts = st.checkbox("散点", True); bw = st.slider("宽", 0.1, 1.0, 0.5)
                opts = sorted(feats); dx = opts.index(sig_mets[0]) if sig_mets else 0
                tgt = st.selectbox("Metabolite", opts, index=dx)
                if tgt:
                    bdf = df_sub[[group_col, tgt]].copy()
                    pt_arg = "all" if show_pts else "outliers"
                    fb = px.box(bdf, x=group_col, y=tgt, color=group_col, color_discrete_sequence=GROUP_COLORS, points=pt_arg)
                    fb.update_traces(width=bw, marker=dict(size=6, opacity=0.7, line=dict(width=1, color='black')), jitter=0.5, pointpos=0)
                    update_layout_square(fb, tgt, "Group", "Log2 Int", 500, 500)
                    st.plotly_chart(fb, use_container_width=False)
