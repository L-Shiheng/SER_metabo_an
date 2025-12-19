import streamlit as st
import pandas as pd
import numpy as np
import os
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
# 移除了 mpatches，不再需要手动画图例
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

# ==========================================
# 0. 导入数据清洗模块
# ==========================================
try:
    from data_preprocessing import data_cleaning_pipeline, parse_metdna_file, merge_multiple_dfs, apply_sample_info
except ImportError:
    st.error("❌ 严重错误：未找到 'data_preprocessing.py'。")
    st.stop()

# ==========================================
# 1. 全局配置与样式
# ==========================================
st.set_page_config(page_title="MetaboAnalyst Pro", page_icon="🧬", layout="wide")

st.markdown("""
<style>
    .block-container {padding-top: 2rem !important; padding-bottom: 3rem !important;}
    h1, h2, h3, div, p {font-family: 'Arial', sans-serif; color: #2c3e50;}
    button[data-baseweb="tab"] {
        font-size: 16px; font-weight: bold; padding: 10px 15px;
        background-color: white; border-radius: 5px 5px 0 0;
    }
    div[data-testid="stForm"] button {
        width: 100%; background-color: #ff4b4b; color: white; font-weight: bold; border: none; padding: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

COLOR_PALETTE = {'Up': '#CD0000', 'Down': '#00008B', 'NS': '#E0E0E0'} 
GROUP_COLORS = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F', '#8491B4', '#91D1C2', '#DC0000', '#7E6148', '#B09C85']

# --- 通用绘图布局 ---
def update_layout_square(fig, title="", x_title="", y_title="", width=600, height=600):
    fig.update_layout(
        template="simple_white", width=width, height=height,
        title={'text': title, 'y':0.95, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top', 'font': dict(size=20, color='black', family="Arial, bold")},
        xaxis=dict(title=x_title, showline=True, linewidth=2, linecolor='black', mirror=True, title_font=dict(size=16, family="Arial, bold")),
        yaxis=dict(title=y_title, showline=True, linewidth=2, linecolor='black', mirror=True, title_font=dict(size=16, family="Arial, bold"), automargin=True),
        legend=dict(yanchor="top", y=1, xanchor="left", x=1.15, bordercolor="Black", borderwidth=0, font=dict(size=12)),
        margin=dict(l=80, r=180, t=80, b=80)
    )
    return fig

# PLS-DA 椭圆
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
    s = np.diag(t.T @ t @ q.T @ q).reshape(h, -1)
    total_s = np.sum(s)
    for i in range(p):
        weight = np.array([(w[i, j] / np.linalg.norm(w[:, j]))**2 for j in range(h)])
        vips[i] = np.sqrt(p * (s.T @ weight) / total_s)
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
    else:
        res_df['FDR'] = 1.0; res_df['-Log10_P'] = 0
    return res_df

# ==========================================
# 2. 侧边栏与数据加载
# ==========================================
with st.sidebar:
    st.header("🛠️ 分析控制台")
    uploaded_files = st.file_uploader("1. 上传 MetDNA 数据 (支持多文件)", type=["csv", "xlsx"], accept_multiple_files=True)
    feature_meta = None; raw_df = None

    if not uploaded_files:
        st.info("👋 请先上传数据文件"); st.stop()
        
    parsed_results = []
    for i, file in enumerate(uploaded_files):
        try:
            file.seek(0)
            file_type = 'csv' if file.name.endswith('.csv') else 'excel'
            unique_name = f"{os.path.splitext(file.name)[0]}_{i+1}{os.path.splitext(file.name)[1]}"
            df_t, meta, err = parse_metdna_file(file, unique_name, file_type=file_type)
            if err: st.warning(f"文件 {file.name} 解析失败: {err}"); continue
            parsed_results.append((df_t, meta, unique_name))
        except Exception as e: st.error(f"出错: {e}")
    
    if not parsed_results: st.stop()
        
    if len(parsed_results) == 1:
        raw_df = parsed_results[0][0]; feature_meta = parsed_results[0][1]
        st.success(f"✅ 加载成功")
    else:
        merged_df, merged_meta, m_err = merge_multiple_dfs(parsed_results)
        if m_err: st.error(m_err); st.stop()
        raw_df = merged_df; feature_meta = merged_meta
        st.success(f"✅ 合并成功")

    sample_info_file = st.file_uploader("2. 上传样本信息表 (可选)", type=["csv", "xlsx"])
    if sample_info_file:
        raw_df, msg = apply_sample_info(raw_df, sample_info_file)
        if "成功" in msg: st.success(msg)
        else: st.warning(msg)

    if raw_df is not None:
        csv_data = raw_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 下载合并数据 (CSV)", csv_data, "merged_data.csv", "text/csv")

    st.divider()
    
    # --- 参数表单 ---
    with st.form(key='analysis_form'):
        st.markdown("### ⚙️ 参数设置")
        
        non_num = raw_df.select_dtypes(exclude=[np.number]).columns.tolist()
        default_grp_idx = non_num.index('Group') if 'Group' in non_num else 0
        group_col = st.selectbox("3. 分组列", non_num, index=default_grp_idx)
        
        filter_option = st.radio("4. 特征过滤:", ["全部特征", "仅已注释特征"], index=0)
        
        with st.expander("数据清洗 (影响 PCA/PLS-DA)", expanded=False):
            st.markdown("MetaboAnalyst 常用设置：Pareto Scaling")
            miss_th = st.slider("剔除缺失率 > X", 0.0, 1.0, 0.5, 0.1)
            impute_m = st.selectbox("填充方法", ["min", "mean", "zero"], index=0)
            norm_m = st.selectbox("样本归一化", ["None", "Sum", "Median"], index=0)
            do_log = st.checkbox("Log2 转化", value=True)
            scale_m = st.selectbox("特征缩放", ["None", "Auto", "Pareto"], index=2)

        current_groups = sorted(raw_df[group_col].astype(str).unique())
        st.markdown("### 5. 组别与对比")
        selected_groups = st.multiselect("纳入分析的组:", current_groups, default=current_groups[:2] if len(current_groups)>=2 else current_groups)
        
        c1, c2 = st.columns(2)
        valid_grps_list = list(selected_groups)
        case_grp = c1.selectbox("Exp (Case)", valid_grps_list, index=0 if valid_grps_list else None)
        ctrl_grp = c2.selectbox("Ctrl (Ref)", valid_grps_list, index=1 if len(valid_grps_list)>1 else 0)
        
        st.markdown("### 6. 统计设置")
        c3, c4 = st.columns(2)
        p_th = c3.number_input("P-value", 0.05, format="%.3f")
        fc_th = c4.number_input("Log2 FC", 1.0)
        
        use_equal_var = st.checkbox("假设方差相等 (Student's t-test)", value=True, help="与 MetaboAnalyst 默认一致")
        enable_jitter = st.checkbox("火山图抖动", value=True)
        
        st.markdown("---")
        submit_button = st.form_submit_button(label='🚀 开始分析 (Run Analysis)')

# ==========================================
# 4. 主逻辑
# ==========================================

if len(selected_groups) < 2:
    if submit_button: st.error("请至少选择 2 个组！")
    else: st.info("👈 请设置参数并点击 '开始分析'")
    st.stop()

with st.spinner("正在计算中..."):
    # 清洗数据
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

    # 差异统计
    if case_grp != ctrl_grp:
        res_stats = run_pairwise_statistics(df_sub, group_col, case_grp, ctrl_grp, feats, equal_var=use_equal_var)
        if feature_meta is not None:
            res_stats = res_stats.merge(feature_meta[['Confidence_Level', 'Clean_Name']], 
                                        left_on='Metabolite', right_index=True, how='left')
            res_stats['Confidence_Level'] = res_stats['Confidence_Level'].fillna('Unknown')
        else: res_stats['Confidence_Level'] = 'N/A'
        
        res_stats['Sig'] = 'NS'
        res_stats.loc[(res_stats['P_Value'] < p_th) & (res_stats['Log2_FC'] > fc_th), 'Sig'] = 'Up'
        res_stats.loc[(res_stats['P_Value'] < p_th) & (res_stats['Log2_FC'] < -fc_th), 'Sig'] = 'Down'
        sig_metabolites = res_stats[res_stats['Sig'] != 'NS']['Metabolite'].tolist()
    else:
        res_stats = pd.DataFrame(); sig_metabolites = []

# ==========================================
# 5. 结果展示
# ==========================================
st.title("📊 代谢组学分析报告")
st.caption(f"对比: {case_grp} vs {ctrl_grp} | 特征数: {len(feats)} | 缩放: {scale_m} | T检验: {'Student' if use_equal_var else 'Welch'}")

tabs = st.tabs(["📊 PCA", "🎯 PLS-DA", "⭐ VIP 特征", "🌋 火山图", "🔥 热图 (美化版)", "📑 详情"])

# --- Tab 1: PCA ---
with tabs[0]:
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        if len(df_sub) < 3: st.warning("样本不足")
        else:
            X = StandardScaler().fit_transform(df_sub[feats])
            pca = PCA(n_components=2).fit(X)
            pcs = pca.transform(X)
            var = pca.explained_variance_ratio_
            fig_pca = px.scatter(x=pcs[:,0], y=pcs[:,1], color=df_sub[group_col], symbol=df_sub[group_col],
                                 color_discrete_sequence=GROUP_COLORS, width=600, height=600)
            fig_pca.update_traces(marker=dict(size=14, line=dict(width=1, color='black'), opacity=0.9))
            update_layout_square(fig_pca, "PCA Score Plot", f"PC1 ({var[0]:.1%})", f"PC2 ({var[1]:.1%})")
            st.plotly_chart(fig_pca, use_container_width=False)

# --- Tab 2: PLS-DA ---
with tabs[1]:
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        if len(df_sub) < 3: st.warning("样本不足")
        else:
            X_pls = df_sub[feats].values
            y_labels = pd.factorize(df_sub[group_col])[0]
            pls_model = PLSRegression(n_components=2).fit(X_pls, y_labels)
            pls_scores = pls_model.x_scores_
            plot_df = pd.DataFrame({'C1': pls_scores[:,0], 'C2': pls_scores[:,1], 'Group': df_sub[group_col].values})
            fig_pls = px.scatter(plot_df, x='C1', y='C2', color='Group', symbol='Group',
                                 color_discrete_sequence=GROUP_COLORS, width=600, height=600)
            for i, grp in enumerate(selected_groups):
                sub_g = plot_df[plot_df['Group'] == grp]
                if len(sub_g) >= 3:
                    ell_x, ell_y = get_ellipse_coordinates(sub_g['C1'], sub_g['C2'])
                    if ell_x is not None:
                        color = GROUP_COLORS[i % len(GROUP_COLORS)]
                        fig_pls.add_trace(go.Scatter(x=ell_x, y=ell_y, mode='lines', line=dict(color=color, width=2, dash='dash'), showlegend=False, hoverinfo='skip'))
            fig_pls.update_traces(marker=dict(size=14, line=dict(width=1.5, color='black'), opacity=1.0))
            update_layout_square(fig_pls, "PLS-DA Score Plot", "Component 1", "Component 2")
            st.plotly_chart(fig_pls, use_container_width=False)

# --- Tab 3: VIP ---
with tabs[2]:
    st.markdown("### Top 25 VIP Features")
    if 'pls_model' in locals():
        vip_scores = calculate_vips(pls_model)
        vip_df = pd.DataFrame({'Metabolite': feats, 'VIP': vip_scores})
        if feature_meta is not None:
             vip_df = vip_df.merge(feature_meta[['Clean_Name']], left_on='Metabolite', right_index=True, how='left')
             vip_df['Display_Name'] = vip_df['Clean_Name'].fillna(vip_df['Metabolite'])
        else: vip_df['Display_Name'] = vip_df['Metabolite']
        top_vip = vip_df.sort_values('VIP', ascending=True).tail(25)
        
        c1, c2, c3 = st.columns([1, 6, 1])
        with c2:
            fig_vip = px.bar(top_vip, x="VIP", y="Display_Name", orientation='h',
                             color="VIP", color_continuous_scale="RdBu_r", width=800, height=700)
            fig_vip.add_vline(x=1.0, line_dash="dash", line_color="black")
            fig_vip.update_traces(marker_line_color='black', marker_line_width=1.0)
            fig_vip.update_layout(
                template="simple_white", width=800, height=700,
                title={'text': "VIP Scores", 'x':0.5, 'xanchor': 'center', 'font': dict(size=20, family="Arial, bold")},
                xaxis=dict(title="VIP Score", showline=True, mirror=True, linewidth=2, linecolor='black'),
                yaxis=dict(title="", showline=True, mirror=True, linewidth=2, linecolor='black'),
                coloraxis_showscale=False, margin=dict(l=200, r=40, t=60, b=60) 
            )
            st.plotly_chart(fig_vip, use_container_width=False)

# --- Tab 4: 火山图 ---
with tabs[3]:
    if case_grp == ctrl_grp: st.warning("请选择不同的组")
    else:
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            plot_df = res_stats.copy()
            x_c, y_c = "Log2_FC", "-Log10_P"
            if enable_jitter:
                np.random.seed(42)
                xr, yr = (plot_df[x_c].max()-plot_df[x_c].min()) or 1, (plot_df[y_c].max()-plot_df[y_c].min()) or 1
                plot_df['Log2_FC_J'] = plot_df[x_c] + np.random.normal(0, xr*0.015, len(plot_df))
                plot_df['-Log10_P_J'] = plot_df[y_c] + np.random.normal(0, yr*0.015, len(plot_df))
                x_c, y_c = "Log2_FC_J", "-Log10_P_J"
            
            hover_dict = {"Metabolite":True, "Log2_FC":':.2f', "P_Value":':.2e', "Confidence_Level":True, x_c:False, y_c:False}
            fig_vol = px.scatter(plot_df, x=x_c, y=y_c, color="Sig", color_discrete_map=COLOR_PALETTE,
                                 hover_data=hover_dict, width=600, height=600)
            fig_vol.add_hline(y=-np.log10(p_th), line_dash="dash", line_color="black", opacity=0.8)
            fig_vol.add_vline(x=fc_th, line_dash="dash", line_color="black", opacity=0.8)
            fig_vol.add_vline(x=-fc_th, line_dash="dash", line_color="black", opacity=0.8)
            fig_vol.update_traces(marker=dict(size=10, opacity=0.75, line=dict(width=1, color='black')))
            update_layout_square(fig_vol, f"Volcano: {case_grp} vs {ctrl_grp}", "Log2 Fold Change", "-Log10(P-value)")
            st.plotly_chart(fig_vol, use_container_width=False)

# --- Tab 5: 热图 (美化修复版) ---
with tabs[4]:
    if not sig_metabolites: st.info("无显著差异物")
    else:
        c1, c2, c3 = st.columns([1, 6, 1])
        with c2:
            top_n = 50
            top_feats = res_stats.sort_values('P_Value').head(top_n)['Metabolite'].tolist()
            # 转置：列=样本，行=代谢物
            hm_data = df_sub.set_index(group_col)[top_feats].T 
            
            # 列颜色 (样本分组)
            sample_groups = df_sub[group_col]
            unique_grps = sample_groups.unique()
            lut = {grp: GROUP_COLORS[i % len(GROUP_COLORS)] for i, grp in enumerate(unique_grps)}
            col_colors = sample_groups.map(lut)
            
            # 使用 Clean Name
            if feature_meta is not None:
                row_labels = [feature_meta.loc[f, 'Clean_Name'] if f in feature_meta.index else f for f in hm_data.index]
                hm_data.index = row_labels

            try:
                g = sns.clustermap(hm_data.astype(float), 
                                   z_score=0, # 行(代谢物) Z-score
                                   cmap="vlag", center=0, 
                                   col_colors=col_colors, 
                                   figsize=(12, 14), 
                                   dendrogram_ratio=(.1, .1), 
                                   # 颜色条放置在顶部中央，水平方向
                                   cbar_pos=(0.35, 0.96, 0.3, 0.02), cbar_kws={'orientation': 'horizontal', 'label': 'Z-Score'}
                                  )
                g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xmajorticklabels(), rotation=90, fontsize=9)
                g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_ymajorticklabels(), rotation=0, fontsize=10)
                # 清除坐标轴标题，防止"Group"覆盖文字
                g.ax_heatmap.set_ylabel("")
                g.ax_heatmap.set_xlabel("")
                st.pyplot(g.fig)
            except Exception as e: st.error(f"绘图错误: {e}")

# --- Tab 6: 详情 ---
with tabs[5]:
    c1, c2 = st.columns([1.5, 1])
    with c1:
        st.subheader("统计表")
        if not res_stats.empty:
            display_df = res_stats.sort_values("P_Value").copy()
            if 'Clean_Name' in display_df.columns: display_df['Name'] = display_df['Clean_Name'].fillna(display_df['Metabolite'])
            else: display_df['Name'] = display_df['Metabolite']
            cols = ["Name", "Log2_FC", "P_Value", "FDR", "Confidence_Level"]
            st.dataframe(display_df[[c for c in cols if c in display_df]].style.format({"Log2_FC": "{:.2f}", "P_Value": "{:.2e}", "FDR": "{:.2e}"})
                         .background_gradient(subset=['P_Value'], cmap="Reds_r", vmin=0, vmax=0.05), use_container_width=True, height=600)
    with c2:
        st.subheader("箱线图")
        feat_options = sorted(feats)
        def_ix = feat_options.index(sig_metabolites[0]) if sig_metabolites else 0
        target_feat = st.selectbox("选择代谢物", feat_options, index=def_ix)
        if target_feat:
            box_df = df_sub[[group_col, target_feat]].copy()
            fig_box = px.box(box_df, x=group_col, y=target_feat, color=group_col,
                             color_discrete_sequence=GROUP_COLORS, points="all", width=500, height=500)
            fig_box.update_traces(width=0.6, marker=dict(size=7, opacity=0.6, line=dict(width=1, color='black')), jitter=0.5, pointpos=0)
            update_layout_square(fig_box, target_feat, "Group", "Log2 Intensity", width=500, height=500)
            st.plotly_chart(fig_box, use_container_width=False)

