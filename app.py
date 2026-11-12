import streamlit as st
import pandas as pd
import numpy as np
import os
import gc
import datetime
import re
import io
import base64
import subprocess
import sys
import csv 
import traceback
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.decomposition import PCA

# ==========================================
# 0. 品牌更新与 UI 配置
# ==========================================
st.set_page_config(page_title="MetaFlow Studio", page_icon="🧬", layout="wide")

plt.rcParams['font.sans-serif'] = ['PingFang SC', 'SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

try:
    from data_preprocessing import (
        data_cleaning_pipeline, 
        parse_metdna_file, 
        parse_universal_single_table, 
        parse_manual_targeted_files,
        merge_multiple_dfs, 
        align_sample_info, 
        make_unique,
        OPLS_DA, 
        run_pathway_enrichment, 
        build_kegg_dictionary
    )
    from stats_utils import run_pairwise_statistics
    from plot_utils import update_layout_square, get_ellipse_coordinates, plot_nomogram
    from report_generator import generate_offline_html, generate_ai_prompt
except ImportError as e:
    st.error(f"❌ 严重错误：未找到核心依赖文件。详情: {e}")
    st.stop()

has_serrf = False
try:
    from serrf_module import serrf_normalization
    has_serrf = True
except ImportError:
    pass

COLOR_PALETTE = {'Up': '#CD0000', 'Down': '#00008B', 'NS': '#E0E0E0'} 
GROUP_COLORS = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F', '#8491B4', '#91D1C2', '#DC0000', '#7E6148', '#B09C85']
MODE_SYMBOLS = ['circle', 'square', 'diamond', 'cross', 'triangle-up', 'star']

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

st.markdown("""
<div class="logo-container">
    <span style="font-size: 3rem;">♾️</span>
    <span class="logo-text">MetaFlow Studio</span>
    <span class="logo-badge">Pro</span>
</div>
""", unsafe_allow_html=True)

# ==========================================
# 1. 极简侧边栏数据流控制台
# ==========================================
if 'raw_df' not in st.session_state: st.session_state.raw_df = None
if 'feature_meta' not in st.session_state: st.session_state.feature_meta = None
if 'mode_datasets' not in st.session_state: st.session_state.mode_datasets = []
if 'data_loaded' not in st.session_state: st.session_state.data_loaded = False

DEFAULT_4_MODES = ["RP+", "RP-", "HILIC+", "HILIC-"]

def infer_default_mode_name(filename, idx):
    """根据文件名智能推断模式名，若文件名完全相同则按 RP+, RP-, HILIC+, HILIC- 顺序赋予默认值"""
    fn_upper = os.path.splitext(filename)[0].upper()
    if "HILIC" in fn_upper and ("POS" in fn_upper or "+" in fn_upper): return "HILIC+"
    if "HILIC" in fn_upper and ("NEG" in fn_upper or "-" in fn_upper): return "HILIC-"
    if "RP" in fn_upper and ("POS" in fn_upper or "+" in fn_upper): return "RP+"
    if "RP" in fn_upper and ("NEG" in fn_upper or "-" in fn_upper): return "RP-"
    if idx < len(DEFAULT_4_MODES): return DEFAULT_4_MODES[idx]
    return f"Mode_{idx+1}"

with st.sidebar:
    st.header("🛠️ 数据控制台")
    
    data_source = st.radio(
        "选择数据流模式", 
        ["1. MA 标准单表 (自带分组)", "2. 拟靶向 MRM 宽表 (需后缀)", "3. MetDNA 原始宽表"], 
        index=0
    )
    
    info_df = None; candidate_samples = []; user_sample_col = None; user_group_col = None
    excluded_samples = []
    use_serrf = False; serrf_ready = False
    run_order_col = None; sample_type_col = None; qc_label = "QC"
    
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
            except Exception as e: st.error(f"Info 读取失败: {e}")
            
        excluded_samples = st.multiselect("2. 样本剔除 (黑名单)", options=candidate_samples, default=[])
        
        use_serrf = st.checkbox("3. 启用 SERRF 批次校正 (按单表独立执行)", value=False)
        if use_serrf:
            if not has_serrf: st.warning("⚠️ 缺失 serrf_module.py")
            elif info_df is not None:
                idx_order = next((i for i, c in enumerate(cols_lower) if any(x in c for x in ['order', 'run', 'idx', 'seq'])), 0)
                sc1, sc2, sc3 = st.columns(3)
                run_order_col = sc1.selectbox("进样顺序列", cols, index=idx_order)
                sample_type_col = sc2.selectbox("样本类型列", cols, index=cols.index(user_group_col) if user_group_col else 0)
                default_qc = next((v for v in info_df[sample_type_col].unique().astype(str) if 'qc' in v.lower()), "QC")
                qc_label = sc3.text_input("QC标识", value=default_qc)
                serrf_ready = True
    else:
        st.markdown("#### 1. 上传数据矩阵")
        ex_str = st.text_input("2. 样本剔除 (选填，逗号分隔)")
        if ex_str: excluded_samples = [s.strip() for s in ex_str.split(',') if s.strip()]

    st.markdown("#### 3. KEGG 通路配置")
    species = st.selectbox("物种背景", ["Human (Homo sapiens)", "Mouse (Mus musculus)", "Rat (Rattus norvegicus)", "General (所有物种)"], index=0)
    species_code = {"Human (Homo sapiens)": "hsa", "Mouse (Mus musculus)": "mmu", "Rat (Rattus norvegicus)": "rno", "General (所有物种)": "map"}[species]
    db_filename = f"kegg_{species_code}.csv"
    custom_pathway_file = st.file_uploader("自定义通路库 (.csv)", type=["csv", "gmt"], key="pathway_db")
    
    db_sync_msg = st.session_state.get('_db_sync_msg', '')
    if st.button(f"🔄 同步 {species_code} 通路库", width='stretch'):
        with st.spinner(f"⏳ 正在后台解析 KGML 抓取 {species_code} 核心代谢通路（约需 1-2 分钟）..."):
            try:
                res_sub = subprocess.run([sys.executable, "get_kegg_db.py", species_code], capture_output=True, text=True)
                if res_sub.returncode == 0:
                    st.session_state['_db_sync_msg'] = f"✅ {species_code} 库同步成功！\n*(更新时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')})*"
                    st.toast("✅ 库同步成功！")
                    st.rerun() 
                else:
                    st.session_state['_db_sync_msg'] = f"❌ 更新脚本执行报错:\n```text\n{res_sub.stderr or res_sub.stdout}\n```"
            except Exception as e:
                st.session_state['_db_sync_msg'] = f"❌ 无法启动更新程序: {str(e)}"
                
    if db_sync_msg:
        if db_sync_msg.startswith('✅'): st.success(db_sync_msg)
        else: st.error(db_sync_msg)
        
    if os.path.exists(db_filename):
        st.caption(f"📦 当前库：`{db_filename}`（{sum(1 for _ in open(db_filename, encoding='utf-8'))-1} 条核心代谢通路）")
    else:
        st.caption(f"⚠️ 未找到 `{db_filename}`，请点击上方同步按钮。")

    st.markdown("#### 4. 上传分析数据")
    feature_scope = "全部特征"
    dict_files = None
    custom_mode_tags = []
    
    if data_source == "1. MA 标准单表 (自带分组)":
        dict_files = st.file_uploader("关联 MetDNA 字典 (可选)", type=["csv", "xlsx"], accept_multiple_files=True, key="dict_files")
        uploaded_files = st.file_uploader("上传 MA 格式单表", type=["csv", "xlsx"], accept_multiple_files=True, key="data")
    elif data_source == "2. 拟靶向 MRM 宽表 (需后缀)":
        suffix = st.text_input("提取指标后缀", value=" : 面积")
        dict_files = st.file_uploader("关联 MetDNA 字典 (可选)", type=["csv", "xlsx"], accept_multiple_files=True, key="dict_files")
        uploaded_files = st.file_uploader("上传 MRM 宽表", type=["csv", "xlsx"], accept_multiple_files=True, key="data")
    else:
        feature_scope = st.radio("特征范围", ["仅已注释特征", "全部特征"], index=0)
        uploaded_files = st.file_uploader("上传 MetDNA 结果表 (支持 4 模式同时上传)", type=["csv", "xlsx"], accept_multiple_files=True, key="data")
        if uploaded_files:
            st.markdown("##### 📌 采集模式标识 (用于图表与清单溯源)")
            st.caption("即使上传的 4 个表格文件名完全相同，系统也会按以下模式名严格区分：")
            cols_m = st.columns(2)
            raw_tags = []
            for idx, f_obj in enumerate(uploaded_files):
                default_tag = infer_default_mode_name(f_obj.name, idx)
                with cols_m[idx % 2]:
                    t_val = st.text_input(f"文件 {idx+1} ({f_obj.name[:12]})", value=default_tag, key=f"mode_tag_{idx}")
                    raw_tags.append(t_val.strip() if t_val.strip() else f"Mode_{idx+1}")
            custom_mode_tags = make_unique(raw_tags)
        
    start_process = st.container().button("📥 加载数据矩阵", width='stretch', type="primary")


def apply_single_table_qc_and_serrf(df_t, meta, mode_tag, info_df, user_sample_col, user_group_col, excluded_samples, use_serrf, serrf_ready, run_order_col, sample_type_col, qc_label):
    """辅助函数：在单个模式表格内部独立完成分组对齐、样本黑名单剔除与 SERRF 批次校正"""
    if info_df is not None:
        info_aligned = align_sample_info(df_t, info_df, sample_col_name=user_sample_col)
        if user_group_col and user_group_col in info_aligned.columns:
            df_t['Group'] = info_aligned[user_group_col].fillna('Unknown').values

    if excluded_samples:
        ex_fps = set([re.sub(r'[^a-z0-9]', '', str(s).strip().lower()) for s in excluded_samples])
        df_t = df_t[~df_t['SampleID'].astype(str).apply(lambda s: re.sub(r'[^a-z0-9]', '', str(s).strip().lower())).isin(ex_fps)].reset_index(drop=True)

    if use_serrf and serrf_ready and has_serrf and info_df is not None:
        try:
            aligned_info = align_sample_info(df_t, info_df, sample_col_name=user_sample_col)
            num_cols = [c for c in df_t.columns if c not in ['SampleID', 'Group', 'Source_Files']]
            df_data_in = df_t[num_cols].copy()
            df_corrected, rsd_info = serrf_normalization(df_data_in, aligned_info, run_order_col, sample_type_col, qc_label)
            if df_corrected is not None:
                df_t = df_t.loc[df_corrected.index].copy()
                df_t.loc[:, num_cols] = df_corrected.values
                if meta is not None and 'Mean_Area' in meta.columns:
                    meta.loc[num_cols, 'Mean_Area'] = df_t[num_cols].mean(axis=0).values
                st.toast(f"✅ [{mode_tag}] SERRF 校正完成: QC 中位 RSD 从 {rsd_info['RSD_Before']:.1f}% 降至 {rsd_info['RSD_After']:.1f}%")
            else:
                st.warning(f"⚠️ [{mode_tag}] SERRF 跳过: {rsd_info}")
        except Exception as e:
            st.warning(f"⚠️ [{mode_tag}] SERRF 异常 ({str(e)})，已保留未校正数据。")
    return df_t, meta

# ==========================================
# 2. 核心路由与解析引擎调用
# ==========================================
if start_process:
    if 'analysis_res' in st.session_state: del st.session_state['analysis_res']
        
    if not uploaded_files: 
        st.error("请先上传主数据文件！")
    else:
        progress_bar = st.progress(0)
        final_df = None; final_meta = None; mode_datasets = []
        
        if data_source == "1. MA 标准单表 (自带分组)":
            with st.spinner("启动 MA 引擎..."):
                ext_dict = build_kegg_dictionary(dict_files) if dict_files else {}
                df_t, meta, err = parse_universal_single_table(uploaded_files, external_kegg_dict=ext_dict)
                if err: st.error(err)
                else:
                    df_t, meta = apply_single_table_qc_and_serrf(df_t, meta, "MA_Single", None, None, None, excluded_samples, False, False, None, None, None)
                    final_df, final_meta = df_t, meta
                    mode_datasets = [(df_t, meta, "MA_Single")]

        elif data_source == "2. 拟靶向 MRM 宽表 (需后缀)":
            if info_df is None: st.error("⚠️ 请先上传 Info 表格！")
            else:
                with st.spinner("启动 MRM 引擎..."):
                    ext_dict = build_kegg_dictionary(dict_files) if dict_files else {}
                    df_t, meta, err = parse_manual_targeted_files(uploaded_files, metric_suffix=suffix, external_kegg_dict=ext_dict)
                    if err: st.error(err)
                    else:
                        df_t, meta = apply_single_table_qc_and_serrf(df_t, meta, "MRM_Targeted", info_df, user_sample_col, user_group_col, excluded_samples, use_serrf, serrf_ready, run_order_col, sample_type_col, qc_label)
                        final_df, final_meta = df_t, meta
                        mode_datasets = [(df_t, meta, "MRM_Targeted")]

        else:
            if info_df is None: 
                st.error("⚠️ 请先上传 Info 表格！")
            else:
                with st.spinner("启动 MetDNA 分模式独立质控与三级漏斗解析引擎..."):
                    parsed_results = []
                    for i, file in enumerate(uploaded_files):
                        assigned_tag = custom_mode_tags[i] if i < len(custom_mode_tags) else infer_default_mode_name(file.name, i)
                        df_t, meta, err = parse_metdna_file(
                            file, file.name, valid_samples=candidate_samples,
                            dedup_within_file=True, custom_mode_tag=assigned_tag
                        )
                        
                        if not err:
                            if feature_scope.startswith("仅已注释"):
                                anno_ids = set(meta[meta['Is_Annotated'] == True].index.astype(str).tolist())
                                keep_cols = ['SampleID', 'Group', 'Source_Files'] + [c for c in df_t.columns if str(c) in anno_ids]
                                df_t = df_t[keep_cols].copy()
                                meta = meta.loc[meta.index.astype(str).isin([str(c) for c in df_t.columns])].copy()
                            
                            df_t, meta = apply_single_table_qc_and_serrf(
                                df_t, meta, assigned_tag, info_df, user_sample_col, user_group_col,
                                excluded_samples, use_serrf, serrf_ready, run_order_col, sample_type_col, qc_label
                            )
                            parsed_results.append((df_t, meta, assigned_tag))
                        else:
                            st.error(f"❌ 文件 {file.name} ({assigned_tag}) 解析失败: {err}")
                            
                        progress_bar.progress((i + 1) / len(uploaded_files))
                    
                    if parsed_results:
                        raw_df, merged_meta, err = merge_multiple_dfs(parsed_results)
                        if err: st.error(err)
                        else:
                            info_aligned = align_sample_info(raw_df, info_df, sample_col_name=user_sample_col)
                            if user_group_col and user_group_col in info_aligned.columns: 
                                raw_df['Group'] = info_aligned[user_group_col].fillna('Unknown').values
                            final_df, final_meta = raw_df, merged_meta
                            mode_datasets = parsed_results
                    else:
                        st.error("⚠️ 未能提取到有效数据，请检查 Info 表格式！")

        if final_df is not None:
            st.session_state.raw_df = final_df
            st.session_state.feature_meta = final_meta
            st.session_state.mode_datasets = mode_datasets
            st.session_state.data_loaded = True
            st.rerun()

# ==========================================
# 3. 统计与可视化参数设置 (含双模式开关与双背景开关)
# ==========================================
if not st.session_state.data_loaded:
    st.info("👈 请在左侧控制台按对应数据格式上传数据。")
    st.stop()

raw_df = st.session_state.raw_df
mode_datasets = st.session_state.get('mode_datasets', [])
mode_count = len(mode_datasets)

if mode_count > 1:
    mode_summary_str = " &nbsp;|&nbsp; ".join([f"**`{m_name}`** ({len(m_df.columns)-3} 特征)" for m_df, _, m_name in mode_datasets])
    st.info(f"📊 **多模式数据总览**: 已成功加载 **{mode_count}** 个独立采集模式 [ {mode_summary_str} ] &nbsp;➔&nbsp; 跨模式合并去重后共 **{len(raw_df)}** 样本 x **{len(raw_df.columns)-3}** 唯一代谢物")
else:
    st.info(f"📊 **数据总览**: {len(raw_df)} 样本 x {len(raw_df.columns)-3} 特征")

st.download_button("📥 导出跨模式合并矩阵 (.csv)", raw_df.to_csv(index=False).encode('utf-8-sig'), "MetaFlow_Merged_Matrix.csv", "text/csv")
st.divider()

with st.form(key='analysis_form'):
    st.markdown("### ⚙️ 分析参数配置")
    non_num = raw_df.select_dtypes(exclude=[np.number]).columns.tolist()
    group_col = st.selectbox("分组列", non_num, index=non_num.index('Group') if 'Group' in non_num else 0)
    
    c_sw1, c_sw2 = st.columns(2)
    with c_sw1:
        integration_mode = st.radio(
            "🔀 多模式数据整合与统计策略",
            [
                "模式 A：单模式独立归一化 ➔ 跨模式优选去重 ➔ 全局统一建模 (推荐临床/机器学习)",
                "模式 B：各模式独立统计建模 ➔ 差异代谢物去重合并 ➔ 通路富集 (JHM 经典非靶向流派)"
            ],
            index=0,
            disabled=(mode_count <= 1),
            help="当上传多个模式文件（如 RP+, RP-, HILIC+, HILIC-）时，可随时切换两种学术界主流范式，无需重新上传数据。"
        )
    with c_sw2:
        kegg_bg_option = st.radio(
            "🎯 KEGG 超几何分布富集背景 (Background Universe)",
            [
                "物种全库参考背景 (同 MetaboAnalyst 默认，高灵敏度)",
                "本次实验实测背景 (严格模式，仅以检出代谢物为分母 N)"
            ],
            index=0,
            help="物种全库参考背景以当前物种 KEGG 代谢网络全集为 N；实验实测背景以本次上传表格中检出的 KEGG 代谢物为 N。"
        )
    
    with st.expander("数据预处理", expanded=False):
        c_p1, c_p2 = st.columns(2)
        miss_th = c_p1.slider("缺失率过滤", 0.0, 1.0, 0.20)
        impute_m = c_p2.selectbox("缺失值填充", ["KNN", "min", "mean", "zero"], index=0)
        c_p3, c_p4 = st.columns(2)
        norm_m = c_p3.selectbox("样本归一化 (单模式内独立执行)", ["PQN", "Median", "Sum", "None"], index=0)
        scale_m = c_p4.selectbox("特征缩放", ["Pareto", "Auto", "None"], index=0)
        do_log = st.checkbox("Log2 对数转化", value=True)

    with st.expander("图表设置", expanded=False):
        c_t1, c_t2, c_t3, c_t4 = st.columns(4)
        vip_show_num = c_t1.slider("VIP 展示数", 10, 50, 25)
        vol_label_num = c_t2.slider("火山图直接标注数", 0, 30, 10, help="在火山图上直接显示『化合物名 [模式]』文字标签的 Top 差异物数量")
        nomo_num = c_t3.slider("列线图标志物数", 2, 8, 4)
        pw_show_num = c_t4.slider("网络图通路数", 5, 30, 15)

    cur_grps = sorted(raw_df[group_col].astype(str).unique())
    default_grps = [g for g in cur_grps if 'qc' not in g.lower()][:2]
    if len(default_grps) < 2: default_grps = cur_grps[:2]
    sel_grps = st.multiselect("选择对比组 (限 2 组)", cur_grps, default=default_grps)
    c1, c2, c3, c4 = st.columns(4)
    case = c1.selectbox("Case 组 (分子)", list(sel_grps), index=0 if sel_grps else None)
    ctrl = c2.selectbox("Control 组 (分母)", list(sel_grps), index=1 if len(sel_grps)>1 else 0)
    p_th = c3.number_input("P-value 阈值", value=0.05, step=0.01)
    fc_th = c4.number_input("Log2 FC 阈值", value=0.58, step=0.10)
    submit_button = st.form_submit_button(label='🚀 执行分析')


# ==========================================
# 辅助函数：构建带模式标注与形状区分的火山图
# ==========================================
def build_volcano_figure(stats_df, p_th, fc_th, title="Volcano Plot", top_label_n=10):
    plot_df = stats_df.copy()
    if 'Source_Mode' not in plot_df.columns:
        plot_df['Source_Mode'] = 'Global'
    if 'Detected_Modes' not in plot_df.columns:
        plot_df['Detected_Modes'] = plot_df['Source_Mode']
        
    plot_df['Name_With_Mode'] = plot_df['Name'].astype(str) + " [" + plot_df['Source_Mode'].astype(str) + "]"
    
    unique_modes = plot_df['Source_Mode'].astype(str).unique().tolist()
    if len(unique_modes) > 1:
        fig_vol = px.scatter(
            plot_df, x="Log2_FC", y="-Log10_P", color="Sig", symbol="Source_Mode",
            color_discrete_map=COLOR_PALETTE, symbol_sequence=MODE_SYMBOLS,
            hover_name="Name_With_Mode",
            hover_data={'Source_Mode': True, 'Detected_Modes': True, 'VIP': ':.2f', 'Log2_FC': ':.2f', 'P_Value': ':.2e', 'FDR': ':.2e'}
        )
    else:
        fig_vol = px.scatter(
            plot_df, x="Log2_FC", y="-Log10_P", color="Sig",
            color_discrete_map=COLOR_PALETTE,
            hover_name="Name_With_Mode",
            hover_data={'Source_Mode': True, 'VIP': ':.2f', 'Log2_FC': ':.2f', 'P_Value': ':.2e', 'FDR': ':.2e'}
        )
        
    fig_vol.update_traces(marker=dict(size=9, opacity=0.85, line=dict(width=0.5, color='white')))
    fig_vol.add_hline(y=-np.log10(p_th), line_dash="dash", line_color="gray")
    fig_vol.add_vline(x=fc_th, line_dash="dash", line_color="gray")
    fig_vol.add_vline(x=-fc_th, line_dash="dash", line_color="gray")

    # 在火山图上直接为 Top 显著差异标志物标注 "化合物名 [采集模式]"
    if top_label_n > 0:
        if 'Is_Biomarker' in plot_df.columns:
            sig_sub = plot_df[plot_df['Is_Biomarker'] == True].copy()
        else:
            sig_sub = plot_df[plot_df['Sig'] != 'NS'].copy()
        if sig_sub.empty:
            sig_sub = plot_df[plot_df['Sig'] != 'NS'].copy()
        if not sig_sub.empty:
            top_annot = sig_sub.sort_values(['P_Value', 'VIP'], ascending=[True, False]).head(int(top_label_n))
            for _, r in top_annot.iterrows():
                fig_vol.add_annotation(
                    x=float(r['Log2_FC']), y=float(r['-Log10_P']),
                    text=f"{r['Name']} [{r['Source_Mode']}]",
                    showarrow=True, arrowhead=1, arrowsize=0.8, arrowwidth=1,
                    arrowcolor="#555", ax=0, ay=-20,
                    font=dict(size=10, color="#222"),
                    bgcolor="rgba(255,255,255,0.8)"
                )

    fig_vol = update_layout_square(fig_vol, title, "Log2 FC", "-Log10(P)")
    fig_vol.update_layout(width=720, height=650)
    return fig_vol


# ==========================================
# 辅助函数：单矩阵统计建模与图表生成
# ==========================================
def run_single_matrix_modeling(df_proc, df_unscaled_log2, feats, meta, group_col, sel_grps, case, ctrl, p_th, fc_th, vip_show_num, vol_label_num=10, mode_title=""):
    prefix = f"[{mode_title}] " if mode_title else ""
    df_sub = df_proc[df_proc[group_col].isin(sel_grps)].copy()
    df_log_sub = df_unscaled_log2[df_unscaled_log2[group_col].isin(sel_grps)].copy()
    
    if len(df_sub) < 4: raise ValueError(f"{prefix}样本数太少 ({len(df_sub)}个)，无法建模。")
    if len(feats) < 2: raise ValueError(f"{prefix}有效特征太少 ({len(feats)}个)，无法建模。")

    stats_df = run_pairwise_statistics(df_sub, group_col, case, ctrl, feats, df_unscaled_log2=df_log_sub)
    
    if meta is not None and 'Clean_Name' in meta.columns: 
        merge_cols = [c for c in ['Clean_Name', 'Original_Name', 'KEGG_ID', 'Source_Mode', 'Detected_Modes'] if c in meta.columns]
        meta_dedup = meta[~meta.index.duplicated(keep='first')]
        stats_df = stats_df.merge(meta_dedup[merge_cols], left_on='Metabolite', right_index=True, how='left')
        stats_df['Name'] = stats_df['Clean_Name'].fillna(stats_df['Metabolite'])
        if 'Source_Mode' not in stats_df.columns:
            stats_df['Source_Mode'] = mode_title or 'Global'
        else:
            stats_df['Source_Mode'] = stats_df['Source_Mode'].fillna(mode_title or 'Global')
        if 'Detected_Modes' not in stats_df.columns:
            stats_df['Detected_Modes'] = stats_df['Source_Mode']
        else:
            stats_df['Detected_Modes'] = stats_df['Detected_Modes'].fillna(stats_df['Source_Mode'])
        
        if 'KEGG_ID' in stats_df.columns:
            kegg_str = stats_df['KEGG_ID'].fillna('').astype(str).replace('nan', '').replace('None', '')
            orig_str = stats_df['Original_Name'].fillna(stats_df['Name']).astype(str)
            stats_df['Search_Name'] = np.where(kegg_str != '', orig_str + '|' + kegg_str, orig_str)
        else:
            stats_df['Search_Name'] = stats_df['Original_Name'].fillna(stats_df['Name'])
    else:
        stats_df['Name'] = stats_df['Search_Name'] = stats_df['Metabolite']
        stats_df['Source_Mode'] = mode_title or 'Global'
        stats_df['Detected_Modes'] = stats_df['Source_Mode']

    stats_df['Sig'] = 'NS'
    stats_df.loc[(stats_df['P_Value'] < p_th) & (stats_df['Log2_FC'] > fc_th), 'Sig'] = 'Up'
    stats_df.loc[(stats_df['P_Value'] < p_th) & (stats_df['Log2_FC'] < -fc_th), 'Sig'] = 'Down'

    y_binary = np.where(df_sub[group_col] == case, 1, -1)
    X_matrix = df_sub[feats].values
    
    opls = OPLS_DA()
    opls.fit(X_matrix, y_binary)
    corrs, r2_perm, q2_perm, R2Y, Q2 = opls.permutation_test(X_matrix, y_binary, n_permutations=100)
    
    m_q2, b_q2 = np.polyfit(corrs, q2_perm, 1) if len(corrs) > 0 else (0, 0)
    m_r2, b_r2 = np.polyfit(corrs, r2_perm, 1) if len(corrs) > 0 else (0, 0)

    vip_df = pd.DataFrame({'Metabolite': feats, 'VIP': opls.vip, 'p_corr': opls.p_corr})
    stats_df = stats_df.merge(vip_df, on='Metabolite')
    stats_df['Name_With_Mode'] = stats_df['Name'].astype(str) + " [" + stats_df['Source_Mode'].astype(str) + "]"
    stats_df['Is_Biomarker'] = (stats_df['VIP'] > 1.0) & (stats_df['P_Value'] < p_th) & (stats_df['Log2_FC'].abs() > fc_th)
    out_df = stats_df[stats_df['Is_Biomarker']].sort_values('VIP', ascending=False).copy()

    # 1. OPLS-DA 得分图
    opls_score_df = pd.DataFrame({'t1 (Predictive)': opls.t, 't_ortho (Orthogonal)': opls.t_ortho, 'Group': df_sub[group_col].values, 'SampleID': df_sub['SampleID'].values})
    fig_opls = px.scatter(opls_score_df, x='t1 (Predictive)', y='t_ortho (Orthogonal)', color='Group', symbol='Group', hover_data=['SampleID'], color_discrete_sequence=GROUP_COLORS)
    for i, g in enumerate(list(sel_grps)):
        sub_grp = opls_score_df[opls_score_df['Group'] == g]
        if len(sub_grp) >= 3:
            el_x, el_y = get_ellipse_coordinates(sub_grp['t1 (Predictive)'], sub_grp['t_ortho (Orthogonal)'])
            if el_x is not None: fig_opls.add_trace(go.Scatter(x=el_x, y=el_y, mode='lines', line=dict(color=GROUP_COLORS[i % len(GROUP_COLORS)], width=2, dash='dash'), showlegend=False, hoverinfo='skip'))
    fig_opls.update_traces(marker=dict(size=14, line=dict(width=1, color='black'), opacity=0.9))
    fig_opls.add_hline(y=0, line_dash="dash", line_color="gray"); fig_opls.add_vline(x=0, line_dash="dash", line_color="gray")
    fig_opls = update_layout_square(fig_opls, f"{prefix}OPLS-DA Score Plot", "t [1]", "to [1]")

    # 2. 置换检验图
    fig_perm = go.Figure()
    fig_perm.add_trace(go.Scatter(x=corrs, y=r2_perm, mode='markers', name='R2', marker=dict(color='green', symbol='circle-open', size=8)))
    fig_perm.add_trace(go.Scatter(x=corrs, y=q2_perm, mode='markers', name='Q2', marker=dict(color='blue', symbol='square-open', size=8)))
    fig_perm.add_trace(go.Scatter(x=[1], y=[R2Y], mode='markers', name='Original R2', marker=dict(color='green', symbol='circle', size=12)))
    fig_perm.add_trace(go.Scatter(x=[1], y=[Q2], mode='markers', name='Original Q2', marker=dict(color='blue', symbol='square', size=12)))
    x_line = np.array([0, 1])
    fig_perm.add_trace(go.Scatter(x=x_line, y=m_r2*x_line + b_r2, mode='lines', name=f'R2 Line (Int: {b_r2:.2f})', line=dict(color='green', dash='dash')))
    fig_perm.add_trace(go.Scatter(x=x_line, y=m_q2*x_line + b_q2, mode='lines', name=f'Q2 Line (Int: {b_q2:.2f})', line=dict(color='blue', dash='dash')))
    fig_perm.update_layout(template="simple_white", width=600, height=600, title={'text': f"{prefix}Permutation Test", 'y':0.95, 'x':0.5, 'xanchor': 'center'}, xaxis_title="Correlation", yaxis_title="R2 / Q2")

    # 3. S-Plot (带模式名称提示)
    splot_df = stats_df.copy()
    splot_df['Color'] = np.where(splot_df['Is_Biomarker'], f'VIP>1 & P<{p_th}', 'NS')
    fig_splot = px.scatter(
        splot_df, x='Log2_FC', y='p_corr', color='Color',
        hover_name='Name_With_Mode', hover_data=['Source_Mode', 'VIP', 'P_Value'],
        color_discrete_map={f'VIP>1 & P<{p_th}': '#CD0000', 'NS': '#E0E0E0'}
    )
    fig_splot.add_hline(y=0.5, line_dash="dash", line_color="gray"); fig_splot.add_hline(y=-0.5, line_dash="dash", line_color="gray")
    fig_splot = update_layout_square(fig_splot, f"{prefix}S-Plot", "Log2 Fold Change", "p(corr)")

    # 4. VIP 图 (纵轴直接显示 化合物名 [模式])
    top_vip_df = stats_df.sort_values('VIP', ascending=True).tail(vip_show_num).copy()
    top_vip_df['VIP_Label'] = make_unique(top_vip_df['Name_With_Mode'].tolist())
    fig_vip = px.bar(top_vip_df, x="VIP", y="VIP_Label", orientation='h', color="VIP", hover_data=['Source_Mode', 'Log2_FC', 'P_Value'], color_continuous_scale="RdBu_r")
    fig_vip.add_vline(x=1.0, line_dash="dash", line_color="black")
    fig_vip.update_layout(template="simple_white", width=820, height=700, title={'text': f"{prefix}Top VIP Scores", 'x':0.5, 'xanchor': 'center'}, yaxis_title="Metabolite [Mode]", coloraxis_showscale=False)

    # 5. PCA 图
    fig_pca = None
    if len(df_proc) >= 3:
        valid_feats_pca = df_proc[feats].var()[df_proc[feats].var() > 1e-9].index.tolist()
        if len(valid_feats_pca) >= 2:
            X_pca_input = df_proc[valid_feats_pca].values
            pca_all = PCA(n_components=2).fit(X_pca_input)
            pcs_all = pca_all.transform(X_pca_input)
            var_all = pca_all.explained_variance_ratio_
            pca_df_all = pd.DataFrame({'PC1': pcs_all[:, 0], 'PC2': pcs_all[:, 1], 'Group': df_proc[group_col].values, 'SampleID': df_proc['SampleID']})
            fig_pca = px.scatter(pca_df_all, x='PC1', y='PC2', color='Group', symbol='Group', hover_data=['SampleID'], color_discrete_sequence=GROUP_COLORS)
            for i, g in enumerate(sorted(df_proc[group_col].unique())):
                sub_grp = pca_df_all[pca_df_all['Group'] == g]
                if len(sub_grp) >= 3:
                    el_x, el_y = get_ellipse_coordinates(sub_grp['PC1'], sub_grp['PC2'])
                    if el_x is not None: fig_pca.add_trace(go.Scatter(x=el_x, y=el_y, mode='lines', line=dict(color=GROUP_COLORS[i % len(GROUP_COLORS)], width=1, dash='dot'), showlegend=False, hoverinfo='skip'))
            fig_pca.update_traces(marker=dict(size=14, line=dict(width=1, color='black'), opacity=0.9))
            fig_pca.add_hline(y=0, line_dash="dash", line_color="gray"); fig_pca.add_vline(x=0, line_dash="dash", line_color="gray")
            fig_pca = update_layout_square(fig_pca, f"{prefix}PCA Plot", f"PC1 ({var_all[0]:.1%})", f"PC2 ({var_all[1]:.1%})")

    # 6. 火山图 (含模式形状区分与静态文字标签)
    fig_vol = build_volcano_figure(stats_df, p_th, fc_th, title=f"{prefix}Volcano Plot", top_label_n=vol_label_num)

    return {
        'df_sub': df_sub, 'stats_df': stats_df, 'out_df': out_df,
        'R2Y': R2Y, 'Q2': Q2, 'b_q2': b_q2,
        'fig_opls': fig_opls, 'fig_perm': fig_perm, 'fig_splot': fig_splot,
        'fig_vip': fig_vip, 'fig_pca': fig_pca, 'fig_vol': fig_vol
    }


# ==========================================
# 4. 执行核心分析计算 (支持模式 A 与模式 B 双轨运行)
# ==========================================
if submit_button:
    if len(sel_grps) != 2 or case == ctrl: 
        st.error("⚠️ 必须选择 2 个不同的对比组！")
        st.stop()

    plt.close('all')
    use_mode_b = integration_mode.startswith("模式 B") and (len(mode_datasets) > 1)
    bg_mode_key = "reference" if kegg_bg_option.startswith("物种全库") else "experimental"

    with st.spinner("正在运行代谢组学统计与通路分析引擎..."):
        try:
            mode_results = {}
            mode_summary_rows = []
            fig_vol_merged_b = None

            if not use_mode_b:
                # ==========================================================
                # 轨道 A：各模式独立归一化 ➔ 跨模式优选去重 ➔ 全局统一建模
                # ==========================================================
                if len(mode_datasets) > 1:
                    normed_mode_list = []
                    for m_df, m_meta, m_tag in mode_datasets:
                        m_norm, m_feats = data_cleaning_pipeline(
                            m_df, group_col, missing_thresh=miss_th,
                            impute_method=impute_m, norm_method=norm_m,
                            log_transform=False, scale_method='None'
                        )
                        m_meta_sub = m_meta.loc[m_meta.index.intersection(m_feats)].copy()
                        normed_mode_list.append((m_norm, m_meta_sub, m_tag))
                    
                    merged_norm_df, meta, err = merge_multiple_dfs(normed_mode_list)
                    if err: raise RuntimeError(err)
                    grp_map = dict(zip(st.session_state.raw_df['SampleID'], st.session_state.raw_df[group_col]))
                    merged_norm_df[group_col] = merged_norm_df['SampleID'].map(grp_map).fillna('Unknown')
                    df_proc, feats, df_unscaled_log2 = data_cleaning_pipeline(
                        merged_norm_df, group_col, missing_thresh=1.0,
                        impute_method='zero', norm_method='None',
                        log_transform=do_log, scale_method=scale_m, return_unscaled=True
                    )
                else:
                    raw_df = st.session_state.raw_df
                    meta = st.session_state.feature_meta
                    df_proc, feats, df_unscaled_log2 = data_cleaning_pipeline(
                        raw_df, group_col, missing_thresh=miss_th, 
                        impute_method=impute_m, norm_method=norm_m, 
                        log_transform=do_log, scale_method=scale_m, return_unscaled=True
                    )

                global_res = run_single_matrix_modeling(
                    df_proc, df_unscaled_log2, feats, meta, group_col,
                    sel_grps, case, ctrl, p_th, fc_th, vip_show_num, vol_label_num=vol_label_num
                )
                df_sub = global_res['df_sub']
                stats_df = global_res['stats_df']
                out_df = global_res['out_df']
                R2Y, Q2, b_q2 = global_res['R2Y'], global_res['Q2'], global_res['b_q2']
                fig_opls, fig_perm = global_res['fig_opls'], global_res['fig_perm']
                fig_splot, fig_vip = global_res['fig_splot'], global_res['fig_vip']
                fig_pca, fig_vol = global_res['fig_pca'], global_res['fig_vol']
                bg_universe_list = stats_df['Search_Name'].unique().tolist()
                integration_label = "模式 A：单模式独立归一化 ➔ 跨模式优选去重 ➔ 全局统一建模"

            else:
                # ==========================================================
                # 轨道 B：各模式独立统计建模 ➔ 差异物按最小 P 值裁决去重 ➔ 通路富集
                # ==========================================================
                all_mode_out_dfs = []
                all_mode_stats_dfs = []
                all_mode_bg_names = set()
                scaled_sub_dfs = []

                for m_df, m_meta, m_tag in mode_datasets:
                    m_proc, m_feats, m_unscaled_log2 = data_cleaning_pipeline(
                        m_df, group_col, missing_thresh=miss_th,
                        impute_method=impute_m, norm_method=norm_m,
                        log_transform=do_log, scale_method=scale_m, return_unscaled=True
                    )
                    m_res = run_single_matrix_modeling(
                        m_proc, m_unscaled_log2, m_feats, m_meta, group_col,
                        sel_grps, case, ctrl, p_th, fc_th, vip_show_num, vol_label_num=vol_label_num, mode_title=m_tag
                    )
                    mode_results[m_tag] = m_res
                    
                    n_up = int((m_res['out_df']['Log2_FC'] > 0).sum()) if not m_res['out_df'].empty else 0
                    n_down = int((m_res['out_df']['Log2_FC'] < 0).sum()) if not m_res['out_df'].empty else 0
                    diag_str = "优秀" if (m_res['b_q2'] < 0.05 and m_res['Q2'] > 0.5) else ("局部差异(未过拟合)" if m_res['b_q2'] < 0.05 else "过拟合风险")
                    mode_summary_rows.append({
                        '采集模式': m_tag,
                        '特征数': len(m_feats),
                        '显著差异物总数': len(m_res['out_df']),
                        '上调 (Up)': n_up,
                        '下调 (Down)': n_down,
                        'R2Y': m_res['R2Y'],
                        'Q2': m_res['Q2'],
                        'Q2截距': m_res['b_q2'],
                        '模型诊断': diag_str
                    })
                    all_mode_bg_names.update(m_res['stats_df']['Search_Name'].dropna().unique().tolist())
                    all_mode_stats_dfs.append(m_res['stats_df'])
                    if not m_res['out_df'].empty:
                        all_mode_out_dfs.append(m_res['out_df'])
                    
                    m_sub_idx = m_res['df_sub'].drop_duplicates(subset=['SampleID']).set_index('SampleID')
                    scaled_sub_dfs.append((m_sub_idx, m_feats))

                # 构建模式 B 跨模式汇总火山图 (展示所有模式的特征分布及模式来源)
                if all_mode_stats_dfs:
                    combined_all_stats = pd.concat(all_mode_stats_dfs, ignore_index=True)
                    fig_vol_merged_b = build_volcano_figure(
                        combined_all_stats, p_th, fc_th,
                        title="Combined Volcano Plot (All Modes)", top_label_n=vol_label_num
                    )

                # 汇总四个模式的显著差异代谢物并按 Clean_Name 去重（最小 P 值优先裁决 + 记录所有显著模式）
                if all_mode_out_dfs:
                    combined_sig = pd.concat(all_mode_out_dfs, ignore_index=True)
                    sig_modes_map = combined_sig.groupby('Name')['Source_Mode'].apply(lambda s: ", ".join(list(dict.fromkeys(s.astype(str))))).to_dict()
                    combined_sig = combined_sig.sort_values(['P_Value', 'VIP'], ascending=[True, False])
                    out_df = combined_sig.drop_duplicates(subset=['Name'], keep='first').copy()
                    out_df['Detected_Modes'] = out_df['Name'].map(sig_modes_map)
                    out_df['Name_With_Mode'] = out_df['Name'].astype(str) + " [" + out_df['Source_Mode'].astype(str) + "]"
                    out_df = out_df.sort_values('VIP', ascending=False).reset_index(drop=True)
                else:
                    out_df = pd.DataFrame(columns=['Metabolite', 'Name', 'Name_With_Mode', 'Search_Name', 'Source_Mode', 'Detected_Modes', 'Log2_FC', 'P_Value', 'FDR', 'VIP', 'p_corr'])

                # 拼接各模式缩放后的特征矩阵供全局热图与列线图提取代表特征使用
                base_sub = mode_results[list(mode_results.keys())[0]]['df_sub'][['SampleID', group_col]].drop_duplicates(subset=['SampleID']).set_index('SampleID')
                feat_frames = [s_df[s_feats] for s_df, s_feats in scaled_sub_dfs]
                df_sub = pd.concat([base_sub] + feat_frames, axis=1, join='outer').fillna(0).reset_index()

                best_m = max(mode_results.keys(), key=lambda k: mode_results[k]['Q2'])
                R2Y, Q2, b_q2 = mode_results[best_m]['R2Y'], mode_results[best_m]['Q2'], mode_results[best_m]['b_q2']
                fig_opls, fig_perm = mode_results[best_m]['fig_opls'], mode_results[best_m]['fig_perm']
                fig_splot, fig_pca = mode_results[best_m]['fig_splot'], mode_results[best_m]['fig_pca']
                fig_vol = fig_vol_merged_b if fig_vol_merged_b is not None else mode_results[best_m]['fig_vol']

                if not out_df.empty:
                    top_vip_merged = out_df.sort_values('VIP', ascending=True).tail(vip_show_num).copy()
                    top_vip_merged['Display_Name'] = make_unique(top_vip_merged['Name_With_Mode'].tolist())
                    fig_vip = px.bar(top_vip_merged, x="VIP", y="Display_Name", orientation='h', color="VIP", hover_data=['Detected_Modes', 'Log2_FC', 'P_Value'], color_continuous_scale="RdBu_r")
                    fig_vip.add_vline(x=1.0, line_dash="dash", line_color="black")
                    fig_vip.update_layout(template="simple_white", width=820, height=700, title={'text': "Merged Top Biomarkers VIP (Across Modes)", 'x':0.5, 'xanchor': 'center'}, yaxis_title="Metabolite [Best Mode]", coloraxis_showscale=False)
                else:
                    fig_vip = mode_results[best_m]['fig_vip']

                bg_universe_list = list(all_mode_bg_names)
                integration_label = "模式 B：各采集模式独立统计建模 ➔ 差异代谢物去重合并 (JHM 经典流派)"

            mode_summary_df = pd.DataFrame(mode_summary_rows) if mode_summary_rows else None

            # ==========================================================
            # 公共下游 1：300 DPI 高清聚类热图 (生成稳定 Base64 PNG 二进制流，独立标签页展示)
            # ==========================================================
            hm_base64 = ""
            sig_mets = [m for m in out_df['Metabolite'].tolist() if m in df_sub.columns]
            if sig_mets:
                hm_feats = sig_mets[:50]
                label_lookup = dict(zip(out_df['Metabolite'], out_df.get('Name_With_Mode', out_df['Name'])))
                df_sub_sorted = df_sub.sort_values(by=[group_col, 'SampleID']).reset_index(drop=True)
                
                sample_col_labels = make_unique([f"{sid} ({grp})" for sid, grp in zip(df_sub_sorted['SampleID'], df_sub_sorted[group_col])])
                hm_raw = df_sub_sorted[hm_feats].T.astype(float)
                hm_raw.columns = sample_col_labels
                hm_raw.index = make_unique([str(label_lookup.get(f, f)) for f in hm_raw.index])
                
                row_mean = hm_raw.mean(axis=1)
                row_std = hm_raw.std(axis=1).replace(0, 1.0)
                hm_z = hm_raw.sub(row_mean, axis=0).div(row_std, axis=0).replace([np.inf, -np.inf], np.nan).fillna(0)
                
                unique_grps = df_sub_sorted[group_col].unique().tolist()
                lut = {g: GROUP_COLORS[i % len(GROUP_COLORS)] for i, g in enumerate(unique_grps)}
                col_colors_series = pd.Series(
                    [lut[g] for g in df_sub_sorted[group_col]],
                    index=sample_col_labels,
                    name="Group"
                )
                
                try:
                    plt.close('all')
                    g = sns.clustermap(
                        hm_z, z_score=None, cmap="vlag", center=0, 
                        col_colors=col_colors_series,
                        figsize=(10, max(6.5, min(16, len(hm_feats) * 0.30 + 3))),
                        row_cluster=(len(hm_feats) >= 2), col_cluster=False
                    )
                    g.ax_heatmap.set_xlabel("")
                    g.ax_heatmap.set_ylabel("")
                    plt.setp(g.ax_heatmap.get_xticklabels(), rotation=45, ha="right", fontsize=9)
                    plt.setp(g.ax_heatmap.get_yticklabels(), fontsize=9)
                    
                    buf = io.BytesIO()
                    g.savefig(buf, format='png', dpi=300, bbox_inches='tight')
                    buf.seek(0)
                    hm_base64 = base64.b64encode(buf.read()).decode('utf-8')
                    plt.close(g.fig)
                except Exception as hm_err:
                    st.warning(f"⚠️ 热图渲染提示: {str(hm_err)}")

            # ==========================================================
            # 公共下游 2：诊断列线图 (Nomogram)
            # ==========================================================
            fig_nomogram = None
            if len(sig_mets) >= 2:
                try:
                    nomo_sub_df = out_df[out_df['Metabolite'].isin(sig_mets)].head(min(nomo_num, len(sig_mets)))
                    nomo_labels = nomo_sub_df.get('Name_With_Mode', nomo_sub_df['Name']).tolist()
                    fig_nomogram = plot_nomogram(df_sub, nomo_sub_df['Metabolite'].tolist(), nomo_labels, group_col, case)
                except Exception:
                    pass

            # ==========================================================
            # 公共下游 3：KEGG 通路富集与机制网络图
            # ==========================================================
            pathway_df, filtered_db_df, fig_pathway, fig_network = pd.DataFrame(), pd.DataFrame(), None, None
            sig_mets_fullnames = out_df['Search_Name'].dropna().unique().tolist()
            if sig_mets_fullnames:
                pathway_df, filtered_db_df = run_pathway_enrichment(
                    sig_mets_fullnames, bg_universe_list,
                    custom_db_source=custom_pathway_file if custom_pathway_file else db_filename,
                    bg_mode=bg_mode_key
                )
                if not pathway_df.empty and 'Error' not in pathway_df.columns:
                    plot_pw_df = pathway_df[pathway_df['Hits'] > 0].head(pw_show_num)
                    fig_pathway = px.scatter(
                        plot_pw_df, x='Enrichment_Factor', y='-Log10_P', size='Hits', color='P_Value',
                        hover_name='Pathway', hover_data={'Hit_Metabolites': True, 'P_Value': ':.4f', 'FDR': ':.4f'},
                        color_continuous_scale='Reds_r', size_max=40
                    )
                    fig_pathway.update_traces(marker=dict(line=dict(width=1, color='black'), opacity=0.75))
                    fig_pathway.add_hline(y=-np.log10(0.05), line_dash="dash", line_color="gray")
                    fig_pathway.update_layout(template="simple_white", width=800, height=600, title={'text': "KEGG Pathway Enrichment", 'x':0.5, 'xanchor': 'center'}, xaxis_title="Enrichment Factor", yaxis_title="-Log10(P)")
                    
                    sig_pws = pathway_df[pathway_df['P_Value'] < 0.05].head(pw_show_num)
                    if sig_pws.empty: sig_pws = pathway_df.head(min(5, len(pathway_df)))
                    if not sig_pws.empty:
                        G = nx.Graph()
                        fc_dict = {}
                        mode_dict = {}
                        for _, r_row in out_df.iterrows():
                            k1 = str(r_row['Name']).split(';')[0].strip()
                            k2 = str(r_row['Search_Name']).split('|')[0].split(';')[0].strip()
                            fc_dict[k1] = r_row['Log2_FC']; fc_dict[k2] = r_row['Log2_FC']
                            mode_dict[k1] = r_row.get('Source_Mode', ''); mode_dict[k2] = r_row.get('Source_Mode', '')
                            
                        for _, row in sig_pws.iterrows():
                            pw_name = row['Pathway']
                            G.add_node(pw_name, node_type='pathway', size=max(16, min(38, -np.log10(row['P_Value']) * 10)), disp_name=pw_name)
                            if pd.notna(row['Hit_Metabolites']) and str(row['Hit_Metabolites']).strip() != "":
                                for hit in [m.strip() for m in str(row['Hit_Metabolites']).split(',') if m.strip()]:
                                    fc_val = fc_dict.get(hit, 0.0)
                                    m_tag_str = mode_dict.get(hit, '')
                                    disp_label = f"{hit} [{m_tag_str}]" if m_tag_str else hit
                                    G.add_node(hit, node_type='metabolite', size=11, fc=fc_val, disp_name=disp_label)
                                    G.add_edge(pw_name, hit)
                                    
                        if len(G.nodes) > 0:
                            pos = nx.spring_layout(G, k=0.75, seed=42)
                            edge_x, edge_y = [], []
                            for e0, e1 in G.edges():
                                edge_x.extend([pos[e0][0], pos[e1][0], None])
                                edge_y.extend([pos[e0][1], pos[e1][1], None])
                                
                            edge_trace = go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(width=1.2, color='#999'), hoverinfo='none')
                            node_trace = go.Scatter(
                                x=[pos[n][0] for n in G.nodes()],
                                y=[pos[n][1] for n in G.nodes()],
                                mode='markers+text',
                                text=[G.nodes[n].get('disp_name', n) for n in G.nodes()],
                                textposition="top center",
                                marker=dict(
                                    symbol=['square' if G.nodes[n]['node_type']=='pathway' else 'circle' for n in G.nodes()],
                                    color=['#FFD700' if G.nodes[n]['node_type']=='pathway' else ('#CD0000' if G.nodes[n]['fc'] >= 0 else '#00008B') for n in G.nodes()],
                                    size=[G.nodes[n]['size'] for n in G.nodes()],
                                    line_width=1, line_color='black'
                                )
                            )
                            fig_network = go.Figure(data=[edge_trace, node_trace])
                            fig_network.update_layout(title={'text': "Mechanism Network (Pathway - Metabolite [Mode])", 'y':0.95, 'x':0.5, 'xanchor': 'center'}, showlegend=False, xaxis=dict(showgrid=False, zeroline=False, showticklabels=False), yaxis=dict(showgrid=False, zeroline=False, showticklabels=False), width=900, height=700, plot_bgcolor='white')

            html_report = generate_offline_html(
                case, ctrl, feats if not use_mode_b else [], p_th, fc_th, norm_m, scale_m, R2Y, Q2, b_q2,
                out_df, pathway_df, fig_opls, fig_perm, fig_splot, fig_vip, fig_vol, fig_pca,
                hm_base64, fig_nomogram, fig_pathway, fig_network, vip_show_num, pw_show_num, nomo_num,
                mode_results=mode_results if use_mode_b else None,
                mode_summary_df=mode_summary_df,
                integration_label=integration_label
            )
            prompt_md = generate_ai_prompt(case, ctrl, norm_m, scale_m, R2Y, Q2, b_q2, p_th, fc_th, out_df, pathway_df, mode_summary_df=mode_summary_df)

            st.session_state['analysis_res'] = {
                'case': case, 'ctrl': ctrl, 'R2Y': R2Y, 'Q2': Q2, 'b_q2': b_q2,
                'use_mode_b': use_mode_b, 'mode_results': mode_results, 'mode_summary_df': mode_summary_df,
                'fig_vol_merged_b': fig_vol_merged_b,
                'fig_opls': fig_opls, 'fig_perm': fig_perm, 'fig_splot': fig_splot,
                'fig_vip': fig_vip, 'fig_pca': fig_pca, 'fig_vol': fig_vol,
                'hm_base64': hm_base64, 'out_df': out_df, 'pathway_df': pathway_df,
                'filtered_db_df': filtered_db_df, 
                'fig_nomogram': fig_nomogram, 'fig_pathway': fig_pathway, 'fig_network': fig_network,
                'html_report': html_report, 'prompt_md': prompt_md
            }
        except Exception as e:
            st.error(f"❌ 分析计算失败。错误摘要: {str(e)}")
            with st.expander("点击查看详细报错日志"): st.code(traceback.format_exc())

# ==========================================
# 5. UI 展示层 (火山图与聚类热图拆分为独立全宽标签页)
# ==========================================
if 'analysis_res' in st.session_state:
    res = st.session_state['analysis_res']
    
    st.title("📊 综合代谢组学分析报告")
    if res.get('use_mode_b') and res.get('mode_summary_df') is not None:
        st.markdown(f"**对比**: `{res['case']}` vs `{res['ctrl']}` &nbsp;&nbsp;|&nbsp;&nbsp; **整合流派**: `模式 B (各采集模式独立统计建模 ➔ 差异物去重富集)` &nbsp;&nbsp;|&nbsp;&nbsp; **去重后总差异代谢物**: `{len(res['out_df'])}` 个")
        st.dataframe(res['mode_summary_df'].style.format({"R2Y": "{:.3f}", "Q2": "{:.3f}", "Q2截距": "{:.3f}"}), width='stretch')
    else:
        st.markdown(f"**对比**: `{res['case']}` vs `{res['ctrl']}` &nbsp;&nbsp;|&nbsp;&nbsp; **模型**: R²Y = `{res['R2Y']:.3f}` &nbsp;&nbsp;|&nbsp;&nbsp; Q² = `{res['Q2']:.3f}` &nbsp;&nbsp;|&nbsp;&nbsp; **显著差异物**: `{len(res['out_df'])}` 个")
        b_q2_val, q2_val = res['b_q2'], res['Q2']
        if b_q2_val < 0.05 and q2_val > 0.5:
            st.success(f"✅ OPLS-DA 模型预测能力强，且未发生过拟合 (Q²={q2_val:.3f}, 截距={b_q2_val:.3f})")
        elif b_q2_val < 0.05 and q2_val <= 0.5:
            st.info(f"💡 模型未过拟合，但组间整体代谢差异偏弱 (Q²={q2_val:.3f} < 0.5, 截距={b_q2_val:.3f})")
        else:
            st.warning(f"⚠️ 警告：模型存在严重的过拟合风险，请谨慎解读 VIP 值！(Q²截距={b_q2_val:.3f} ≥ 0.05)")

    def render_mode_chart(fig_key, include_merged_vol=False):
        """若为模式 B 则自动生成各采集模式的切换子标签页"""
        if res.get('use_mode_b') and res.get('mode_results'):
            m_names = list(res['mode_results'].keys())
            if include_merged_vol and res.get('fig_vol_merged_b') is not None:
                sub_tabs = st.tabs(["🌐 四模式汇总火山图"] + [f"🔬 {m}" for m in m_names])
                with sub_tabs[0]:
                    st.plotly_chart(res['fig_vol_merged_b'], key="vol_merged_all_modes")
                for idx, m in enumerate(m_names):
                    with sub_tabs[idx + 1]:
                        fig_obj = res['mode_results'][m].get(fig_key)
                        if fig_obj: st.plotly_chart(fig_obj, key=f"{fig_key}_{m}")
                        else: st.warning("图表生成失败")
            else:
                sub_tabs = st.tabs([f"🔬 {m}" for m in m_names])
                for idx, m in enumerate(m_names):
                    with sub_tabs[idx]:
                        fig_obj = res['mode_results'][m].get(fig_key)
                        if fig_obj: st.plotly_chart(fig_obj, key=f"{fig_key}_{m}")
                        else: st.warning("图表生成失败")
        else:
            if res.get(fig_key): st.plotly_chart(res[fig_key], key=f"{fig_key}_global")
            else: st.warning("图表生成失败")

    # 拆分为 12 个独立标签页：火山图 (tabs[5]) 与 聚类热图 (tabs[6]) 完全分开显示
    tabs = st.tabs([
        "🎯 OPLS-DA",
        "🔄 置换检验",
        "🧬 S-Plot",
        "📊 VIP",
        "🌐 PCA",
        "🌋 火山图",
        "🔥 聚类热图",
        "📑 清单",
        "📏 列线图",
        "🕸️ 通路富集",
        "🔗 机制网络图",
        "📄 导出报告"
    ])
    
    with tabs[0]:
        c1, c2 = st.columns([1, 5])
        with c2: render_mode_chart('fig_opls')
            
    with tabs[1]:
        c1, c2 = st.columns([1, 5])
        with c2: render_mode_chart('fig_perm')
            
    with tabs[2]:
        c1, c2 = st.columns([1, 5])
        with c2: render_mode_chart('fig_splot')
            
    with tabs[3]:
        c1, c2 = st.columns([1, 6])
        with c2: 
            if res.get('use_mode_b') and res.get('mode_results'):
                m_names = list(res['mode_results'].keys())
                vip_tabs = st.tabs(["🏆 跨模式合并 Top VIP"] + [f"🔬 {m}" for m in m_names])
                with vip_tabs[0]:
                    if res['fig_vip']: st.plotly_chart(res['fig_vip'], key="vip_main")
                for idx, m in enumerate(m_names):
                    with vip_tabs[idx + 1]:
                        if res['mode_results'][m].get('fig_vip'):
                            st.plotly_chart(res['mode_results'][m]['fig_vip'], key=f"vip_{m}")
            else:
                if res['fig_vip']: st.plotly_chart(res['fig_vip'], key="vip_main")
                else: st.warning("图表生成失败")
            
    with tabs[4]:
        c1, c2 = st.columns([1, 5])
        with c2: render_mode_chart('fig_pca')
            
    # 独立标签页 1：🌋 火山图 (全宽展示，带模式形状与化合物[模式]标签)
    with tabs[5]:
        c1, c2 = st.columns([1, 6])
        with c2:
            render_mode_chart('fig_vol', include_merged_vol=True)
            
    # 独立标签页 2：🔥 聚类热图 (全宽展示 Base64 PNG 二进制图，永不丢失，自带 300DPI 下载按钮)
    with tabs[6]:
        st.markdown("#### 🔥 显著差异代谢物分层聚类热图 (Top 50, 附带来源模式标识)")
        hm_b64_str = res.get('hm_base64', '')
        if hm_b64_str:
            hm_bytes = base64.b64decode(hm_b64_str)
            st.download_button(
                "📸 下载 300 DPI 出版级高清热图 (.png)",
                data=hm_bytes,
                file_name=f"Heatmap_{res['case']}_vs_{res['ctrl']}_300DPI.png",
                mime="image/png",
                type="primary"
            )
            c_h1, c_h2, c_h3 = st.columns([1, 5, 1])
            with c_h2:
                st.image(hm_bytes, caption=f"Top Significant Differential Metabolites Heatmap ({res['case']} vs {res['ctrl']})")
        else:
            st.info("当前筛选阈值下无显著差异代谢物，无法生成聚类热图。")
            
    with tabs[7]:
        show_cols = [c for c in ['Name', 'Source_Mode', 'Detected_Modes', 'Log2_FC', 'P_Value', 'FDR', 'VIP', 'p_corr'] if c in res['out_df'].columns]
        if res.get('use_mode_b') and res.get('mode_results'):
            m_names = list(res['mode_results'].keys())
            table_tabs = st.tabs(
                [f"🏆 跨模式去重总表 ({len(res['out_df'])} 个)"] +
                [f"🔬 {m} 独立统计结果 ({len(res['mode_results'][m]['out_df'])} 个)" for m in m_names]
            )
            with table_tabs[0]:
                st.markdown("#### 🏆 跨模式汇总去重后的显著差异标志物清单 (以最小 P 值模式为代表)")
                st.dataframe(res['out_df'][show_cols].style.format({"Log2_FC":"{:.2f}", "P_Value":"{:.3e}", "FDR":"{:.3e}", "VIP":"{:.2f}", "p_corr":"{:.2f}"}).background_gradient(subset=['VIP'], cmap="Reds"), width='stretch')
                st.download_button("📥 导出跨模式去重总差异物表 (.csv)", res['out_df'][show_cols].to_csv(index=False).encode('utf-8-sig'), f"Merged_Biomarkers_{res['case']}_vs_{res['ctrl']}.csv", "text/csv", key="dl_merged_bio")
                
            for idx, m in enumerate(m_names):
                with table_tabs[idx + 1]:
                    m_out = res['mode_results'][m]['out_df']
                    m_stats = res['mode_results'][m]['stats_df']
                    m_cols = [c for c in ['Name', 'Source_Mode', 'Log2_FC', 'P_Value', 'FDR', 'VIP', 'p_corr', 'Sig'] if c in m_stats.columns]
                    st.markdown(f"#### 🔬 模式 `{m}` 显著差异代谢物清单 (共 {len(m_out)} 个)")
                    st.dataframe(m_out[m_cols].style.format({"Log2_FC":"{:.2f}", "P_Value":"{:.3e}", "FDR":"{:.3e}", "VIP":"{:.2f}", "p_corr":"{:.2f}"}).background_gradient(subset=['VIP'], cmap="Reds"), width='stretch')
                    c_dl1, c_dl2 = st.columns(2)
                    with c_dl1:
                        st.download_button(f"📥 导出 [{m}] 显著差异代谢物 (.csv)", m_out[m_cols].to_csv(index=False).encode('utf-8-sig'), f"Biomarkers_{m}_{res['case']}_vs_{res['ctrl']}.csv", "text/csv", key=f"dl_sig_{m}")
                    with c_dl2:
                        st.download_button(f"📥 导出 [{m}] 全特征完整统计表 (.csv)", m_stats[m_cols].to_csv(index=False).encode('utf-8-sig'), f"All_Stats_{m}_{res['case']}_vs_{res['ctrl']}.csv", "text/csv", key=f"dl_all_{m}")
                    with st.expander(f"展开查看模式 `{m}` 全部 {len(m_stats)} 个特征的统计结果"):
                        st.dataframe(m_stats[m_cols].style.format({"Log2_FC":"{:.2f}", "P_Value":"{:.3e}", "FDR":"{:.3e}", "VIP":"{:.2f}", "p_corr":"{:.2f}"}), width='stretch')
        else:
            st.markdown("### 🏆 显著差异生物标志物清单")
            st.dataframe(res['out_df'][show_cols].style.format({"Log2_FC":"{:.2f}", "P_Value":"{:.3e}", "FDR":"{:.3e}", "VIP":"{:.2f}", "p_corr":"{:.2f}"}).background_gradient(subset=['VIP'], cmap="Reds"), width='stretch')
            st.download_button("📥 导出差异代谢物清单 (.csv)", res['out_df'][show_cols].to_csv(index=False).encode('utf-8-sig'), f"Biomarkers_{res['case']}_vs_{res['ctrl']}.csv", "text/csv", key="dl_single_bio")
        
    with tabs[8]:
        c1, c2 = st.columns([1, 6])
        with c2:
            if res['fig_nomogram']: st.plotly_chart(res['fig_nomogram'], key="nomo_main")
            else: st.warning("⚠️ 显著差异代谢物不足 2 个或分组异常，无法构建列线图。")
            
    with tabs[9]:
        if 'filtered_db_df' in res and not res['filtered_db_df'].empty:
            st.download_button("📥 导出专属 MA 背景库", res['filtered_db_df'].to_csv(index=False, header=False, quoting=csv.QUOTE_ALL).encode('utf-8'), f"MA_Background_{res['case']}_{res['ctrl']}.csv", "text/csv", type="primary")
        c1, c2 = st.columns([1, 6])
        with c2:
            if res['pathway_df'].empty: st.warning("未能匹配到富集通路。")
            else:
                if res['fig_pathway']: st.plotly_chart(res['fig_pathway'], key="pw_main")
                st.dataframe(res['pathway_df'].drop(columns=['-Log10_P'], errors='ignore').style.format({"P_Value":"{:.3e}", "FDR":"{:.3e}", "Enrichment_Factor":"{:.2f}"}).background_gradient(subset=['P_Value'], cmap="Reds_r", vmin=0, vmax=0.05), width='stretch')
                
    with tabs[10]:
        if res['fig_network']: st.plotly_chart(res['fig_network'], key="net_main")
        else: st.info("没有找到通路与代谢物的有效映射。")
        
    with tabs[11]:
        c_rep1, c_rep2 = st.columns(2)
        with c_rep1: st.download_button("📥 离线网页报告 (.html)", res['html_report'].encode('utf-8'), f"Report_{res['case']}_{res['ctrl']}.html", "text/html")
        with c_rep2: st.download_button("📥 AI Prompt (.md)", res['prompt_md'].encode('utf-8'), f"Prompt_{res['case']}_{res['ctrl']}.md", "text/markdown")
