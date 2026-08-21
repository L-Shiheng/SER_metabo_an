import pandas as pd
import numpy as np

def generate_ai_prompt(case, ctrl, norm_m, scale_m, R2Y, Q2, b_q2, p_th, fc_th, out_df, pathway_df):
    """生成用于大模型交互分析的 Markdown 提示词"""
    if b_q2 < 0.05 and Q2 > 0.5:
        model_eval_text = f"OPLS-DA 模型预测能力强，且未发生过拟合 (Q²={Q2:.3f}, 截距={b_q2:.3f})。提示两组间存在显著且广泛的全局代谢轮廓差异。"
    elif b_q2 < 0.05 and Q2 <= 0.5:
        model_eval_text = f"模型未过拟合 (截距={b_q2:.3f} < 0.05)，证明所提取的差异特征真实可靠；但 Q²={Q2:.3f} 偏低，提示两组间不存在全局性的代谢颠覆，属于典型的局部微弱代谢表型改变。请重点聚焦于 VIP 和 P 值双显著的核心标志物。"
    else:
        model_eval_text = f"警告：置换检验提示模型存在严重的过拟合风险 (Q²截距={b_q2:.3f} ≥ 0.05)，这意味着组间差异极小或存在系统误差，建议极其谨慎地解读本模型提取的 VIP 值。"

    top_mets = out_df.head(15)['Name'].tolist() if not out_df.empty else ["无显著差异代谢物"]
    top_pws = pathway_df.head(5)['Pathway'].tolist() if not pathway_df.empty else ["无显著富集通路"]

    prompt = f"""# 代谢组学分析专家解读请求

你是一个资深的代谢组学生信专家与生物学家。请根据以下我的实验结果，帮我撰写一段用于学术论文的 `Results` 或 `Discussion` 部分的草稿。

## 1. 实验背景与数据处理
- **实验组别**: {case} vs {ctrl}
- **数据预处理**: 归一化方法为 {norm_m}，特征缩放方法为 {scale_m}。

## 2. 统计学模型评估 (SIMCA 标准)
- **模型指标**: R²Y = {R2Y:.3f}, Q² = {Q2:.3f}, 置换检验 Q² 截距 = {b_q2:.3f}。
- **专家诊断结论**: {model_eval_text}
- **筛选阈值**: P-value < {p_th}, |Log2FC| > {fc_th}, VIP > 1.0

## 3. 核心发现
- **Top 显著差异标志物**: {', '.join(top_mets)}
- **Top 显著改变代谢通路**: {', '.join(top_pws)}

## 你的任务：
1. **模型陈述**: 用专业的学术语言描述上述 OPLS-DA 模型的可靠性（请参考专家诊断结论）。
2. **生物学意义**: 挑选几个核心代谢物和通路，解释它们在生物学或病理学上的潜在联系。
3. **机理推测**: 根据这些差异，推测在这两个组别之间发生了怎样的代谢重编程（Metabolic Reprogramming）？
"""
    return prompt


def generate_offline_html(case, ctrl, feats, p_th, fc_th, norm_m, scale_m, R2Y, Q2, b_q2,
                         out_df, pathway_df, fig_opls, fig_perm, fig_splot, fig_vip, fig_vol, fig_pca,
                         hm_base64, fig_nomogram, fig_pathway, fig_network,
                         vip_show_num, pw_show_num, nomo_num):
    """生成独立离线 HTML 报告，所有 Plotly 图表支持高清 PNG 导出"""
    
    # ------------------- 辅助函数：安全地将 Plotly 图转为 HTML 片段，并配置高清导出 -------------------
    def safe_plotly_html(fig, default_msg="<p style='color:#999;text-align:center;'>图表生成失败或无数据</p>"):
        if fig is None:
            return default_msg
        try:
            config = {
                'displayModeBar': True,
                'toImageButtonOptions': {
                    'format': 'png',
                    'scale': 3,          # 3 倍缩放，适合印刷
                    'filename': 'plot'
                }
            }
            return fig.to_html(full_html=False, include_plotlyjs=False,
                               default_height=600, default_width=700,
                               config=config)
        except Exception as e:
            return f"<p style='color:red;text-align:center;'>图表渲染错误: {str(e)}</p>"
    
    # ------------------- 模型评价动态文本 -------------------
    if b_q2 < 0.05 and Q2 > 0.5:
        model_eval_html = f"<div class='highlight-box success'><strong>模型评价：优秀。</strong> OPLS-DA 模型预测能力强且未过拟合 (Q²={Q2:.3f}, 截距={b_q2:.3f})，表明组间存在显著的代谢表型差异。</div>"
    elif b_q2 < 0.05 and Q2 <= 0.5:
        model_eval_html = f"<div class='highlight-box info'><strong>模型评价：局部差异 (未过拟合)。</strong> 截距={b_q2:.3f} &lt; 0.05 证明差异特征真实有效。Q²={Q2:.3f} &lt; 0.5 提示组间为局部微弱变化而非全局颠覆，需重点关注核心差异标志物。</div>"
    else:
        model_eval_html = f"<div class='highlight-box warning'><strong>模型评价：过拟合风险。</strong> 置换检验截距={b_q2:.3f} ≥ 0.05，模型存在假阳性可能，结果解释需非常谨慎。</div>"

    # ------------------- 构建 HTML 内容 -------------------
    html_content = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MetaFlow Studio 综合代谢组学分析报告 ({case} vs {ctrl})</title>
    <script src="http://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        body {{ font-family: 'Helvetica Neue', Arial, sans-serif; line-height: 1.6; color: #333; max-width: 1300px; margin: 0 auto; padding: 20px; background-color: #f8f9fa; }}
        .container {{ background-color: #fff; padding: 40px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
        h1 {{ color: #1e3c72; border-bottom: 2px solid #1e3c72; padding-bottom: 10px; text-align: center; }}
        h2 {{ color: #2a5298; margin-top: 40px; border-left: 4px solid #2a5298; padding-left: 10px; }}
        h3 {{ color: #1e3c72; margin-top: 30px; }}
        .summary-table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; }}
        .summary-table th, .summary-table td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        .summary-table th {{ background-color: #f2f2f2; color: #333; font-weight: bold; width: 30%; }}
        .plot-container {{ background: #fff; border: 1px solid #eee; border-radius: 4px; padding: 15px; margin-bottom: 25px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }}
        .plot-row {{ display: flex; flex-wrap: wrap; justify-content: space-between; }}
        .plot-row .plot-container {{ flex: 0 0 48%; }}
        @media (max-width: 768px) {{ .plot-row .plot-container {{ flex: 0 0 100%; }} }}
        .highlight-box {{ padding: 15px; border-radius: 4px; margin-bottom: 20px; }}
        .success {{ background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb; }}
        .info {{ background-color: #d1ecf1; color: #0c5460; border: 1px solid #bee5eb; }}
        .warning {{ background-color: #fff3cd; color: #856404; border: 1px solid #ffeeba; }}
        .footer {{ margin-top: 50px; text-align: center; color: #777; font-size: 0.9em; border-top: 1px solid #ddd; padding-top: 20px; }}
        .heatmap-img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🧬 MetaFlow Studio 综合代谢组学分析报告</h1>
        
        <h2>📋 1. 分析概要与模型评价</h2>
        <table class="summary-table">
            <tr><th>对比组别</th><td>{case} (Case) vs {ctrl} (Control)</td></tr>
            <tr><th>预处理方法</th><td>归一化: {norm_m} | 缩放: {scale_m}</td></tr>
            <tr><th>特征筛选阈值</th><td>P-value &lt; {p_th} | |Log2FC| &gt; {fc_th} | VIP &gt; 1.0</td></tr>
            <tr><th>显著差异代谢物数量</th><td>{len(out_df)} 个</td></tr>
            <tr><th>OPLS-DA 评价参数</th><td>R²Y = {R2Y:.3f} | Q² = {Q2:.3f} | 截距 = {b_q2:.3f}</td></tr>
        </table>
        {model_eval_html}

        <h2>🧪 2. 多变量统计与模式识别 (OPLS-DA & PCA)</h2>
        <div class="plot-row">
            <div class="plot-container">{safe_plotly_html(fig_opls)}</div>
            <div class="plot-container">{safe_plotly_html(fig_pca)}</div>
        </div>
        <div class="plot-container">
            <h3 style="text-align:center;">置换检验 (Permutation Test)</h3>
            {safe_plotly_html(fig_perm)}
        </div>

        <h2>🎯 3. 差异代谢物筛选 (单变量 + 多变量)</h2>
        <div class="plot-row">
            <div class="plot-container">{safe_plotly_html(fig_vol)}</div>
            <div class="plot-container">{safe_plotly_html(fig_splot)}</div>
        </div>
        <div class="plot-container">
            <h3 style="text-align:center;">VIP 排序 (Top {vip_show_num})</h3>
            {safe_plotly_html(fig_vip)}
        </div>

        <h2>🔥 4. 差异代谢物热图</h2>
        <div class="plot-container" style="text-align:center;">
            {f'<img src="data:image/png;base64,{hm_base64}" class="heatmap-img" alt="Heatmap">' if hm_base64 else '<p style="color:#999;">无热图数据</p>'}
        </div>

        <h2>📏 5. 诊断列线图 (Nomogram)</h2>
        <div class="plot-container">
            {safe_plotly_html(fig_nomogram, default_msg="<p style='color:#999;text-align:center;'>显著差异代谢物不足或无法构建列线图</p>")}
        </div>

        <h2>🕸️ 6. 通路富集分析 (KEGG)</h2>
        <div class="plot-container">
            {safe_plotly_html(fig_pathway)}
        </div>
        <div class="plot-container">
            <h3 style="text-align:center;">代谢机制网络图</h3>
            {safe_plotly_html(fig_network)}
        </div>

        <h2>📑 7. 显著差异代谢物清单</h2>
        <div style="overflow-x:auto;">
            {out_df[['Name', 'Log2_FC', 'P_Value', 'FDR', 'VIP', 'p_corr']].head(30).to_html(index=False, classes='summary-table', float_format=lambda x: f'{x:.4g}') if not out_df.empty else '<p>无显著差异代谢物</p>'}
        </div>

        <div class="footer">
            <p>报告生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')} | 由 MetaFlow Studio Pro 生成</p>
        </div>
    </div>
</body>
</html>
    """
    return html_content
