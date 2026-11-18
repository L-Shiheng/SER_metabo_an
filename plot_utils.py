import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from scipy.stats import chi2

def update_layout_square(fig, title, xlab, ylab):
    """统一将散点图布局设置为出版级正方形白底黑框样式"""
    fig.update_layout(
        template="simple_white",
        width=600,
        height=600,
        title={'text': title, 'y': 0.95, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'},
        xaxis_title=xlab,
        yaxis_title=ylab,
        xaxis=dict(mirror=True, ticks='outside', showline=True, linecolor='black'),
        yaxis=dict(mirror=True, ticks='outside', showline=True, linecolor='black'),
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99, bgcolor="rgba(255,255,255,0.8)")
    )
    return fig

def get_ellipse_coordinates(x, y, confidence=0.95):
    """计算 95% 置信椭圆坐标，对协方差特征值做非负保护防止 RuntimeWarning"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = ~(np.isnan(x) | np.isnan(y))
    x, y = x[valid], y[valid]
    if len(x) < 3:
        return None, None
    try:
        cov = np.cov(x, y)
        if np.any(np.isnan(cov)) or np.any(np.isinf(cov)):
            return None, None
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        vals = vals[order]
        vecs = vecs[:, order]
        theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
        chi2_val = chi2.ppf(confidence, df=2)
        width, height = 2.0 * np.sqrt(np.maximum(vals, 0.0) * chi2_val)
        if width <= 1e-12 or height <= 1e-12:
            return None, None
        t = np.linspace(0, 2 * np.pi, 100)
        ell = np.array([width / 2.0 * np.cos(t), height / 2.0 * np.sin(t)])
        rad = np.radians(theta)
        R = np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])
        ell_rot = np.dot(R, ell)
        return ell_rot[0, :] + np.mean(x), ell_rot[1, :] + np.mean(y)
    except Exception:
        return None, None

def _format_nomogram_label(name_str, max_len=28):
    """截断超长代谢物名称但完整保留末尾的 [模式] 标识"""
    s = str(name_str).strip()
    if len(s) <= max_len:
        return s
    if s.endswith(']') and '[' in s:
        idx = s.rfind('[')
        mode_part = s[idx:].strip()
        chem_part = s[:idx].strip()
        keep_len = max(8, max_len - len(mode_part) - 4)
        if len(chem_part) > keep_len:
            chem_part = chem_part[:keep_len].rstrip() + "..."
        return f"{chem_part} {mode_part}"
    return s[:max_len - 3].rstrip() + "..."

def plot_nomogram(df_sub, features, feature_names, group_col, case_label):
    """构建基于 L2 正则化逻辑回归的生物标志物联合诊断列线图 (Nomogram)"""
    if len(features) < 2:
        return None
    X = df_sub[features].values.astype(float)
    y = (df_sub[group_col].astype(str) == str(case_label)).astype(int).values
    if len(np.unique(y)) < 2:
        return None

    clf = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000)
    clf.fit(X, y)
    coefs = clf.coef_[0]

    spans = []
    min_vals, max_vals = [], []
    for i in range(len(features)):
        v = X[:, i]
        vmin, vmax = float(np.min(v)), float(np.max(v))
        if abs(vmax - vmin) < 1e-9:
            vmax = vmin + 1.0
        min_vals.append(vmin)
        max_vals.append(vmax)
        spans.append(abs(coefs[i] * (vmax - vmin)))

    max_span = max(spans) if max(spans) > 1e-12 else 1.0

    # 生成保尾截断后的唯一 Y 轴标签
    raw_labels = [_format_nomogram_label(fn, max_len=28) for fn in feature_names]
    unique_labels = []
    seen = set()
    for lbl in raw_labels:
        new_lbl = lbl
        cnt = 1
        while new_lbl in seen:
            new_lbl = f"{lbl}_{cnt}"
            cnt += 1
        seen.add(new_lbl)
        unique_labels.append(new_lbl)

    fig = go.Figure()
    y_positions = list(range(len(features), 0, -1))

    for i, y_pos in enumerate(y_positions):
        pts_max = (spans[i] / max_span) * 100.0
        fig.add_trace(go.Scatter(
            x=[0, pts_max],
            y=[y_pos, y_pos],
            mode='lines+markers+text',
            line=dict(color='#2a5298', width=4),
            marker=dict(size=10, color=['#4DBBD5', '#E64B35'], line=dict(width=1, color='black')),
            text=[
                f"{min_vals[i]:.2f}" if coefs[i] >= 0 else f"{max_vals[i]:.2f}",
                f"{max_vals[i]:.2f}" if coefs[i] >= 0 else f"{min_vals[i]:.2f}"
            ],
            textposition="top center",
            name=unique_labels[i],
            hovertemplate=f"<b>{feature_names[i]}</b><br>Points: %{{x:.1f}}<extra></extra>"
        ))

    fig.update_layout(
        template="simple_white",
        width=860,
        height=max(420, 110 * len(features) + 120),
        title={'text': f"Diagnostic Nomogram (Target: {case_label})", 'x': 0.5, 'xanchor': 'center'},
        xaxis=dict(title="Points (0 - 100 Scale)", range=[-5, 110], gridcolor='#eee', zeroline=False),
        yaxis=dict(
            tickmode='array',
            tickvals=y_positions,
            ticktext=unique_labels,
            range=[0.3, len(features) + 0.7]
        ),
        showlegend=False,
        margin=dict(l=210, r=50, t=50, b=50)
    )
    return fig
