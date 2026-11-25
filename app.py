"""Depo-Vision - PVD AlCu 증착 품질 예측 대시보드 (Streamlit)."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import logic as L

st.set_page_config(page_title="Depo-Vision", page_icon="🛡️", layout="wide")

# ---------------------------------------------------------------- 스타일
BG, PANEL, BORDER, TEXT, MUTED, ACCENT = "#0d1117", "#121821", "#243041", "#e6edf3", "#9aa7b5", "#38bdf8"
RED, YELLOW, TEAL, BLUE, ORANGE = "#f87171", "#fbbf24", "#5eead4", "#60a5fa", "#fb923c"

st.markdown(f"""
<style>
.stApp {{ background: {BG}; }}
.block-container {{ padding-top: 2rem; }}
.card {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 12px; padding: 14px 16px; height: 100%; }}
.card .k {{ font-size: 12px; color: {MUTED}; }}
.card .v {{ font-size: 28px; font-weight: 700; margin-top: 2px; }}
.card .s {{ font-size: 12px; color: {MUTED}; margin-top: 2px; }}
.kv {{ display: flex; justify-content: space-between; gap: 10px; font-size: 14px; padding: 6px 0; border-bottom: 1px solid #1c2633; }}
.kv span:first-child {{ color: {MUTED}; }}
.box {{ background: {BG}; border: 1px solid {BORDER}; border-radius: 10px; padding: 12px 14px; }}
.small {{ font-size: 12px; color: {MUTED}; }}
</style>
""", unsafe_allow_html=True)


def card(label, value, sub="", color=TEXT, border=BORDER):
    st.markdown(f'<div class="card" style="border-color:{border}"><div class="k">{label}</div>'
                f'<div class="v" style="color:{color}">{value}</div><div class="s">{sub}</div></div>',
                unsafe_allow_html=True)


def td_html(text):
    """'T_d'를 아래첨자 HTML로."""
    return text.replace("T_d", "T<sub>d</sub>")


def base_layout(fig, height=360, ytitle="", xtitle=""):
    fig.update_layout(
        height=height, margin=dict(l=10, r=90, t=10, b=10), paper_bgcolor=PANEL, plot_bgcolor=PANEL,
        font=dict(color=TEXT, size=12), showlegend=False, hoverlabel=dict(bgcolor="#1c2633", bordercolor="#3a4656", font=dict(size=13, color="#e6edf3"), align="left"))
    fig.update_xaxes(gridcolor=BORDER, zeroline=False, title=xtitle, title_font=dict(color=MUTED, size=12),
                     tickfont=dict(color=MUTED))
    fig.update_yaxes(gridcolor=BORDER, zeroline=False, title=ytitle, title_font=dict(color=MUTED, size=12),
                     tickfont=dict(color=MUTED))
    return fig


# ---------------------------------------------------------------- 데이터
@st.cache_data(show_spinner="원본 데이터 불러오는 중")
def get_raw():
    df = L.load_raw()
    pm, pbar = L.p_monthly(df)
    return df, L.spc_weekly(df), pm, pbar


@st.cache_resource(show_spinner="모델 불러오는 중")
def get_processed():
    return L.load_processed()


df, SPC, PM, PBAR = get_raw()
P = get_processed()
K, LSL_RHO = P["C"]["K_rho"], P["C"]["RHO_LSL"]
META, X = P["meta"], P["X_test"]

# ---------------------------------------------------------------- 사이드바
with st.sidebar:
    st.markdown("## 🛡️ Depo-Vision")
    st.caption("PVD AlCu 증착 품질 예측")
    page = st.radio("화면", ["설비 SPC 모니터링", "Thickness · Rs 예측", "What-if 시뮬레이션"], label_visibility="collapsed")
    st.divider()
    st.markdown(f"""<div class="small">
    ① Thickness 17개 위치 예측<br>② Rs 17개 위치 예측<br>
    ③ Resistivity = Thickness × Rs × 1.472<br>④ Resistivity LSL Margin으로 위험 등급<br><br>
    Thickness · Rs: 주간 평균 I-MR 관리도<br>불량률: 월별 p 관리도<br>
    관리 한계: 2023.07 - 2024.06 (52주)</div>""", unsafe_allow_html=True)


def wafer_selector(prefix):
    """머신 → 설비 → 웨이퍼 선택 (평가 구간 웨이퍼)."""
    c1, c2, c3 = st.columns(3)
    num = lambda s: int(s.split("_")[1])
    machine = c1.selectbox("머신", sorted(META.machine.unique(), key=num), key=prefix + "_m")
    eqs = sorted(META[META.machine == machine].equipment.unique(), key=num)
    eq = c2.selectbox("설비", eqs, key=prefix + "_e")
    sub = META[META.equipment == eq].sort_values("runstart", ascending=False)
    idx = c3.selectbox("웨이퍼", sub.index, key=prefix + "_w",
                       format_func=lambda i: f"{META.wafer_id[i]} · {META.runstart[i]:%Y-%m-%d}")
    return idx


# ================================================================ 1. SPC 모니터링
def page_spc():
    head, sw = st.columns([3, 2])
    metric = sw.radio("지표", ["Thickness", "Rs", "불량률"], horizontal=True, label_visibility="collapsed")
    is_p = metric == "불량률"
    key = "t" if metric == "Thickness" else "r"
    head.markdown("# 설비 SPC 모니터링")
    head.caption("설비별 월간 Resistivity 불량률" if is_p else f"설비별 주간 평균 {metric}")

    A = {e: (L.analyze_p(PM[e], PBAR) if is_p else L.analyze_i(SPC[e][key])) for e in SPC}
    A = {e: a for e, a in A.items() if a}
    cnt = lambda s: sum(a["status"] == s for a in A.values())
    cols = st.columns(3 if is_p else 4)
    with cols[0]:
        card("모니터링 설비", f"{len(A) - cnt('가동 종료')} / 16", "machine_1 3대 가동 종료")
    with cols[1]:
        card("관리 한계 이탈", f"{cnt('이탈')}대", "UCL 초과 월 발생" if is_p else "±3σ 밖 주간 평균 발생", RED, "#5c1d1d")
    if not is_p:
        with cols[2]:
            card("치우침 경고", f"{cnt('치우침')}대", "중심선 한쪽 연속 8주 이상", YELLOW, "#5a3a12")
    with cols[-1]:
        card("정상", f"{cnt('정상')}대", "관리 상태 유지", TEAL)
    st.write("")

    icon = {"이탈": "🔴", "치우침": "🟡", "정상": "🟢", "가동 종료": "⚫"}
    num = lambda s: int(s.split("_")[1])
    names = sorted(A, key=num)
    default = names.index("equipment_12") if "equipment_12" in names else 0
    eq = st.radio("설비", names, index=default, horizontal=True, label_visibility="collapsed",
                  format_func=lambda e: f"{icon[A[e]['status']]} EQ {num(e)}")
    a = A[eq]
    detail = [f"관리 한계 이탈 {a['out']}{'개월' if is_p else '회'}" if a["out"] else None,
              f"중심선 한쪽 {a['run']}주 연속" if (not is_p and a["run"] >= L.RUN_LEN) else None]
    detail = " · ".join(d for d in detail if d)
    dcol = RED if a["out"] else YELLOW
    st.markdown(f'### {eq} <span style="font-size:15px;color:{MUTED};font-weight:400">{SPC[eq]["machine"]}</span> '
                f'<span style="font-size:15px;color:{dcol}">{detail}</span>', unsafe_allow_html=True)

    left, right = st.columns([4, 1])
    with left:
        fig = spc_p_fig(eq, a) if is_p else spc_i_fig(eq, a, key)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        legend = [("●", BLUE, "정상"), ("●", "#ef4444", "3σ 초과 (이탈)"), ("▲", TEXT, "Target 교체 " + ("월" if is_p else "주"))]
        if not is_p:
            legend.append(("■", YELLOW, "치우침 구간 (8주 이상 연속)"))
        st.markdown(" &nbsp; ".join(f'<span class="small"><span style="color:{c}">{s}</span> {t}</span>'
                                    for s, c, t in legend), unsafe_allow_html=True)
    with right:
        if is_p:
            rows = [r for i, r in enumerate(PM[eq]) if r]
            us = [L.p_ucl(PBAR, r["n"]) * 100 for r in rows]
            mon = [r for i, r in enumerate(PM[eq]) if r and i >= L.P_PHASE1_MONTHS]
            nw, nb = sum(r["n"] for r in mon), sum(r["bad"] for r in mon)
            card("중심선 (CL, 전체 설비)", f"{PBAR * 100:.2f}%")
            card("관리 한계 (UCL, 웨이퍼 수별)", f"{min(us):.1f} - {max(us):.1f}%")
            card("모니터링 구간 불량률", f"{nb / nw * 100:.1f}%" if nw else "-", f"{nb} / {nw}장" if nw else "")
        else:
            unit = "" if key == "t" else " (×10⁻⁴)"
            card("중심선 (CL)" + unit, f"{a['cl']:.2f}")
            card("공정 표준편차 σ", f"{a['sigma']:.3f}")
            card("관리 한계 (LCL - UCL)", f"{a['lcl']:.1f} - {a['ucl']:.1f}")


def week_x(w):
    return L.SPC_START + pd.Timedelta(weeks=w)


def spc_i_fig(eq, a, key):
    o = SPC[eq]
    vals = o[key]
    weeks = [i for i, v in enumerate(vals) if v is not None]
    x = [week_x(i) for i in weeks]
    y = [vals[i] for i in weeks]
    z = [(v - a["cl"]) / a["sigma"] for v in y]
    dec = 2 if key == "t" else 3
    hover = []
    for i, v, zz in zip(weeks, y, z):
        inf = o["info"][i]
        st_ = "이탈" if abs(zz) > 3 else "정상"
        lo, hi = inf[key + "_min"], inf[key + "_max"]
        sc = "#f87171" if abs(zz) > 3 else "#5eead4"
        s = (f"<b>{week_x(i):%Y.%m.%d} - {week_x(i) + pd.Timedelta(days=6):%m.%d}</b><br>"
             f"웨이퍼 {inf['n']}장 · 주간 평균 {v:.{dec}f}<br>"
             f"<span style='color:#c9d4df'>웨이퍼 간 표준편차 {inf[key + '_sd']:.{dec}f}</span><br>"
             f"<span style='color:{sc}'><b>CL 대비 {zz:+.2f}σ · {st_}</b></span><br>"
             f"<span style='color:#c9d4df'>최저 {lo[0]} ({lo[1]:.{dec}f})<br>최고 {hi[0]} ({hi[1]:.{dec}f})</span>")
        for s0, e0 in a["runs"]:
            if s0 <= i <= e0:
                n_run = sum(1 for w in weeks if s0 <= w <= e0)
                side = "위" if v > a["cl"] else "아래"
                s += f"<br><span style='color:#fbbf24'><b>치우침 구간 · 중심선 {side}쪽 {n_run}주 연속</b></span>"
        if i in o["tc"]:
            s += "<br><b>Target 교체 포함 주</b>"
        hover.append(s)
    fig = go.Figure()
    x0, x1 = week_x(0) - pd.Timedelta(days=3), week_x(L.SPC_WEEKS) - pd.Timedelta(days=4)
    fig.add_vrect(x0=week_x(L.PHASE1_WEEKS) - pd.Timedelta(days=3), x1=x1, fillcolor="#0f1620", line_width=0, layer="below")
    fig.add_hrect(y0=a["cl"] - 2 * a["sigma"], y1=a["cl"] + 2 * a["sigma"], fillcolor="#16324a", opacity=0.35, line_width=0, layer="below")
    for s0, e0 in a["runs"]:
        fig.add_vrect(x0=week_x(s0) - pd.Timedelta(days=3), x1=week_x(e0) + pd.Timedelta(days=3),
                      fillcolor=YELLOW, opacity=0.10, line_width=0, layer="below")
    for v, c, lab, dash in [(a["ucl"], RED, "UCL", "dash"), (a["cl"], TEAL, "CL", "solid"), (a["lcl"], RED, "LCL", "dash")]:
        fig.add_hline(y=v, line_color=c, line_dash=dash, line_width=1.5,
                      annotation_text=f"{lab} {v:.{1 if key == 't' else 2}f}", annotation_position="right",
                      annotation_font=dict(color=c, size=11))
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers", line=dict(color="#7aa2c8", width=1.5),
                             marker=dict(size=[11 if abs(v) > 3 else 7 for v in z],
                                         color=["#ef4444" if abs(v) > 3 else BLUE for v in z],
                                         line=dict(color=BG, width=1)),
                             hovertext=hover, hoverinfo="text"))
    ymin = min(min(y), a["lcl"] - a["sigma"])
    ymax = max(max(y), a["ucl"] + a["sigma"])
    fig.add_trace(go.Scatter(x=[week_x(i) for i in o["tc"]], y=[ymin] * len(o["tc"]), mode="markers",
                             marker=dict(symbol="triangle-up", size=10, color=TEXT, line=dict(color=BG, width=1)),
                             hovertext=["Target 교체"] * len(o["tc"]), hoverinfo="text"))
    fig.add_annotation(x=week_x(1), y=ymax, text="관리 한계 산출 구간", showarrow=False, xanchor="left",
                       font=dict(color=MUTED, size=11))
    fig.add_annotation(x=week_x(L.PHASE1_WEEKS + 1), y=ymax, text="모니터링 구간", showarrow=False, xanchor="left",
                       font=dict(color=MUTED, size=11))
    fig.update_xaxes(range=[x0, x1], tickformat="%y.%m", dtick="M3")
    fig.update_yaxes(range=[ymin - a["sigma"] * 0.3, ymax + a["sigma"] * 0.5])
    return base_layout(fig, 380, ytitle="Thickness" if key == "t" else "Rs (×10⁻⁴)")


def spc_p_fig(eq, a):
    rows = PM[eq]
    months = [m.to_timestamp() for m in L.P_MONTHS]
    xs, ys, ucls, hover, colors, tcx = [], [], [], [], [], []
    for m, r in zip(months, rows):
        if not r:
            continue
        p, u = r["bad"] / r["n"] * 100, L.p_ucl(PBAR, r["n"]) * 100
        xs.append(m); ys.append(p); ucls.append(u)
        colors.append("#ef4444" if p > u else BLUE)
        stt = "이탈 (UCL 초과)" if p > u else "불량 발생 · 한계 이내" if r["bad"] else "불량 없음"
        sc = "#f87171" if p > u else "#fbbf24" if r["bad"] else "#5eead4"
        s = (f"<b>{m:%Y.%m}</b><br>웨이퍼 {r['n']}장 · 불량 {r['bad']}장<br>"
             f"<span style='color:#c9d4df'>불량률 {p:.1f}% · UCL {u:.1f}%</span><br>"
             f"<span style='color:{sc}'><b>{stt}</b></span>")
        if r["bad"]:
            s += f"<br><span style='color:#c9d4df'>불량 중 Target Life 100 이하 {r['bad_early']}장</span>"
        if r["ids"]:
            s += "<br><span style='color:#c9d4df'>불량 웨이퍼 " + ", ".join(r["ids"]) + (" 외" if r["bad"] > 3 else "") + "</span>"
        if r["tc"]:
            s += "<br><b>Target 교체 포함 월</b>"
            tcx.append(m)
        hover.append(s)
    fig = go.Figure()
    fig.add_vrect(x0=months[L.P_PHASE1_MONTHS] - pd.Timedelta(days=15), x1=months[-1] + pd.Timedelta(days=15),
                  fillcolor="#0f1620", line_width=0, layer="below")
    fig.add_trace(go.Scatter(x=xs, y=ucls, mode="lines", line=dict(color=RED, dash="dash", width=1.5, shape="hvh"),
                             hoverinfo="skip"))
    fig.add_hline(y=PBAR * 100, line_color=TEAL, line_width=1.5, annotation_text=f"CL {PBAR * 100:.2f}%",
                  annotation_position="right", annotation_font=dict(color=TEAL, size=11))
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", line=dict(color="#7aa2c8", width=1.5),
                             marker=dict(size=9, color=colors, line=dict(color=BG, width=1)),
                             hovertext=hover, hoverinfo="text"))
    fig.add_trace(go.Scatter(x=tcx, y=[0] * len(tcx), mode="markers", hoverinfo="skip",
                             marker=dict(symbol="triangle-up", size=10, color=TEXT)))
    fig.update_xaxes(tickformat="%y.%m", dtick="M3")
    return base_layout(fig, 380, ytitle="불량률 (%)")


# ================================================================ 공통: 위치별 그래프, Wafer Map
def judge_text(v, lsl, usl, dec, kind):
    if kind == "rho":
        g = L.rho_grade(v, lsl)
        return g, L.GRADE_COLOR[g]
    if v < lsl:
        return "이탈 (LSL 미만)", RED
    if usl is not None and v > usl:
        return "이탈 (USL 초과)", RED
    return "정상", TEAL


def point_fig(pred, act, lsl, usl, ytitle, dec, bad, kind="t", pred_name="예측", act_name="실측",
              act_color=TEXT, diff_name="차이"):
    xs = list(range(1, 18))
    fig = go.Figure()
    for v, lab in [(lsl, "LSL"), (usl, "USL")]:
        if v is not None:
            fig.add_hline(y=v, line_color="#ef4444", line_dash="dash", line_width=1.5,
                          annotation_text=f"{lab} {v:.{dec}f}", annotation_position="right",
                          annotation_font=dict(color=RED, size=11))
    hover = []
    for k, (p, a) in enumerate(zip(pred, act)):
        jt, jc = judge_text(p, lsl, usl, dec, kind)
        hover.append(f"<b>위치 {k + 1}</b><br><span style='color:{ACCENT}'><b>{pred_name} {p:.{dec}f}</b></span><br>"
                     f"{act_name} {a:.{dec}f}<br>{diff_name} {p - a:+.{dec}f}<br>"
                     f"<span style='color:{jc}'><b>판정: {jt}</b></span>")
    fig.add_trace(go.Scatter(x=xs, y=act, mode="lines+markers", name=act_name, hoverinfo="skip",
                             line=dict(color=act_color, dash="dash", width=1.6),
                             marker=dict(size=7, color=BG, line=dict(color=act_color, width=1.5))))
    fig.add_trace(go.Scatter(x=xs, y=pred, mode="lines+markers", name=pred_name,
                             line=dict(color=ACCENT, width=2.5),
                             marker=dict(size=9, color=[RED if b else ACCENT for b in bad], line=dict(color=BG, width=1.5)),
                             hovertext=hover, hoverinfo="text"))
    fig.update_xaxes(tickmode="array", tickvals=xs)
    return base_layout(fig, 300, ytitle=ytitle, xtitle="측정 위치 (Point No.)")


def chart_head(title, items):
    """그래프 제목(왼쪽) + 범례(오른쪽) 한 줄."""
    leg = " &nbsp;&nbsp; ".join(f'<span style="color:{c}">{sym}</span> {t}' for sym, c, t in items)
    st.markdown(f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px">'
                f'<span style="font-size:17px;font-weight:700">{title}</span>'
                f'<span class="small" style="color:{TEXT}">{leg}</span></div>', unsafe_allow_html=True)


def wafer_fig(values, kind, lsl, usl=None):
    img = L.wafer_rgb(values, kind, lsl, usl)
    n = img.shape[0]
    pos = L.point_positions()
    fig = go.Figure(go.Image(z=img, x0=-1, dx=2 / (n - 1), y0=1, dy=-2 / (n - 1), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=pos[:, 0], y=pos[:, 1], mode="markers+text", text=[str(i) for i in range(1, 18)],
                             textposition="top right", textfont=dict(color="white", size=11),
                             marker=dict(size=8, color="white", line=dict(color=BG, width=1.5)),
                             hovertext=[f"위치 {i + 1}: {v:.4f}" if kind == "rho" else f"위치 {i + 1}: {v:.2f}"
                                        for i, v in enumerate(values)], hoverinfo="text"))
    fig.add_shape(type="circle", x0=-1, y0=-1, x1=1, y1=1, line=dict(color="#3a4656", width=2))
    fig.update_xaxes(visible=False, range=[-1.08, 1.08])
    fig.update_yaxes(visible=False, range=[-1.08, 1.08], scaleanchor="x", autorange=False)
    fig.update_layout(height=330, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor=PANEL, plot_bgcolor=PANEL,
                      showlegend=False)
    return fig


def map_legend(kind):
    sw = lambda c: f'<span style="display:inline-block;width:12px;height:12px;border-radius:3px;background:{c};margin-right:6px"></span>'
    items = ([("#ef4444", "LSL 미만 (이탈)"), ("#f97316", f"LSL + {L.RISK_M} 이내 (위험)"),
              ("#fbbf24", f"LSL + {L.WARN_M} 이내 (주의)"), ("linear-gradient(90deg,#0b2a5b,#bfdbfe)", "정상")]
             if kind == "rho" else
             [("#ef4444", "LSL 미만"), ("#b91c1c", "USL 초과"), ("linear-gradient(90deg,#0b2a5b,#bfdbfe)", "정상 (진할수록 낮음)")])
    st.markdown("<br>".join(f'<span class="small">{sw(c)}{t}</span>' for c, t in items), unsafe_allow_html=True)


# ================================================================ 2. Thickness · Rs 예측
def page_pred():
    st.markdown("# Thickness · Rs · Resistivity 예측")
    st.caption("웨이퍼별 예측 결과와 Resistivity 위험 등급")
    i = wafer_selector("pred")
    m, row = META.loc[i], X.loc[[i]]
    pt, pr = L.predict(P["bt"], row)[0], L.predict(P["br"], row)[0]
    at, ar, ap = P["Yt"].loc[i].values, P["Yr"].loc[i].values, P["Yp"].loc[i].values
    pp = pt * pr * K

    lsl_t, usl_t = L.spec_limits(P["fleet_t"])
    lsl_r, usl_r = L.spec_limits(P["fleet_r"])
    bad_t = (pt < lsl_t) | (pt > usl_t)
    bad_r = (pr < lsl_r) | (pr > usl_r)
    grades = [L.rho_grade(v, LSL_RHO) for v in pp]
    rho_min = pp.min()
    grade = L.rho_grade(rho_min, LSL_RHO)
    n_grade = grades.count(grade)

    st.write("")
    left, right = st.columns([1, 3])
    with left:
        st.markdown("#### 공정 조건")
        conds = [("머신", m.machine), ("설비", m.equipment), ("공정 시작", f"{m.runstart:%Y-%m-%d %H:%M}"),
                 ("Target Life", f"{row.LIFE_TARGET_RUN_A.iloc[0]:.2f}"),
                 ("Deposition Time (T<sub>d</sub>)", f"{row.T_DEP1_PROCESS.iloc[0]:.2f}"),
                 ("Target Voltage", f"{row.VOLTS_TARGET_DEP1_A.iloc[0]:.1f}"),
                 ("Target Current", f"{row.AMPS_TARGET_DEP1_A.iloc[0]:.1f}")]
        st.markdown("".join(f'<div class="kv"><span>{k}</span><span>{v}</span></div>' for k, v in conds),
                    unsafe_allow_html=True)
        st.write("")
        crit = [("이탈 예상", f"{LSL_RHO:.4f} 미만"), ("위험", f"{LSL_RHO:.4f} - {LSL_RHO + L.RISK_M:.4f}"),
                ("주의", f"{LSL_RHO + L.RISK_M:.4f} - {LSL_RHO + L.WARN_M:.4f}"), ("정상", f"{LSL_RHO + L.WARN_M:.4f} 초과")]
        crit_html = "".join(f'<div style="display:flex;gap:8px;font-size:12px"><span style="color:{L.GRADE_COLOR[g]};'
                            f'font-weight:700;width:60px">● {g}</span><span>{t}</span></div>' for g, t in crit)
        st.markdown(f"""<div class="box" style="background:#1a1416;border-color:#5c1d1d">
            <div class="small" style="color:#fecaca">Resistivity 이탈 위험</div>
            <div style="font-size:26px;font-weight:700;color:{L.GRADE_COLOR[grade]}">{grade}</div>
            <div style="font-size:13px">LSL Margin <b style="color:{L.GRADE_COLOR[grade]}">{rho_min - LSL_RHO:+.4f}</b></div>
            <div style="border-top:1px solid #3a2224;margin-top:10px;padding-top:8px">
            <div class="small" style="margin-bottom:4px">판정 기준 · Min Resistivity</div>{crit_html}</div></div>""",
                    unsafe_allow_html=True)

    with right:
        c1, c2 = st.columns([3, 2])
        labels = {"Thickness": f"Thickness (이탈 : {bad_t.sum()})", "Rs": f"Rs (이탈 : {bad_r.sum()})",
                  "Resistivity": f"Resistivity ({grade}" + ("" if grade == "정상" else f" : {n_grade}") + ")"}
        metric = c1.radio("지표", list(labels), horizontal=True, format_func=labels.get, label_visibility="collapsed")
        view = c2.radio("Wafer Map", ["예측", "실측"], horizontal=True)
        if metric == "Thickness":
            fig = point_fig(pt, at, lsl_t, usl_t, "Thickness", 2, bad_t, "t")
            vals, kind, lo, hi = (pt if view == "예측" else at), "t", lsl_t, usl_t
        elif metric == "Rs":
            fig = point_fig(pr, ar, lsl_r, usl_r, "Rs (×10⁻⁴)", 3, bad_r, "r")
            vals, kind, lo, hi = (pr if view == "예측" else ar), "r", lsl_r, usl_r
        else:
            fig = point_fig(pp, ap, LSL_RHO, None, "Resistivity", 4, [g in ("이탈 예상", "위험") for g in grades], "rho")
            vals, kind, lo, hi = (pp if view == "예측" else ap), "rho", LSL_RHO, None
        chart_head(f"측정 위치별 {metric}", [("╌○╌", TEXT, "실측"), ("━●━", ACCENT, "예측"), ("●", RED, "기준 이탈 지점")])
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.write("")
        m1, m2 = st.columns([3, 2])
        with m1:
            st.plotly_chart(wafer_fig(vals, kind, lo, hi), use_container_width=True, config={"displayModeBar": False})
        with m2:
            st.markdown(f"**Wafer Map · {view} {metric}**")
            map_legend(kind)


# ================================================================ 3. What-if
def apply_rec(td):
    st.session_state.wi_td = td


def page_whatif():
    st.markdown("# What-if 시뮬레이션")
    st.caption("Parameter 변경에 따른 예상 Thickness")
    i = wafer_selector("wi")
    row = X.loc[i]
    life0, td0 = float(row.LIFE_TARGET_RUN_A), float(row.T_DEP1_PROCESS)
    if st.session_state.get("wi_wafer") != i:          # 웨이퍼가 바뀌면 슬라이더를 그 웨이퍼 조건으로
        st.session_state.wi_wafer = i
        st.session_state.wi_life = round(min(max(life0, 0.0), 475.0), 1)
        st.session_state.wi_td = round(min(max(td0, 15.5), 21.3), 2)

    st.write("")
    left, right = st.columns([1, 3])
    with left:
        st.markdown("#### Parameter 변경")
        st.markdown(f'<div class="box"><div class="kv" style="border:0"><span>기준 Target Life</span><span>{life0:.2f}</span></div>'
                    f'<div class="kv" style="border:0"><span>기준 T<sub>d</sub></span><span>{td0:.2f}</span></div></div>',
                    unsafe_allow_html=True)
        life = st.slider("Target Life", 0.0, 475.0, step=1.0, key="wi_life")
        td = st.slider("Deposition Time ($T_d$)", 15.5, 21.3, step=0.05, key="wi_td")

        rec = L.recommend(P["bt"], row, life, td, td0, P["fleet_t"])
        rows_html = "".join(f'<div class="kv" style="border:0;padding:3px 0"><span>{td_html(k)}</span><span>{v}</span></div>'
                            for k, v in rec["rows"])
        st.markdown(f"""<div class="box"><div style="font-size:17px;font-weight:700;color:{rec['color']}">{td_html(rec['title'])}</div>
            <div class="small">권장 공정 조건</div>
            <div style="border-top:1px solid {BORDER};margin-top:8px;padding-top:6px">{rows_html}</div></div>""",
                    unsafe_allow_html=True)
        if rec["can_apply"]:
            st.button("권장 조건 적용", on_click=apply_rec, args=(rec["rec_td"],), use_container_width=True)

        def reset():
            st.session_state.wi_life = round(min(max(life0, 0.0), 475.0), 1)
            st.session_state.wi_td = round(min(max(td0, 15.5), 21.3), 2)
        st.button("이 웨이퍼 조건으로 되돌리기", on_click=reset, use_container_width=True)
        st.markdown(f'<div class="small" style="margin-top:8px">Target Life 구간<br>· 교체 직후 ≤ {L.LIFE_NEW}<br>'
                    f'· 교체 준비 ≥ {L.LIFE_PREP}<br>· 교체 권장 ≥ {L.LIFE_END}</div>', unsafe_allow_html=True)

    base_row, new_row = X.loc[[i]], pd.DataFrame([L.set_params(row, life, td)])
    bt, nt = L.predict(P["bt"], base_row)[0], L.predict(P["bt"], new_row)[0]
    br, nr = L.predict(P["br"], base_row)[0], L.predict(P["br"], new_row)[0]
    lsl, usl = L.spec_limits(P["fleet_t"])
    bad_b, bad_n = (bt < lsl) | (bt > usl), (nt < lsl) | (nt > usl)
    pts = lambda b: " ".join(f"#{k + 1}" for k in np.nonzero(b)[0]) or "없음"

    with right:
        c1, c2, c3 = st.columns(3)
        with c1:
            card("평균 Thickness (기준 → 변경)", f'{bt.mean():.2f} → <span style="color:{ACCENT}">{nt.mean():.2f}</span>',
                 f"변화량 {nt.mean() - bt.mean():+.2f}")
        with c2:
            card("평균 Rs ×10⁻⁴ (기준 → 변경)", f'{br.mean():.3f} → <span style="color:{ORANGE}">{nr.mean():.3f}</span>',
                 f"변화량 {nr.mean() - br.mean():+.3f}")
        with c3:
            nc = RED if bad_n.sum() else TEAL
            card("기준 이탈 지점 (기준 → 변경)", f'{bad_b.sum()} → <span style="color:{nc}">{bad_n.sum()}</span> '
                 f'<span style="font-size:14px;color:{MUTED}">/ 17</span>',
                 f'이탈 위치 {pts(bad_b)} → <span style="color:{nc}">{pts(bad_n)}</span>')
        st.write("")
        chart_head("측정 위치별 예상 Thickness", [("╌○╌", MUTED, "기준"), ("━●━", ACCENT, "변경 후"), ("●", RED, "기준 이탈 지점")])
        fig = point_fig(nt, bt, lsl, usl, "Thickness", 2, bad_n, "t", pred_name="변경 후", act_name="기준",
                        act_color=MUTED, diff_name="변화")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown('<div class="small" style="margin-top:10px">모델의 예측이며 실제 레시피 변경 효과를 확정하지 않습니다.</div>',
                    unsafe_allow_html=True)
        if P["curves"]:
            with st.expander("참고: SHAP 영향 곡선 (평가 웨이퍼 평균 효과)"):
                e1, e2 = st.columns(2)
                for col, f, lab in [(e1, "LIFE_TARGET_RUN_A", "Target Life"), (e2, "T_DEP1_PROCESS", "Deposition Time (T_d)")]:
                    c = P["curves"][f]
                    g = go.Figure(go.Scatter(x=c["x"], y=c["thk"], mode="lines+markers", line=dict(color=ACCENT)))
                    g.add_hline(y=0, line_color=MUTED, line_width=1)
                    col.plotly_chart(base_layout(g, 240, ytitle="Thickness 영향", xtitle=lab), use_container_width=True,
                                     config={"displayModeBar": False})


# ---------------------------------------------------------------- 라우팅
{"설비 SPC 모니터링": page_spc, "Thickness · Rs 예측": page_pred, "What-if 시뮬레이션": page_whatif}[page]()
