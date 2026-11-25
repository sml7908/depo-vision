"""Depo-Vision 대시보드 계산 함수 (Streamlit, Plotly 없이 동작)."""
import json
import os

import joblib
import numpy as np
import pandas as pd

# ---------------------------------------------------------------- 경로
BASE = os.environ.get("DEPOGUARD_BASE", ".")
RAW_PATH = os.path.join(BASE, "data/Version2/apc_spc_AlCu_inputs_outputs.csv")
DATA_DIR = os.path.join(BASE, "data/processed_rev")
MODEL_DIR = os.path.join(BASE, "models")

# ---------------------------------------------------------------- 기준값
SPC_START = pd.Timestamp("2023-07-03")   # SPC 주차 0
SPC_WEEKS = 100
PHASE1_WEEKS = 52                         # 관리 한계 산출 구간 (2023.07.03 - 2024.06.30)
P_MONTHS = pd.period_range("2023-07", "2025-05", freq="M")
P_PHASE1_MONTHS = 12
RUN_LEN = 8                               # 치우침: 중심선 한쪽 8주 연속
TOL = 0.04                                # Thickness, Rs 규격: 학습 평균 ±4%
RISK_M, WARN_M = 0.003, 0.005             # Resistivity 위험 / 주의 여유
LIFE_NEW, LIFE_PREP, LIFE_END = 10, 400, 430
FLEET_TOL_TD = 1.0                        # 기준 Thickness와 이 이상 차이 나면 T_d 조정 권장


# ---------------------------------------------------------------- 데이터 불러오기
def load_raw(path=RAW_PATH):
    """원본 CSV (첫 줄 제목, 구분자 자동 확인)."""
    with open(path, encoding="utf-8-sig") as f:
        head = [f.readline() for _ in range(5)]
    hi = next(i for i, line in enumerate(head) if "wafer_id" in line)
    sep = ";" if head[hi].count(";") > head[hi].count(",") else ","
    df = pd.read_csv(path, sep=sep, skiprows=hi, encoding="utf-8-sig", low_memory=False)
    df.columns = df.columns.str.strip()
    df["runstart"] = pd.to_datetime(df["runstart"])
    df["runend"] = pd.to_datetime(df["runend"])
    df["rs"] = df["mean_resistance"] * 1e4
    return df


def load_processed(data_dir=DATA_DIR, model_dir=MODEL_DIR):
    L = lambda n: pd.read_csv(os.path.join(data_dir, n))
    out = {
        "X_test": L("X_test.csv"),
        "Yt": L("Y_thk_test.csv"), "Yr": L("Y_rs_test.csv"), "Yp": L("Y_rho_test.csv"),
        "Yt_train": L("Y_thk_train.csv"), "Yr_train": L("Y_rs_train.csv"),
        "meta": L("meta_test.csv"),
        "bt": joblib.load(os.path.join(model_dir, "thk_17pt_rev.pkl")),
        "br": joblib.load(os.path.join(model_dir, "rs_17pt_rev.pkl")),
    }
    with open(os.path.join(data_dir, "constants.json"), encoding="utf-8") as f:
        out["C"] = json.load(f)
    curves = os.path.join(data_dir, "shap_whatif_curves.json")
    out["curves"] = json.load(open(curves)) if os.path.exists(curves) else None
    out["meta"]["runstart"] = pd.to_datetime(out["meta"]["runstart"])
    out["X_test"] = out["X_test"][out["bt"]["features"]]
    out["fleet_t"] = float(out["Yt_train"].values.mean())
    out["fleet_r"] = float(out["Yr_train"].values.mean())
    return out


# ---------------------------------------------------------------- 예측
def predict(bundle, X):
    """17개 위치 예측 (원래 단위). X: DataFrame (행 여러 개 가능)."""
    X = X[bundle["features"]]
    return bundle["y_scaler"].inverse_transform(bundle["model"].predict(X))


def set_params(row, life, td):
    """What-if: Target Life, T_d를 바꾼 입력 행 (파생 변수도 다시 계산)."""
    r = row.copy()
    r["LIFE_TARGET_RUN_A"] = life
    r["T_DEP1_PROCESS"] = td
    if "is_new_target" in r.index:
        r["is_new_target"] = int(life <= 10)
    if "is_early_target" in r.index:
        r["is_early_target"] = int(life <= 100)
    return r


def spec_limits(fleet):
    return fleet * (1 - TOL), fleet * (1 + TOL)


def rho_grade(v, lsl):
    m = v - lsl
    return "이탈 예상" if m < 0 else "위험" if m <= RISK_M else "주의" if m <= WARN_M else "정상"


GRADE_COLOR = {"이탈 예상": "#f87171", "위험": "#fbbf24", "주의": "#fde68a", "정상": "#5eead4"}


# ---------------------------------------------------------------- SPC (I 관리도)
def spc_weekly(df):
    """설비별 주간 평균 Thickness, Rs와 주별 상세 정보."""
    d = df[(df.runstart >= SPC_START) & (df.runstart < SPC_START + pd.Timedelta(weeks=SPC_WEEKS))].copy()
    d["wk"] = (d.runstart - SPC_START).dt.days // 7
    out = {}
    for eq, g in d.groupby("equipment"):
        o = {"machine": g.machine.iloc[0], "t": [None] * SPC_WEEKS, "r": [None] * SPC_WEEKS,
             "info": [None] * SPC_WEEKS, "tc": []}
        for w, gw in g.groupby("wk"):
            w = int(w)
            o["t"][w] = gw.mean_thickness.mean()
            o["r"][w] = gw.rs.mean()
            info = {"n": len(gw)}
            for key, col in [("t", "mean_thickness"), ("r", "rs")]:
                info[key + "_sd"] = gw[col].std() if len(gw) > 1 else 0.0
                lo, hi = gw.loc[gw[col].idxmin()], gw.loc[gw[col].idxmax()]
                info[key + "_min"] = (lo.wafer_id, lo[col])
                info[key + "_max"] = (hi.wafer_id, hi[col])
            o["info"][w] = info
            if (gw.LIFE_TARGET_RUN_A <= LIFE_NEW).any():
                o["tc"].append(w)
        out[eq] = o
    return out


def analyze_i(values):
    """I 관리도 판정: CL, σ(MR/1.128), 3σ 이탈 수, 치우침 구간."""
    base = [x for x in values[:PHASE1_WEEKS] if x is not None]
    if len(base) < 5:
        return None
    cl = float(np.mean(base))
    sigma = float(np.mean(np.abs(np.diff(base)))) / 1.128
    mon = [(i, x) for i, x in enumerate(values) if i >= PHASE1_WEEKS and x is not None]
    out = sum(abs(x - cl) > 3 * sigma for _, x in mon)
    runs, run, prev, start, best = [], 0, 0, None, 0
    for k, (i, x) in enumerate(mon):
        sd = 1 if x > cl else -1
        if sd == prev:
            run += 1
        else:
            if run >= RUN_LEN:
                runs.append((start, mon[k - 1][0]))
            run, start = 1, i
        prev, best = sd, max(best, run)
    if run >= RUN_LEN:
        runs.append((start, mon[-1][0]))
    status = "가동 종료" if not mon else "이탈" if out else "치우침" if best >= RUN_LEN else "정상"
    return {"cl": cl, "sigma": sigma, "ucl": cl + 3 * sigma, "lcl": cl - 3 * sigma,
            "out": out, "run": best, "runs": runs, "status": status}


# ---------------------------------------------------------------- SPC (p 관리도)
def p_monthly(df):
    d = df[(df.runstart >= P_MONTHS[0].start_time) & (df.runstart <= P_MONTHS[-1].end_time)].copy()
    d["mo"] = d.runstart.dt.to_period("M")
    d["bad"] = (d.quality_resistivity == "bad").astype(int)
    ph = d[d.mo <= P_MONTHS[P_PHASE1_MONTHS - 1]]
    pbar = ph.bad.mean()
    out = {}
    for eq, g in d.groupby("equipment"):
        rows = []
        for m in P_MONTHS:
            gm = g[g.mo == m]
            if gm.empty:
                rows.append(None)
                continue
            bads = gm[gm.bad == 1]
            rows.append({"n": len(gm), "bad": int(gm.bad.sum()), "tc": bool((gm.LIFE_TARGET_RUN_A <= LIFE_NEW).any()),
                         "bad_early": int((bads.LIFE_TARGET_RUN_A <= 100).sum()), "ids": list(bads.wafer_id[:3])})
        out[eq] = rows
    return out, float(pbar)


def p_ucl(pbar, n):
    return pbar + 3 * np.sqrt(pbar * (1 - pbar) / n)


def analyze_p(rows, pbar):
    mon = [(i, r) for i, r in enumerate(rows) if i >= P_PHASE1_MONTHS and r]
    out = sum(r["bad"] / r["n"] > p_ucl(pbar, r["n"]) for _, r in mon)
    status = "가동 종료" if not mon else "이탈" if out else "정상"
    return {"cl": pbar, "out": out, "status": status, "run": 0, "runs": []}


# ---------------------------------------------------------------- Wafer Map
def point_positions():
    """측정 위치 좌표 (가정: 1 중심, 2-9 안쪽 원, 10-17 바깥 원), 반지름 1 기준."""
    pos = [(0.0, 0.0)]
    for ring, n0 in [(95 / 205, 0), (165 / 205, 8)]:
        for k in range(8):
            a = np.deg2rad(-90 + k * 45)
            pos.append((ring * np.cos(a), -ring * np.sin(a)))
    return np.array(pos)


BLUES = ["#0b2a5b", "#103a7a", "#1d4ed8", "#2563eb", "#3b82f6", "#60a5fa", "#93c5fd", "#bfdbfe"]


def _hex(c):
    c = c.lstrip("#")
    return [int(c[i:i + 2], 16) for i in (0, 2, 4)]


def wafer_rgb(values, kind, lsl, usl=None, n=161):
    """IDW 보간 Wafer Map을 RGB 배열로 (원 밖은 배경색)."""
    P = point_positions()
    xs = np.linspace(-1, 1, n)
    gx, gy = np.meshgrid(xs, xs[::-1])
    d2 = (gx[..., None] - P[:, 0]) ** 2 + (gy[..., None] - P[:, 1]) ** 2
    w = 1 / np.maximum(d2, 1e-6) ** 1
    z = (w * np.asarray(values)).sum(-1) / w.sum(-1)
    img = np.zeros((n, n, 3), dtype=np.uint8)
    img[:] = _hex("#121821")
    inside = gx ** 2 + gy ** 2 <= 1
    for iy, ix in zip(*np.nonzero(inside)):
        v = z[iy, ix]
        if kind == "rho":
            m = v - lsl
            col = "#ef4444" if m < 0 else "#f97316" if m <= RISK_M else "#fbbf24" if m <= WARN_M else \
                BLUES[min(7, max(0, int((m - WARN_M) / 0.0014)))]
        else:
            col = "#ef4444" if v < lsl else "#b91c1c" if v > usl else BLUES[min(7, int((v - lsl) / (usl - lsl) * 8))]
        img[iy, ix] = _hex(col)
    return img


# ---------------------------------------------------------------- What-if 권장
def recommend(bt, row, life, td, td0, fleet):
    """Target Life 상태와 Thickness로 T_d 조정 / Target 교체 권장."""
    grid = np.round(np.arange(16.0, 21.1001, 0.05), 2)
    cand = pd.DataFrame([set_params(row, life, t) for t in grid])
    means = predict(bt, cand).mean(axis=1)
    now = float(predict(bt, pd.DataFrame([set_params(row, life, td)])).mean())
    stable = float(predict(bt, pd.DataFrame([set_params(row, 50, td)])).mean())
    loss = now - stable                               # Target 상태로 줄어든 Thickness
    state = ("교체 직후" if life <= LIFE_NEW else "수명 말기" if life >= LIFE_END
             else "교체 준비 구간" if life >= LIFE_PREP else "안정 구간")
    if life <= LIFE_NEW or life >= LIFE_PREP:
        target = now - loss                           # Target 상태 손실만 보정
    else:
        target = fleet
    k = int(np.argmin(np.abs(means - target)))
    rec_td, rec_mean = float(grid[k]), float(means[k])
    applied = abs(rec_td - td) < 0.05
    if life <= LIFE_NEW and applied:
        title, color, rows = "보정 반영됨", "#5eead4", [("Target", state)]
    elif life <= LIFE_NEW:
        title, color = "T_d 조정 권장", "#fb923c"
        rows = [("Target", state), ("T_d", f"{td:.2f} → {rec_td:.2f}"), ("예상 Thickness", f"{rec_mean:.2f}")]
    elif life >= LIFE_END:
        title, color, rows = "Target 교체 권장", "#f87171", [("Target", state), ("보정 T_d", f"{rec_td:.2f}")]
    elif life >= LIFE_PREP:
        title, color, rows = "Target 교체 준비", "#fbbf24", [("Target", state), ("보정 T_d", f"{rec_td:.2f}")]
    elif abs(now - fleet) > FLEET_TOL_TD and not applied:
        title, color = "T_d 조정 권장", "#fb923c"
        rows = [("Target", state), ("T_d", f"{td:.2f} → {rec_td:.2f}"), ("예상 Thickness", f"{rec_mean:.2f}")]
    else:
        title, color, rows = "현재 조건 유지", "#5eead4", [("Target", state), ("T_d", f"{td:.2f} 유지")]
    can_apply = title not in ("현재 조건 유지", "보정 반영됨") and not applied
    return {"title": title, "color": color, "rows": rows, "rec_td": rec_td, "can_apply": can_apply,
            "loss": loss, "now": now}
