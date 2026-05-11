"""
東京不動産価格推定 Streamlit アプリ
マンション / 戸建て を選択して価格推定・感度分析・シナリオ比較
"""
import os, sys, tempfile, shutil, warnings, math
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
import requests
import streamlit as st
import plotly.graph_objects as go

# ── パス設定（app.py からの相対パス → Streamlit Cloud でも動作）──
_APP_DIR   = os.path.dirname(os.path.abspath(__file__))
_MODEL_DIR = os.path.join(_APP_DIR, "models")

PATHS = {
    "mansion": {
        "artifacts": os.path.join(_MODEL_DIR, "artifacts.joblib"),
        "lgb":       os.path.join(_MODEL_DIR, "lgb_model_native.txt"),
    },
    "tochi": {
        "artifacts": os.path.join(_MODEL_DIR, "artifacts_tochi.joblib"),
        "lgb":       os.path.join(_MODEL_DIR, "lgb_model_tochi.txt"),
    },
}

CURRENT_MACRO = {
    "JGB2Y": 0.50, "JGB10Y": 1.50, "USDJPY": 150.0,
    "建材_前年比": 4.5, "着工戸数": 70000, "全国人口増減率": -0.4,
    "日経平均": 35000,
}

SCENARIOS = {
    "現状維持":     {"USDJPY": 150, "JGB10Y": 1.5, "JGB2Y": 0.5,  "建材_前年比":  4.5, "日経平均": 35000},
    "円安継続":     {"USDJPY": 170, "JGB10Y": 1.5, "JGB2Y": 0.5,  "建材_前年比":  5.0, "日経平均": 40000},
    "金利正常化":   {"USDJPY": 140, "JGB10Y": 2.5, "JGB2Y": 1.0,  "建材_前年比":  4.5, "日経平均": 30000},
    "景気過熱":     {"USDJPY": 160, "JGB10Y": 2.0, "JGB2Y": 0.8,  "建材_前年比":  8.0, "日経平均": 45000},
    "複合ストレス": {"USDJPY": 170, "JGB10Y": 3.0, "JGB2Y": 1.5,  "建材_前年比": 10.0, "日経平均": 22000},
}

# 23区 重心座標
WARD_COORDS = {
    "千代田区": (35.6942, 139.7536), "中央区":   (35.6706, 139.7728),
    "港区":     (35.6581, 139.7514), "新宿区":   (35.6938, 139.7034),
    "文京区":   (35.7077, 139.7518), "台東区":   (35.7126, 139.7803),
    "墨田区":   (35.7101, 139.8014), "江東区":   (35.6734, 139.8170),
    "品川区":   (35.6090, 139.7302), "目黒区":   (35.6414, 139.6985),
    "大田区":   (35.5614, 139.7161), "世田谷区": (35.6464, 139.6536),
    "渋谷区":   (35.6640, 139.6982), "中野区":   (35.7075, 139.6640),
    "杉並区":   (35.6994, 139.6366), "豊島区":   (35.7282, 139.7177),
    "北区":     (35.7528, 139.7336), "荒川区":   (35.7361, 139.7826),
    "板橋区":   (35.7508, 139.7097), "練馬区":   (35.7358, 139.6516),
    "足立区":   (35.7751, 139.8046), "葛飾区":   (35.7474, 139.8469),
    "江戸川区": (35.7064, 139.8687),
}

SHAPE_SCORE_MAP = {
    "長方形 / 正方形": 1.00,
    "ほぼ長方形":      0.90,
    "台形":            0.70,
    "L字型":           0.30,
    "不整形":          0.00,
}

# 間取り → 居室数（モデルの学習値に合わせる）
MADORI_MAP = {
    "1R / 1K":   1,
    "1DK":       1,
    "1LDK":      1,
    "2K / 2DK":  2,
    "2LDK":      2,
    "3DK":       3,
    "3LDK":      3,
    "4LDK":      4,
    "5LDK以上":  5,
}

SWEEP_MANSION = [
    ("面積（㎡）",               [20, 30, 40, 50, 60, 70, 80, 100]),
    ("築年数",                   [0, 5, 10, 15, 20, 25, 30, 40]),
    ("東京駅までの直線距離(km)", [1, 3, 5, 8, 10, 15, 20, 30]),
    ("最寄駅：距離（分）",       [1, 3, 5, 7, 10, 15, 20]),
    ("USDJPY",                   [120, 130, 140, 150, 160, 170]),
    ("JGB10Y",                   [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]),
    ("日経平均",                 [20000, 25000, 30000, 35000, 40000, 45000, 50000]),
    ("建材_前年比",              [-3, 0, 2, 4.5, 6, 8, 10]),
]

SWEEP_TOCHI = [
    ("面積（㎡）",               [50, 75, 100, 125, 150, 200, 300]),
    ("延床面積（㎡）",           [50, 70, 90, 110, 130, 160, 200]),
    ("築年数",                   [0, 5, 10, 15, 20, 30, 40]),
    ("東京駅までの直線距離(km)", [3, 5, 8, 10, 15, 20, 25, 30]),
    ("最寄駅：距離（分）",       [3, 5, 7, 10, 15, 20]),
    ("前面道路：幅員（ｍ）",     [2, 3, 4, 6, 8, 10]),
    ("USDJPY",                   [120, 130, 140, 150, 160, 170]),
    ("日経平均",                 [20000, 25000, 30000, 35000, 40000, 45000, 50000]),
]

_HEADERS = {"User-Agent": "Mozilla/5.0"}

# ══════════════════════════════════════════════════════════════════
# モデルロード / マクロ取得
# ══════════════════════════════════════════════════════════════════

@st.cache_resource
def load_model(prop_type: str):
    p   = PATHS[prop_type]
    art = joblib.load(p["artifacts"])
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        tmp = f.name
    shutil.copy(p["lgb"], tmp)
    try:
        booster = lgb.Booster(model_file=tmp)
    finally:
        os.unlink(tmp)
    return art, booster


@st.cache_data(ttl=3600)
def fetch_macro_live() -> dict:
    macro = dict(CURRENT_MACRO)
    try:
        r = requests.get(
            "https://query1.finance.yahoo.com/v8/finance/chart/USDJPY%3DX?interval=1d&range=5d",
            headers=_HEADERS, timeout=10)
        closes = [c for c in r.json()["chart"]["result"][0]["indicators"]["quote"][0]["close"] if c]
        if closes:
            macro["USDJPY"] = round(closes[-1], 2)
    except Exception:
        pass
    try:
        r = requests.get(
            "https://www.mof.go.jp/jgbs/reference/interest_rate/data/jgbcm_all.csv",
            headers=_HEADERS, timeout=15)
        r.encoding = "shift_jis"
        lines = [l for l in r.text.strip().splitlines()
                 if l and not l.startswith("国債") and not l.startswith("基準")]
        last = lines[-1].split(",")
        macro["JGB2Y"]  = round(float(last[2]),  3)
        macro["JGB10Y"] = round(float(last[10]), 3)
    except Exception:
        pass
    try:
        r = requests.get(
            "https://query1.finance.yahoo.com/v8/finance/chart/%5EN225?interval=1d&range=5d",
            headers=_HEADERS, timeout=10)
        closes = [c for c in r.json()["chart"]["result"][0]["indicators"]["quote"][0]["close"] if c]
        if closes:
            macro["日経平均"] = round(closes[-1])
    except Exception:
        pass
    return macro


# ══════════════════════════════════════════════════════════════════
# 駅名 → 東京駅距離
# ══════════════════════════════════════════════════════════════════
_TOKYO_ST = (35.6812, 139.7671)  # 東京駅

def _haversine(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(a))


@st.cache_data(ttl=86400, show_spinner=False)
def geocode_station(name: str) -> tuple[float, float] | None:
    """駅名 → (lat, lon)  Nominatim (OpenStreetMap) 使用"""
    query = f"{name}駅 東京都"
    try:
        r = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": query, "format": "json", "limit": 1, "countrycodes": "jp"},
            headers={"User-Agent": "tokyo-realestate-app/1.0 (roxity10@gmail.com)"},
            timeout=10,
        )
        data = r.json()
        if data:
            return float(data[0]["lat"]), float(data[0]["lon"])
    except Exception:
        pass
    return None


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_stations_23wards() -> tuple[pd.DataFrame, str]:
    """Overpass API で23区内の鉄道駅を取得（キャッシュ24時間）。(df, error_msg) を返す"""
    # node のみのシンプルクエリ → 失敗時は空を返す
    query = (
        "[out:json][timeout:30];\n"
        "node[\"railway\"=\"station\"](35.50,139.50,35.85,139.95);\n"
        "out body;"
    )
    try:
        r = requests.post(
            "https://overpass-api.de/api/interpreter",
            data={"data": query},
            headers={"User-Agent": "tokyo-realestate-app/1.0 (roxity10@gmail.com)"},
            timeout=40,
        )
        r.raise_for_status()
        elements = r.json().get("elements", [])
        rows = []
        for e in elements:
            tags = e.get("tags", {})
            name = tags.get("name:ja") or tags.get("name", "")
            if name:
                rows.append({"駅名": name, "lat": e["lat"], "lon": e["lon"]})
        if not rows:
            return pd.DataFrame(columns=["駅名", "lat", "lon"]), "API は応答しましたが駅データが空でした"
        df = pd.DataFrame(rows).drop_duplicates("駅名").reset_index(drop=True)
        return df, ""
    except requests.exceptions.Timeout:
        return pd.DataFrame(columns=["駅名", "lat", "lon"]), "Overpass API タイムアウト（サーバー混雑）"
    except Exception as ex:
        return pd.DataFrame(columns=["駅名", "lat", "lon"]), str(ex)


def _nearest_ward(lat: float, lon: float) -> str:
    """最近傍の区を返す"""
    return min(WARD_COORDS, key=lambda w: _haversine(lat, lon, *WARD_COORDS[w]))


def station_to_dist(name: str) -> float | None:
    """駅名 → 東京駅からの直線距離(km)。失敗時はNone"""
    coords = geocode_station(name.strip())
    if coords:
        return round(_haversine(*_TOKYO_ST, *coords), 1)
    return None


# ══════════════════════════════════════════════════════════════════
# 予測ヘルパー
# ══════════════════════════════════════════════════════════════════

def apply_flags(params: dict, art: dict, prop_type: str) -> dict:
    p  = params.copy()
    nm = art["num_medians"]
    dk = p.get("東京駅までの直線距離(km)", nm.get("東京駅までの直線距離(km)", 10))
    dm = p.get("最寄駅：距離（分）",        nm.get("最寄駅：距離（分）", 7))
    p.setdefault("都心フラグ",     1 if dk <= 5  else 0)
    p.setdefault("東京駅5km以内",  1 if dk <= 5  else 0)
    p.setdefault("東京駅15km以内", 1 if dk <= 15 else 0)
    p.setdefault("徒歩5分以内",    1 if dm <= 5  else 0)
    p.setdefault("徒歩10分超",     1 if dm > 10  else 0)
    if prop_type == "mansion":
        area = p.get("面積（㎡）", nm.get("面積（㎡）", 55))
        p.setdefault("log_面積", np.log1p(area))
    else:
        land  = p.get("面積（㎡）",          nm.get("面積（㎡）",          100))
        floor = p.get("延床面積（㎡）",       nm.get("延床面積（㎡）",       95))
        rw    = p.get("前面道路：幅員（ｍ）", nm.get("前面道路：幅員（ｍ）", 4))
        p.setdefault("接道フラグ",   1.0 if rw >= 4 else 0.0)
        p.setdefault("log_土地面積", np.log1p(land))
        p.setdefault("log_延床面積", np.log1p(floor))
        if "延床面積比" not in p:
            p["延床面積比"] = float(np.clip(floor / land, 0.1, 5.0)) if land > 0 else 1.0
    return p


def build_row(params: dict, art: dict) -> pd.DataFrame:
    row = {}
    for col in art["all_num_features"]:
        row[col] = float(params.get(col, art["num_medians"][col]))
    for col in art["cat_features"]:
        val = str(params.get(col, art["cat_modes"][col]))
        le  = art["encoders"][col]
        row[col] = (le.transform([val])[0] if val in le.classes_
                    else le.transform([art["cat_modes"][col]])[0])
    arr = art["imputer"].transform(pd.DataFrame([row], columns=art["feature_cols"]))
    return pd.DataFrame(arr, columns=art["feature_cols"])


def predict(params: dict, art: dict, booster, prop_type: str) -> dict:
    p   = apply_flags(params, art, prop_type)
    row = build_row(p, art)
    raw = float(booster.predict(row.values)[0])
    if prop_type == "mansion":
        area = float(params.get("面積（㎡）", art["num_medians"]["面積（㎡）"]))
        return {"単価_万円/㎡": raw, "総額_万円": raw * area}
    return {"総額_万円": float(np.expm1(raw))}


def get_ward_pop(ward: str, art: dict) -> dict:
    wl  = art.get("ward_latest", pd.DataFrame())
    row = wl[wl["市区町村名"] == ward] if len(wl) else pd.DataFrame()
    if len(row):
        return {"区別人口": int(row["区別人口"].iloc[0]),
                "区別人口増減率": float(row["区別人口増減率"].iloc[0])}
    return {"区別人口": int(art["num_medians"].get("区別人口", 500000)),
            "区別人口増減率": 0.0}


# ══════════════════════════════════════════════════════════════════
# Streamlit UI
# ══════════════════════════════════════════════════════════════════

st.set_page_config(page_title="東京不動産価格推定", page_icon="🏠", layout="wide")
st.title("🏠 東京不動産価格推定")
st.caption("LightGBM モデル ｜ 学習データ: 2005〜2025年 東京都取引実績")

# ── サイドバー: 物件タイプ ────────────────────────────────────────
with st.sidebar:
    st.header("物件タイプ")
    prop_label = st.radio("", ["中古マンション", "中古戸建て"], horizontal=True)
    prop_type  = "mansion" if prop_label == "中古マンション" else "tochi"

# ── モデルロード ────────────────────────────────────────────────────
try:
    art, booster = load_model(prop_type)
except Exception as e:
    st.error(f"モデルが見つかりません。先に学習スクリプトを実行してください。\n{e}")
    st.stop()

_23WARDS = [
    "千代田区", "中央区", "港区", "新宿区", "文京区",
    "台東区", "墨田区", "江東区", "品川区", "目黒区",
    "大田区", "世田谷区", "渋谷区", "中野区", "杉並区",
    "豊島区", "北区", "荒川区", "板橋区", "練馬区",
    "足立区", "葛飾区", "江戸川区",
]

def _sort_wards(ward_list: list[str]) -> list[str]:
    wards_23  = [w for w in _23WARDS   if w in ward_list]
    others    = sorted(w for w in ward_list if w not in _23WARDS)
    return wards_23 + others

WARD_LIST = _sort_wards(art["WARD_LIST"])
lgb_r2    = art["lgb_r2"]

# ── サイドバー: 物件情報 ────────────────────────────────────────────
with st.sidebar:
    st.header("物件情報")
    default_ward = "世田谷区" if "世田谷区" in WARD_LIST else WARD_LIST[0]
    ward = st.selectbox("区", WARD_LIST, index=WARD_LIST.index(default_ward))

    if prop_type == "mansion":
        area   = st.slider("専有面積（㎡）",   20,  150, 60, 5)
        age    = st.slider("築年数（年）",       0,   50, 10, 1)
        dist_m = st.slider("最寄駅 徒歩（分）", 1,   30,  7,  1)
        madori = st.selectbox("間取り", list(MADORI_MAP.keys()), index=4)
        rooms  = MADORI_MAP[madori]
        struct = st.selectbox("建物の構造", ["RC", "SRC", "鉄骨造", "木造"])
    else:
        land   = st.slider("土地面積（㎡）",    30,  400, 100, 5)
        floor  = st.slider("延床面積（㎡）",    30,  400,  95, 5)
        age    = st.slider("築年数（年）",        0,   50,  10, 1)
        dist_m = st.slider("最寄駅 徒歩（分）",  1,   30,   7,  1)
        road_w = st.slider("前面道路幅員（m）", 2.0, 12.0,  4.0, 0.5)
        struct = st.selectbox("建物の構造", ["木造", "鉄骨造", "RC", "SRC"])
        shape_label = st.selectbox("土地の形状", list(SHAPE_SCORE_MAP.keys()))
        shape_score = SHAPE_SCORE_MAP[shape_label]

    # ── 最寄り駅 → 東京駅距離 ──────────────────────────────────────
    st.divider()
    _default_km = 8.0 if prop_type == "mansion" else 13.0
    station_name = st.text_input("最寄り駅名", placeholder="例: 渋谷、吉祥寺、浦安")
    if station_name.strip():
        _d = station_to_dist(station_name)
        if _d is not None:
            dist_k = _d
            st.caption(f"📍 {station_name}駅 → 東京駅 **{dist_k:.1f} km**")
        else:
            st.warning("駅が見つかりませんでした。手動で入力してください。")
            dist_k = st.number_input("東京駅 直線距離（km）", 0.5, 50.0, _default_km, 0.5)
    else:
        dist_k = st.number_input("東京駅 直線距離（km）", 0.5, 50.0, _default_km, 0.5,
                                 help="駅名を入力すると自動計算されます")

    # ── マクロ条件 ──────────────────────────────────────────────────
    st.header("マクロ条件")
    if st.button("🔄 最新値を自動取得", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    live = fetch_macro_live()
    usdjpy  = st.number_input("USDJPY（円/ドル）",      value=float(live["USDJPY"]),       step=1.0,  format="%.2f")
    jgb2y   = st.number_input("JGB2Y 短期金利（%）",    value=float(live["JGB2Y"]),        step=0.05, format="%.3f")
    jgb10y  = st.number_input("JGB10Y 長期金利（%）",   value=float(live["JGB10Y"]),       step=0.05, format="%.3f")
    nikkei  = st.number_input("日経平均株価（円）",      value=int(live["日経平均"]),       step=500)
    zaim    = st.number_input("建材_前年比（%）",        value=float(live["建材_前年比"]),  step=0.5,  format="%.1f")
    starts  = st.number_input("住宅着工戸数（月間）",    value=int(live["着工戸数"]),       step=1000)
    pop_chg = st.number_input("全国人口増減率（%）",     value=float(live["全国人口増減率"]), step=0.1, format="%.1f")

# ── パラメータ構築 ────────────────────────────────────────────────
macro_params = {
    "USDJPY": usdjpy, "JGB2Y": jgb2y, "JGB10Y": jgb10y,
    "建材_前年比": zaim, "着工戸数": starts, "全国人口増減率": pop_chg,
    "日経平均": nikkei,
}
ward_pop = get_ward_pop(ward, art)

if prop_type == "mansion":
    base_params = {
        "市区町村名": ward, "建物の構造": struct,
        "面積（㎡）": area, "築年数": age,
        "東京駅までの直線距離(km)": dist_k,
        "最寄駅：距離（分）": dist_m,
        "居室数": rooms,
        **macro_params, **ward_pop,
    }
else:
    base_params = {
        "市区町村名": ward, "建物の構造": struct,
        "面積（㎡）": land, "延床面積（㎡）": floor,
        "築年数": age,
        "東京駅までの直線距離(km)": dist_k,
        "最寄駅：距離（分）": dist_m,
        "前面道路：幅員（ｍ）": road_w,
        "土地形状スコア": shape_score,
        **macro_params, **ward_pop,
    }

result  = predict(base_params, art, booster, prop_type)
tot     = result["総額_万円"]

# ── タブ ─────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📊 推定結果", "📈 感度分析", "🔮 シナリオ比較", "🗺️ 23区マップ"])

# ══ Tab1: 推定結果 ════════════════════════════════════════════════
with tab1:
    if prop_type == "mansion":
        unit = result["単価_万円/㎡"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("推定総額",    f"{tot:,.0f} 万円")
        c2.metric("億円換算",    f"{tot/10000:.2f} 億円")
        c3.metric("㎡単価",      f"{unit:,.1f} 万円/㎡")
        c4.metric("坪単価",      f"{unit*3.3057:,.1f} 万円/坪")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("推定総額",    f"{tot:,.0f} 万円")
        c2.metric("億円換算",    f"{tot/10000:.2f} 億円")
        c3.metric("㎡単価（土地）", f"{tot/land:,.1f} 万円/㎡" if prop_type == "tochi" else "-")

    st.divider()

    if prop_type == "mansion":
        st.subheader("面積別 換算表")
        rows = []
        for a in [20, 30, 40, 50, 60, 70, 80, 100, 120]:
            p = {**base_params, "面積（㎡）": a}
            r = predict(p, art, booster, prop_type)
            rows.append({
                "面積(㎡)": a,
                "総額(万円)": f"{r['総額_万円']:,.0f}",
                "㎡単価":     f"{r['単価_万円/㎡']:,.1f}",
                "坪単価":     f"{r['単価_万円/㎡']*3.3057:,.1f}",
            })
        st.dataframe(pd.DataFrame(rows).set_index("面積(㎡)"), use_container_width=True)
    else:
        st.subheader("延床面積別 換算表（土地面積固定）")
        rows = []
        for f in [50, 60, 80, 95, 110, 130, 160, 200]:
            p = {**base_params, "延床面積（㎡）": f}
            p.pop("延床面積比",    None)
            p.pop("log_延床面積", None)
            r = predict(p, art, booster, prop_type)
            rows.append({"延床面積(㎡)": f, "総額(万円)": f"{r['総額_万円']:,.0f}"})
        st.dataframe(pd.DataFrame(rows).set_index("延床面積(㎡)"), use_container_width=True)

    st.caption(f"モデル: LightGBM ｜ 学習R²: {lgb_r2:.4f} ｜ ※ 学習データへの自己評価スコア（過大評価の可能性あり）")

# ══ Tab2: 感度分析 ════════════════════════════════════════════════
with tab2:
    st.subheader("感度分析")
    sweep = SWEEP_MANSION if prop_type == "mansion" else SWEEP_TOCHI

    # Tornado chart データ生成
    tornado_rows = []
    for param, values in sweep:
        prices = []
        for v in values:
            p = {**base_params, param: v}
            if param == "面積（㎡）" and prop_type == "tochi":
                ratio = base_params.get("延床面積（㎡）", 95) / max(base_params.get("面積（㎡）", 100), 1)
                p["延床面積（㎡）"] = float(np.clip(v * ratio, 20, 500))
                p.pop("延床面積比",    None)
                p.pop("log_延床面積", None)
            prices.append(predict(p, art, booster, prop_type)["総額_万円"])
        lo, hi = min(prices), max(prices)
        tornado_rows.append({
            "パラメータ": param,
            "最小(万円)": lo, "最大(万円)": hi,
            "最小変化率(%)": (lo - tot) / tot * 100,
            "最大変化率(%)": (hi - tot) / tot * 100,
            "変動幅(万円)": hi - lo,
        })

    df_t = pd.DataFrame(tornado_rows).sort_values("変動幅(万円)", ascending=True)

    fig_t = go.Figure()
    for _, row in df_t.iterrows():
        lo_c = row["最小変化率(%)"]
        hi_c = row["最大変化率(%)"]
        left = min(lo_c, hi_c)
        width = abs(hi_c - lo_c)
        fig_t.add_trace(go.Bar(
            y=[row["パラメータ"]],
            x=[width],
            base=[left],
            orientation="h",
            marker_color="steelblue",
            showlegend=False,
            text=f"{lo_c:+.1f}% 〜 {hi_c:+.1f}%",
            textposition="outside",
        ))
    fig_t.add_vline(x=0, line_dash="dash", line_color="gray", line_width=1)
    fig_t.update_layout(
        title="各パラメータの価格変動幅（vs 現在設定値）",
        xaxis_title="変化率（%）",
        height=420,
        margin=dict(l=10, r=120, t=50, b=40),
    )
    st.plotly_chart(fig_t, use_container_width=True)

    # 詳細折れ線グラフ
    st.subheader("パラメータ別 詳細グラフ")
    sel_param  = st.selectbox("パラメータを選択", [s[0] for s in sweep])
    sel_values = next(v for p, v in sweep if p == sel_param)

    detail_rows = []
    for v in sel_values:
        p = {**base_params, sel_param: v}
        if sel_param == "面積（㎡）" and prop_type == "tochi":
            ratio = base_params.get("延床面積（㎡）", 95) / max(base_params.get("面積（㎡）", 100), 1)
            p["延床面積（㎡）"] = float(np.clip(v * ratio, 20, 500))
            p.pop("延床面積比",    None)
            p.pop("log_延床面積", None)
        r = predict(p, art, booster, prop_type)
        detail_rows.append({"value": v, "総額_万円": r["総額_万円"]})

    df_d    = pd.DataFrame(detail_rows)
    base_x  = base_params.get(sel_param, art["num_medians"].get(sel_param))

    fig_d = go.Figure()
    fig_d.add_trace(go.Scatter(
        x=df_d["value"], y=df_d["総額_万円"],
        mode="lines+markers", name="推定総額",
        line=dict(color="steelblue", width=2),
        marker=dict(size=8),
    ))
    if base_x is not None:
        fig_d.add_vline(
            x=float(base_x), line_dash="dash", line_color="orange",
            annotation_text="現在値", annotation_position="top right",
        )
    fig_d.update_layout(
        xaxis_title=sel_param,
        yaxis_title="推定総額（万円）",
        height=350,
        margin=dict(l=10, r=10, t=30, b=40),
    )
    st.plotly_chart(fig_d, use_container_width=True)

# ══ Tab3: シナリオ比較 ════════════════════════════════════════════
with tab3:
    st.subheader("マクロシナリオ別 価格比較")

    # ── カスタムシナリオ入力 ────────────────────────────────────────
    with st.expander("➕ カスタムシナリオを追加", expanded=False):
        st.caption("任意のマクロ条件を設定して既存シナリオと比較できます")
        cc1, cc2 = st.columns(2)
        with cc1:
            c_name   = st.text_input("シナリオ名", value="カスタム")
            c_usd    = st.number_input("USDJPY（円/ドル）",    value=float(CURRENT_MACRO["USDJPY"]),      step=1.0,  format="%.1f", key="c_usd")
            c_jgb2y  = st.number_input("JGB2Y 短期金利（%）", value=float(CURRENT_MACRO["JGB2Y"]),       step=0.05, format="%.3f", key="c_jgb2y")
            c_jgb10y = st.number_input("JGB10Y 長期金利（%）",value=float(CURRENT_MACRO["JGB10Y"]),      step=0.05, format="%.3f", key="c_jgb10y")
        with cc2:
            c_nikkei = st.number_input("日経平均株価（円）",   value=int(CURRENT_MACRO["日経平均"]),      step=500,               key="c_nikkei")
            c_zaim   = st.number_input("建材_前年比（%）",     value=float(CURRENT_MACRO["建材_前年比"]), step=0.5,  format="%.1f", key="c_zaim")
            c_starts = st.number_input("住宅着工戸数（月間）", value=int(CURRENT_MACRO["着工戸数"]),      step=1000,              key="c_starts")
            c_pop    = st.number_input("全国人口増減率（%）",  value=float(CURRENT_MACRO["全国人口増減率"]), step=0.1, format="%.1f", key="c_pop")
        use_custom = st.checkbox("グラフに追加する", value=True)

    custom_macro = {
        "USDJPY": c_usd, "JGB2Y": c_jgb2y, "JGB10Y": c_jgb10y,
        "日経平均": c_nikkei, "建材_前年比": c_zaim,
        "着工戸数": c_starts, "全国人口増減率": c_pop,
    }

    # ── 全シナリオ計算 ──────────────────────────────────────────────
    scen_rows = []
    base_scen = None
    for name, macro in SCENARIOS.items():
        p = {**base_params, **CURRENT_MACRO, **macro}
        r = predict(p, art, booster, prop_type)
        if base_scen is None:
            base_scen = r["総額_万円"]
        chg = (r["総額_万円"] - base_scen) / base_scen * 100
        scen_rows.append({
            "シナリオ": name,
            "総額_万円": r["総額_万円"],
            "変化率_%": chg,
            "USDJPY":    macro["USDJPY"],
            "JGB10Y":    macro["JGB10Y"],
            "建材_前年比": macro.get("建材_前年比", CURRENT_MACRO["建材_前年比"]),
            "日経平均":  macro.get("日経平均", CURRENT_MACRO["日経平均"]),
            "_custom":   False,
        })

    # カスタムシナリオを追加
    if use_custom:
        p = {**base_params, **CURRENT_MACRO, **custom_macro}
        r = predict(p, art, booster, prop_type)
        chg = (r["総額_万円"] - base_scen) / base_scen * 100
        scen_rows.append({
            "シナリオ": c_name if c_name.strip() else "カスタム",
            "総額_万円": r["総額_万円"],
            "変化率_%": chg,
            "USDJPY":    c_usd,
            "JGB10Y":    c_jgb10y,
            "建材_前年比": c_zaim,
            "日経平均":  c_nikkei,
            "_custom":   True,
        })

    df_s = pd.DataFrame(scen_rows)

    # ── 棒グラフ ────────────────────────────────────────────────────
    colors = []
    for _, row in df_s.iterrows():
        if row["_custom"]:
            colors.append("#E8A020")        # カスタム = オレンジ
        elif row["変化率_%"] >= 0:
            colors.append("#4878CF")        # プラス = 青
        else:
            colors.append("#CF4848")        # マイナス = 赤

    fig_s = go.Figure(go.Bar(
        x=df_s["シナリオ"],
        y=df_s["総額_万円"],
        marker_color=colors,
        text=[f"{v:,.0f}万円<br>{c:+.1f}%" for v, c in zip(df_s["総額_万円"], df_s["変化率_%"])],
        textposition="outside",
    ))
    ymin = df_s["総額_万円"].min()
    ymax = df_s["総額_万円"].max()
    fig_s.update_layout(
        yaxis=dict(range=[ymin * 0.88, ymax * 1.12]),
        yaxis_title="推定総額（万円）",
        height=420,
        margin=dict(l=10, r=10, t=40, b=40),
    )
    st.plotly_chart(fig_s, use_container_width=True)

    # ── 詳細テーブル ────────────────────────────────────────────────
    st.subheader("シナリオ詳細")
    disp = df_s.drop(columns=["_custom"]).copy()
    disp["総額(万円)"] = disp["総額_万円"].apply(lambda x: f"{x:,.0f}")
    disp["変化率"]     = disp["変化率_%"].apply(lambda x: f"{x:+.1f}%")
    disp["日経平均"]   = disp["日経平均"].apply(lambda x: f"{x:,}")
    st.dataframe(
        disp[["シナリオ", "USDJPY", "JGB10Y", "建材_前年比", "日経平均", "総額(万円)", "変化率"]]
        .set_index("シナリオ"),
        use_container_width=True,
    )

# ══ Tab4: 23区マップ ══════════════════════════════════════════════
with tab4:
    st.subheader("23区 推定価格マップ（主要駅）")

    # ── コントロール ──────────────────────────────────────────────
    ctrl1, ctrl2, ctrl3 = st.columns([3, 1, 1])
    with ctrl1:
        zoom_level = st.slider("ズームレベル", 9.0, 13.0, 10.5, 0.5, label_visibility="collapsed")
    with ctrl2:
        st.caption(f"🔍 ズーム: {zoom_level}")
    with ctrl3:
        show_rings = st.toggle("距離リング", value=True)

    # ── 距離リング生成 ────────────────────────────────────────────
    def _ring(radius_km: float, n: int = 120):
        clat, clon = _TOKYO_ST
        lats, lons = [], []
        for i in range(n + 1):
            a = 2 * math.pi * i / n
            lats.append(clat + math.degrees(radius_km / 6371.0) * math.sin(a))
            lons.append(clon + math.degrees(radius_km / (6371.0 * math.cos(math.radians(clat)))) * math.cos(a))
        return lats, lons

    # ── 駅データ取得 ─────────────────────────────────────────────
    with st.spinner("鉄道駅データを取得中..."):
        df_st, fetch_err = fetch_stations_23wards()

    if df_st.empty:
        st.warning(f"駅データの取得に失敗しました。\n\n原因: {fetch_err}")
        if st.button("🔄 駅データを再取得", key="retry_stations"):
            st.cache_data.clear()
            st.rerun()
    else:
        st.caption(f"取得駅数: {len(df_st)} 駅 ｜ サイドバー条件で各駅の推定価格を算出")

        # ── 各駅の予測（バッチ処理） ────────────────────────────────
        known_classes = set(art["encoders"]["市区町村名"].classes_)

        # 駅ごとにパラメータ行を生成
        prep_rows = []
        for _, st_row in df_st.iterrows():
            lat, lon = st_row["lat"], st_row["lon"]
            ward     = _nearest_ward(lat, lon)
            if ward not in known_classes:
                continue
            dist = _haversine(*_TOKYO_ST, lat, lon)
            p = {**base_params,
                 "市区町村名":               ward,
                 "東京駅までの直線距離(km)": round(dist, 2),
                 "最寄駅：距離（分）":       1}
            for flag in ["都心フラグ", "東京駅5km以内", "東京駅15km以内",
                         "徒歩5分以内", "徒歩10分超"]:
                p.pop(flag, None)
            p.update(get_ward_pop(ward, art))
            prep_rows.append({
                "_名前": st_row["駅名"], "_lat": lat, "_lon": lon,
                "_区": ward, "_dist": round(dist, 1),
                **apply_flags(p, art, prop_type),
            })

        # DataFrameをまとめて構築してバッチ予測
        map_rows = []
        if prep_rows:
            meta_cols = ["_名前", "_lat", "_lon", "_区", "_dist"]
            batch_params = [{k: v for k, v in row.items() if k not in meta_cols}
                            for row in prep_rows]
            feat_rows = []
            for p in batch_params:
                row_d = {}
                for col in art["all_num_features"]:
                    row_d[col] = float(p.get(col, art["num_medians"][col]))
                for col in art["cat_features"]:
                    val = str(p.get(col, art["cat_modes"][col]))
                    le  = art["encoders"][col]
                    row_d[col] = (le.transform([val])[0] if val in le.classes_
                                  else le.transform([art["cat_modes"][col]])[0])
                feat_rows.append(row_d)
            df_feat = pd.DataFrame(feat_rows, columns=art["feature_cols"])
            arr = art["imputer"].transform(df_feat)
            preds_raw = booster.predict(arr)

            area_val = float(base_params.get("面積（㎡）", art["num_medians"].get("面積（㎡）", 60)))
            for i, row in enumerate(prep_rows):
                raw = float(preds_raw[i])
                if prop_type == "mansion":
                    total = raw * area_val
                    unit  = raw
                else:
                    total = float(np.expm1(raw))
                    unit  = total / area_val
                map_rows.append({
                    "駅名":       row["_名前"],
                    "lat":        row["_lat"],
                    "lon":        row["_lon"],
                    "区":         row["_区"],
                    "距離(km)":   row["_dist"],
                    "総額_万円":    round(total),
                    "単価_万円/㎡": round(unit, 1),
                })

        df_map = pd.DataFrame(map_rows).sort_values("距離(km)")
        price_col   = "単価_万円/㎡" if prop_type == "mansion" else "総額_万円"
        price_label = "万円/㎡"      if prop_type == "mansion" else "総額(万円)"

        # ── 地図描画 ─────────────────────────────────────────────
        fig_map = go.Figure()

        # 距離リング（5・10・15・20km）
        if show_rings:
            for r_km in [5, 10, 15, 20]:
                rlat, rlon = _ring(r_km)
                fig_map.add_trace(go.Scattermapbox(
                    lat=rlat, lon=rlon, mode="lines",
                    line=dict(width=1, color="rgba(80,80,80,0.30)"),
                    hoverinfo="none", showlegend=False,
                ))
                fig_map.add_trace(go.Scattermapbox(
                    lat=[rlat[15]], lon=[rlon[15]],
                    mode="text", text=[f"{r_km}km"],
                    textfont=dict(size=10, color="gray"),
                    hoverinfo="none", showlegend=False,
                ))

        # 東京駅マーカー
        fig_map.add_trace(go.Scattermapbox(
            lat=[_TOKYO_ST[0]], lon=[_TOKYO_ST[1]],
            mode="markers+text",
            marker=dict(size=14, color="black"),
            text=["🚉 東京駅"], textposition="top right",
            textfont=dict(size=11, color="black"),
            hovertext=["東京駅"], hoverinfo="text",
            showlegend=False,
        ))

        # 駅バブル
        hover_texts = [
            f"<b>{row['駅名']}</b>（{row['区']}）<br>"
            f"東京駅から {row['距離(km)']} km<br>"
            f"推定総額: {row['総額_万円']:,} 万円<br>"
            f"㎡単価: {row['単価_万円/㎡']:,.1f} 万円/㎡"
            for _, row in df_map.iterrows()
        ]
        fig_map.add_trace(go.Scattermapbox(
            lat=df_map["lat"],
            lon=df_map["lon"],
            mode="markers",
            marker=dict(
                size=11,
                color=df_map[price_col],
                colorscale="RdYlGn_r",
                showscale=True,
                colorbar=dict(title=price_label, thickness=14),
                opacity=0.85,
            ),
            hovertext=hover_texts,
            hoverinfo="text",
            showlegend=False,
        ))

        fig_map.update_layout(
            mapbox=dict(
                style="open-street-map",
                center=dict(lat=35.685, lon=139.730),
                zoom=zoom_level,
            ),
            height=660,
            margin=dict(l=0, r=0, t=0, b=0),
        )
        st.plotly_chart(fig_map, use_container_width=True)

        # ── テーブル ─────────────────────────────────────────────
        with st.expander("駅別 推定価格一覧（東京駅距離順）"):
            tbl = df_map[["駅名", "区", "距離(km)", "総額_万円", "単価_万円/㎡"]].copy()
            tbl["総額(万円)"] = tbl["総額_万円"].apply(lambda x: f"{x:,}")
            tbl["㎡単価"]     = tbl["単価_万円/㎡"].apply(lambda x: f"{x:,.1f}")
            st.dataframe(
                tbl[["駅名", "区", "距離(km)", "総額(万円)", "㎡単価"]].set_index("駅名"),
                use_container_width=True,
            )
