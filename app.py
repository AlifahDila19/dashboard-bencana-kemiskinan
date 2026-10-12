import altair as alt
import folium
import geopandas as gpd
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

# Konfigurasi Layout Dashboard Profesional
st.set_page_config(
    page_title="Executive Spatial Dashboard | Bencana & Kemiskinan",
    layout="wide",
)

# Custom CSS Korporat & Bersih
st.markdown(
    """
    <style>
        .main-header { font-size: 1.7rem; font-weight: 700; color: #0f172a; margin-bottom: 0px; }
        .sub-header { font-size: 0.9rem; color: #64748b; margin-bottom: 20px; }
        .kpi-container { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 15px; text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.04); min-height: 95px; }
        .kpi-label { font-size: 0.66rem; text-transform: uppercase; color: #64748b; font-weight: 600; letter-spacing: 0.5px; }
        .kpi-value { font-size: 1.25rem; font-weight: 700; color: #1e293b; margin-top: 5px; word-break: break-word; }
        .insight-box { background: #f8fafc; border-left: 4px solid #3b82f6; padding: 14px; border-radius: 0 6px 6px 0; margin-bottom: 10px; font-size: 0.88rem; color: #334155; }
        .recommendation-box { background: #f0fdf4; border-left: 4px solid #22c55e; padding: 14px; border-radius: 0 6px 6px 0; margin-bottom: 10px; font-size: 0.88rem; color: #166534; }
        .priority-box { background: #fef2f2; border-left: 4px solid #dc2626; padding: 14px; border-radius: 0 6px 6px 0; margin-bottom: 10px; font-size: 0.88rem; color: #7f1d1d; }
        .warning-box { background: #fffbeb; border-left: 4px solid #f59e0b; padding: 14px; border-radius: 0 6px 6px 0; margin-bottom: 10px; font-size: 0.88rem; color: #78350f; }
    </style>
""",
    unsafe_allow_html=True,
)

REQUIRED = [
    "key", "nama", "provinsi", "cluster_label", "pct_miskin", "kejadian",
    "rumah_rusak", "jumlah_jiwa", "usia_produktif_pct",
]
PALETTE = ["#fc8d62", "#8da0cb", "#5ab4ac", "#e7298a", "#a6d854", "#ffd92f"]


# ---------------------------------------------------------------- Fungsi bantu
def fmt_int(x):
    return f"{x:,.0f}".replace(",", ".")


def fmt_dec(x, d=2):
    return f"{x:,.{d}f}".replace(",", "#").replace(".", ",").replace("#", ".")


def fmt_ringkas(x):
    """Angka besar ditulis ringkas (juta) agar kartu KPI tidak terpotong."""
    if pd.isna(x):
        return "-"
    return fmt_dec(x / 1e6, 1) + " juta" if x >= 1e6 else fmt_int(x)


def minmax(s):
    rng = s.max() - s.min()
    return (s - s.min()) / rng if rng else s * 0


def weighted_poverty(df):
    """Kemiskinan berbobot penduduk (setara angka nasional BPS), bukan rata-rata sederhana."""
    tot = df["jumlah_jiwa"].sum()
    return float((df["pct_miskin"] * df["jumlah_jiwa"]).sum() / tot) if tot else np.nan


def per100k(df):
    tot = df["jumlah_jiwa"].sum()
    return float(df["kejadian"].sum() / tot * 1e5) if tot else np.nan


def spearman(a, b):
    d = pd.concat([a, b], axis=1).dropna()
    if len(d) < 3:
        return np.nan
    return float(d.iloc[:, 0].rank().corr(d.iloc[:, 1].rank()))


def kekuatan(r):
    a = abs(r)
    return "sangat lemah" if a < 0.2 else "lemah" if a < 0.4 else "sedang" if a < 0.6 else "kuat"


def process_data(m):
    """Bersihkan, hitung metrik turunan, dan beri nama klaster berdasarkan isinya."""
    n_dup = int(m.duplicated("key").sum())
    m = m.drop_duplicates("key").reset_index(drop=True)
    for c in ["kejadian", "rumah_rusak"]:
        m[c] = m[c].fillna(0)

    pop = m["jumlah_jiwa"].where(m["jumlah_jiwa"] > 0)
    m["kejadian_per100k"] = m["kejadian"] / pop * 1e5
    m["rumah_rusak_per100k"] = m["rumah_rusak"] / pop * 1e5
    m["penduduk_miskin_est"] = pop * m["pct_miskin"] / 100

    # K-Means dihitung di sini dengan ukuran per penduduk, supaya klaster sejalan dengan skor kerentanan
    if "cluster_label" not in m.columns:
        m["cluster_label"] = np.nan
    fitur = pd.DataFrame({
        "a": np.log1p(m["kejadian_per100k"]),
        "b": np.log1p(m["rumah_rusak_per100k"]),
        "c": m["pct_miskin"],
    })
    ok = fitur.notna().all(axis=1)
    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        from sklearn.preprocessing import StandardScaler

        X = StandardScaler().fit_transform(fitur[ok])
        sil = {k: silhouette_score(X, KMeans(k, n_init=10, random_state=42).fit_predict(X)) for k in (3, 4, 5)}
        k_best = max(sil, key=sil.get)
        lab = KMeans(k_best, n_init=10, random_state=42).fit_predict(X)
        etetik = pd.Series(index=m.index, dtype="object")
        etetik[ok] = [f"K{c + 1} · klaster" for c in lab]
        m["cluster_label"] = etetik
    except ImportError:
        pass  # tanpa scikit-learn: pakai cluster_label dari CSV bila ada

    # Nama klaster dibuat ulang dari rata-rata tiap klaster, agar label selalu cocok dengan isinya
    m["cid"] = m["cluster_label"].astype(str).str.extract(r"^(K\d+)")[0]
    if m["cid"].notna().any():
        m["_lk"] = np.log1p(m["kejadian_per100k"])
        mu_k, sd_k = m["_lk"].mean(), m["_lk"].std()
        mu_p, sd_p = m["pct_miskin"].mean(), m["pct_miskin"].std()
        g = m.groupby("cid").agg(lk=("_lk", "mean"), p=("pct_miskin", "mean"))
        nama = {}
        for cid, r in g.iterrows():
            rawan = r["lk"] > mu_k + 0.25 * sd_k
            miskin = r["p"] > mu_p + 0.25 * sd_p
            tipe = {
                (True, True): "Rawan dan miskin",
                (True, False): "Rawan, relatif mampu",
                (False, True): "Miskin, bencana tercatat rendah",
                (False, False): "Relatif aman",
            }[(rawan, miskin)]
            nama[cid] = f"{cid} · {tipe}"
        m["cluster_label"] = m["cid"].map(nama).fillna(m["cluster_label"])
        m = m.drop(columns=["_lk"])

    # Skor kerentanan dan kuadran prioritas (bencana per penduduk x kemiskinan)
    m["idx_bencana"] = minmax(np.log1p(m["kejadian_per100k"]))
    m["idx_miskin"] = minmax(m["pct_miskin"])
    m["skor"] = 0.5 * m["idx_bencana"] + 0.5 * m["idx_miskin"]
    hi_b = m["kejadian_per100k"] > m["kejadian_per100k"].median()
    hi_m = m["pct_miskin"] > m["pct_miskin"].median()
    m["kuadran"] = np.select(
        [hi_b & hi_m, hi_b & ~hi_m, ~hi_b & hi_m],
        ["Prioritas utama", "Rawan bencana", "Miskin"],
        default="Relatif aman",
    )

    # Teks siap tampil untuk tooltip peta
    m["t_jiwa"] = m["jumlah_jiwa"].map(lambda v: fmt_int(v) if pd.notna(v) else "-")
    m["t_miskin"] = m["pct_miskin"].map(lambda v: fmt_dec(v) + "%")
    m["t_kejadian"] = m["kejadian"].map(fmt_int)
    m["t_per100k"] = m["kejadian_per100k"].map(lambda v: fmt_dec(v, 1) if pd.notna(v) else "-")
    m["t_usia"] = m["usia_produktif_pct"].map(lambda v: fmt_dec(v, 1) + "%" if pd.notna(v) else "-")
    return m, n_dup


# ---------------------------------------------------------------- Load data
@st.cache_data
def load_data():
    m = pd.read_csv("output/kabkota.csv")
    missing = [c for c in REQUIRED if c not in m.columns]
    if missing:
        return None, None, 0, missing
    m, n_dup = process_data(m)

    gdf = gpd.read_file("output/kabupaten_klaster_bencana_kemiskinan.geojson")
    if gdf["key"].duplicated().any():
        gdf = gdf.dissolve(by="key", as_index=False)
    gdf = gdf[["key", "geometry"]]
    return m, gdf, n_dup, []


@st.cache_data
def model_metrics(df):
    """Hitung metrik validasi klaster dari data, bukan angka tetap."""
    try:
        from sklearn.metrics import (
            calinski_harabasz_score,
            davies_bouldin_score,
            silhouette_score,
        )
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return None
    d = df.dropna(subset=["cid", "kejadian_per100k", "rumah_rusak_per100k", "pct_miskin"])
    if d["cid"].nunique() < 2:
        return None
    X = StandardScaler().fit_transform(
        np.column_stack([np.log1p(d["kejadian_per100k"]), np.log1p(d["rumah_rusak_per100k"]), d["pct_miskin"]])
    )
    y = d["cid"].values
    return (
        silhouette_score(X, y),
        davies_bouldin_score(X, y),
        calinski_harabasz_score(X, y),
    )


m, gdf, n_dup, missing_cols = load_data()
if missing_cols:
    st.error(
        "Kolom berikut belum ada di output/kabkota.csv: "
        + ", ".join(missing_cols)
        + ". Jalankan tambah_penduduk.py terlebih dahulu agar jumlah_jiwa dan"
        + " usia_produktif_pct berasal dari data BPS, bukan angka acak."
    )
    st.stop()

cl_order = sorted(m["cluster_label"].dropna().unique())
cl_scale = alt.Scale(domain=cl_order, range=PALETTE[: len(cl_order)])

# Header Utama (Diperbarui menjadi bahasa yang lebih sederhana)
st.markdown(
    '<p class="main-header">Peta Interaktif Risiko Bencana & Kemiskinan di Indonesia</p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="sub-header">Analisis Wilayah Rawan Bencana Berbasis Data BPS & BNPB</p>',
    unsafe_allow_html=True,
)

# --- SIDEBAR FILTER PROFESIONAL & METRIK MODEL ---
st.sidebar.markdown("### Panel Kontrol & Filter Data")

selected_prov = st.sidebar.selectbox(
    "Provinsi", ["Semua Provinsi"] + sorted(m["provinsi"].dropna().unique().tolist())
)

selected_cluster = st.sidebar.selectbox(
    "Kluster K-Means",
    ["Semua Kluster"] + cl_order,
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Filter Rentang Analisis")

min_miskin, max_miskin = float(m["pct_miskin"].min()), float(m["pct_miskin"].max())
selected_poverty_range = st.sidebar.slider(
    "Rentang Persentase Kemiskinan (%)",
    min_miskin,
    max_miskin,
    (min_miskin, max_miskin),
)

max_kejadian = int(m["kejadian"].max())
selected_disaster_range = st.sidebar.slider(
    "Rentang Total Kejadian Bencana", 0, max_kejadian, (0, max_kejadian)
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Validasi Model K-Means")
mm = model_metrics(m)
if mm:
    st.sidebar.info(
        f"**Silhouette Score:** {mm[0]:.3f}\n\n"
        f"**Davies-Bouldin Index:** {mm[1]:.3f}\n\n"
        f"**Calinski-Harabasz:** {mm[2]:.2f}"
    )
if n_dup:
    st.sidebar.caption(f"{n_dup} baris duplikat (kunci wilayah sama) dihapus saat pemuatan data.")

# Filter Logika Data Utama
filtered_m = m.copy()
if selected_prov != "Semua Provinsi":
    filtered_m = filtered_m[filtered_m["provinsi"] == selected_prov]
if selected_cluster != "Semua Kluster":
    filtered_m = filtered_m[filtered_m["cluster_label"] == selected_cluster]

filtered_m = filtered_m[
    (filtered_m["pct_miskin"] >= selected_poverty_range[0])
    & (filtered_m["pct_miskin"] <= selected_poverty_range[1])
    & (filtered_m["kejadian"] >= selected_disaster_range[0])
    & (filtered_m["kejadian"] <= selected_disaster_range[1])
]

if filtered_m.empty:
    st.warning("Tidak ada kabupaten/kota yang sesuai dengan filter. Longgarkan filter di panel kiri.")
    st.stop()

# Sinkronisasi Spasial GeoJSON
geo_cols = [
    "key", "nama", "provinsi", "pct_miskin", "cluster_label",
    "t_jiwa", "t_miskin", "t_kejadian", "t_per100k", "t_usia",
]
filtered_gdf = gdf.merge(filtered_m[geo_cols], on="key", how="inner")

# --- TOP METRICS / KPI CARDS ---
kpis = [
    ("Total Wilayah", fmt_int(len(filtered_m))),
    ("Total Penduduk (Jiwa)", fmt_ringkas(filtered_m["jumlah_jiwa"].sum())),
    ("Kemiskinan (Berbobot Penduduk)", fmt_dec(weighted_poverty(filtered_m)) + "%"),
    ("Estimasi Penduduk Miskin (Jiwa)", fmt_ringkas(filtered_m["penduduk_miskin_est"].sum())),
    ("Total Rumah Rusak", fmt_int(filtered_m["rumah_rusak"].sum())),
    ("Total Kejadian Bencana", fmt_int(filtered_m["kejadian"].sum())),
]
for col, (label, val) in zip(st.columns(len(kpis)), kpis):
    col.markdown(
        f'<div class="kpi-container"><div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{val}</div></div>',
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# --- TAB NAVIGASI UTAMA ---
tab1, tab2, tab3 = st.tabs([
    "Peta & Rekomendasi Eksekutif",
    "Analisis Korelasi & Statistik",
    "Tentang Data & Metodologi",
])

with tab1:
    map_col, info_col = st.columns([1.3, 0.7])

    with map_col:
        with st.container(border=True):
            st.markdown(
                "<h5>Peta Sebaran Kemiskinan & Wilayah Rawan</h5>",
                unsafe_allow_html=True,
            )

            m_map = folium.Map(
                location=[-2.548926, 118.014863], zoom_start=4.5, tiles="OpenStreetMap"
            )

            folium.Choropleth(
                geo_data=filtered_gdf.__geo_interface__,
                data=filtered_m,
                columns=["key", "pct_miskin"],
                key_on="feature.properties.key",
                fill_color="YlOrRd",
                fill_opacity=0.75,
                line_opacity=0.3,
                legend_name="Persentase Penduduk Miskin (%)",
            ).add_to(m_map)

            folium.GeoJson(
                filtered_gdf,
                style_function=lambda x: {
                    "fillColor": "transparent",
                    "color": "transparent",
                    "weight": 0,
                },
                tooltip=folium.GeoJsonTooltip(
                    fields=[
                        "nama", "provinsi", "t_jiwa", "t_usia",
                        "t_miskin", "t_kejadian", "t_per100k", "cluster_label",
                    ],
                    aliases=[
                        "Kab/Kota:", "Provinsi:", "Jumlah Penduduk (Jiwa):",
                        "Usia Produktif:", "Kemiskinan:", "Total Kejadian Bencana:",
                        "Kejadian per 100 Ribu Penduduk:", "Kluster K-Means:",
                    ],
                ),
            ).add_to(m_map)

            st_folium(m_map, width=700, height=450, use_container_width=True, returned_objects=[])

            pm = filtered_m.groupby("provinsi")["pct_miskin"].mean().dropna()
            if len(pm):
                st.markdown(
                    f'<div class="insight-box"><b>Insight:</b> pada cakupan terpilih, rata-rata'
                    f" kemiskinan kab/kota tertinggi ada di <b>{pm.idxmax()}</b>"
                    f" ({fmt_dec(pm.max(), 1)}%) dan terendah di <b>{pm.idxmin()}</b>"
                    f" ({fmt_dec(pm.min(), 1)}%).</div>",
                    unsafe_allow_html=True,
                )

    with info_col:
        with st.container(border=True):
            st.markdown(
                "<h5>Top 10 Kabupaten/Kota Prioritas Intervensi</h5>",
                unsafe_allow_html=True,
            )
            top = filtered_m.nlargest(10, "skor")[
                ["nama", "provinsi", "pct_miskin", "kejadian", "kejadian_per100k", "skor"]
            ].rename(columns={
                "nama": "Kab/Kota", "provinsi": "Provinsi", "pct_miskin": "% Miskin",
                "kejadian": "Kejadian", "kejadian_per100k": "Kejadian/100 rb", "skor": "Skor",
            }).round(2)
            st.dataframe(top, hide_index=True, height=430)
            st.caption("Skor = 0,5 x indeks bencana per penduduk + 0,5 x indeks kemiskinan (0-1).")

    # --- BARIS KEDUA: BENCHMARK & REKOMENDASI KEBIJAKAN OTOMATIS ---
    bench_col, rec_col = st.columns([1, 1])

    with bench_col:
        with st.container(border=True):
            st.markdown(
                "<h5>Perbandingan & Pembagian Kelompok Wilayah</h5>",
                unsafe_allow_html=True,
            )

            nat_poverty = weighted_poverty(m)
            nat_disaster = per100k(m)
            curr_poverty = weighted_poverty(filtered_m)
            curr_disaster = per100k(filtered_m)
            warna_m = "#dc2626" if curr_poverty > nat_poverty else "#16a34a"
            warna_b = "#dc2626" if curr_disaster > nat_disaster else "#16a34a"
            cakupan = (
                f"{selected_prov} ({len(filtered_m)} Kab/Kota)"
                if selected_prov != "Semua Provinsi"
                else f"Seluruh Indonesia ({len(filtered_m)} Kab/Kota)"
            )

            st.markdown(
                f"""
                <div class="insight-box">
                    <b>Cakupan Wilayah Aktif:</b> {cakupan}<br>
                    <b>Tingkat Kemiskinan:</b> <span style="color: {warna_m}">{fmt_dec(curr_poverty)}%</span> (Nasional: {fmt_dec(nat_poverty)}%)<br>
                    <b>Kejadian Bencana per 100 Ribu Penduduk:</b> <span style="color: {warna_b}">{fmt_dec(curr_disaster, 1)}</span> (Nasional: {fmt_dec(nat_disaster, 1)})
                </div>
                """,
                unsafe_allow_html=True,
            )

            cluster_counts = filtered_m["cluster_label"].value_counts().reset_index()
            cluster_counts.columns = ["Kluster", "Jumlah"]
            chart = (
                alt.Chart(cluster_counts)
                .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
                .encode(
                    y=alt.Y("Kluster:N", sort="-x", title=None, axis=alt.Axis(labelLimit=280)),
                    x=alt.X("Jumlah:Q", title="Jumlah Wilayah"),
                    color=alt.Color("Kluster:N", scale=cl_scale, legend=None),
                    tooltip=["Kluster", "Jumlah"],
                )
                .properties(height=210)
            )
            st.altair_chart(chart, use_container_width=True)
            terbesar = cluster_counts.iloc[0]
            st.markdown(
                f'<div class="insight-box"><b>Insight:</b> klaster terbesar adalah'
                f" <b>{terbesar['Kluster']}</b> dengan {terbesar['Jumlah']} wilayah"
                f" ({terbesar['Jumlah'] / len(filtered_m) * 100:.0f}% dari cakupan terpilih).</div>",
                unsafe_allow_html=True,
            )

    with rec_col:
        with st.container(border=True):
            st.markdown(
                "<h5>Rekomendasi Tindakan & Prioritas Penanganan</h5>",
                unsafe_allow_html=True,
            )

            n_all = len(filtered_m)
            prio = filtered_m[filtered_m["kuadran"] == "Prioritas utama"]
            n_prio = len(prio)
            n_rendah = int(filtered_m["cluster_label"].astype(str).str.contains("bencana tercatat rendah").sum())

            if n_prio > 0:
                top3 = ", ".join(prio.nlargest(3, "skor")["nama"].tolist())
                est = fmt_int(prio["penduduk_miskin_est"].sum())
                st.markdown(
                    f"""
                    <div class="priority-box">
                        <b>Prioritas Utama - Mitigasi dan Perlindungan Sosial:</b>
                        {n_prio} dari {n_all} kab/kota ({n_prio / n_all * 100:.0f}%) berada di kuadran
                        prioritas utama, yaitu kemiskinan dan bencana per penduduk sama-sama di atas
                        median nasional. Estimasi penduduk miskin di daerah ini sekitar {est} jiwa.
                        Skor tertinggi: {top3}. Dahulukan anggaran mitigasi fisik, sistem peringatan
                        dini, dan bantuan sosial adaptif di daerah-daerah ini.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    """
                    <div class="recommendation-box">
                        <b>Pemantauan dan Penguatan Kapasitas:</b> tidak ada kab/kota pada cakupan ini
                        yang kemiskinan dan bencana per penduduknya sama-sama di atas median nasional.
                        Pertahankan pemantauan rutin dan perkuat kesiapsiagaan daerah.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            if n_rendah > 0:
                st.markdown(
                    f"""
                    <div class="warning-box">
                        <b>Catatan Data:</b> {n_rendah} kab/kota termasuk klaster miskin dengan bencana
                        tercatat rendah. Rendahnya kejadian dapat mencerminkan keterbatasan pencatatan
                        DIBI, bukan berarti daerah itu aman. Prioritaskan verifikasi lapangan dan
                        penguatan pelaporan bencana, di samping pengentasan kemiskinan.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            csv_data = (
                filtered_m.drop(columns=[c for c in filtered_m.columns if c.startswith("t_")])
                .to_csv(index=False)
                .encode("utf-8")
            )
            st.download_button(
                label="Unduh Data Terfilter & Rekomendasi (CSV)",
                data=csv_data,
                file_name="data_analisis_bencana_kemiskinan_filtered.csv",
                mime="text/csv",
                use_container_width=True,
            )

with tab2:
    with st.container(border=True):
        st.markdown(
            "<h5>Analisis Korelasi: Frekuensi Bencana vs Persentase Kemiskinan</h5>",
            unsafe_allow_html=True,
        )

        pilihan_y = st.radio(
            "Ukuran bencana (sumbu Y)",
            ["Kejadian per 100 ribu penduduk", "Total kejadian"],
            horizontal=True,
        )
        ycol = "kejadian_per100k" if pilihan_y.startswith("Kejadian per") else "kejadian"
        plot_df = filtered_m.dropna(subset=[ycol, "pct_miskin"])

        base = alt.Chart(plot_df)
        points = base.mark_circle(opacity=0.7).encode(
            x=alt.X("pct_miskin:Q", title="Persentase Penduduk Miskin (%)"),
            y=alt.Y(f"{ycol}:Q", title=pilihan_y + " (skala simetris-log)", scale=alt.Scale(type="symlog")),
            size=alt.Size("jumlah_jiwa:Q", legend=None, scale=alt.Scale(range=[25, 500])),
            color=alt.Color("cluster_label:N", title="Kluster K-Means", scale=cl_scale),
            tooltip=[
                alt.Tooltip("nama:N", title="Kab/Kota"),
                alt.Tooltip("provinsi:N", title="Provinsi"),
                alt.Tooltip("pct_miskin:Q", title="% Miskin", format=".2f"),
                alt.Tooltip("kejadian:Q", title="Total Kejadian"),
                alt.Tooltip("kejadian_per100k:Q", title="Kejadian/100 rb", format=".1f"),
                alt.Tooltip("jumlah_jiwa:Q", title="Penduduk", format=","),
                alt.Tooltip("cluster_label:N", title="Kluster"),
            ],
        )
        vline = alt.Chart(pd.DataFrame({"x": [float(m["pct_miskin"].median())]})).mark_rule(
            strokeDash=[5, 5], color="#94a3b8"
        ).encode(x="x:Q")
        hline = alt.Chart(pd.DataFrame({"y": [float(m[ycol].median())]})).mark_rule(
            strokeDash=[5, 5], color="#94a3b8"
        ).encode(y="y:Q")
        st.altair_chart(
            alt.layer(points, vline, hline).properties(height=420).interactive(),
            use_container_width=True,
        )
        st.caption(
            "Garis putus-putus = median nasional. Kuadran kanan atas adalah daerah miskin dan"
            " sering terdampak bencana. Ukuran titik = jumlah penduduk."
        )

        rho = spearman(plot_df["pct_miskin"], plot_df[ycol])
        c1, c2, c3 = st.columns(3)
        c1.metric("Korelasi Spearman (rho)", fmt_dec(rho) if pd.notna(rho) else "-")
        c2.metric("Kab/kota prioritas utama", fmt_int((filtered_m["kuadran"] == "Prioritas utama").sum()))
        c3.metric("Jumlah data dianalisis", fmt_int(len(plot_df)))
        if pd.notna(rho):
            st.markdown(
                f'<div class="insight-box"><b>Insight:</b> hubungan kemiskinan dan {pilihan_y.lower()}'
                f" bersifat <b>{kekuatan(rho)}</b> (rho = {fmt_dec(rho)}). Frekuensi bencana banyak"
                " ditentukan faktor alam, sehingga bencana tidak hanya menimpa daerah miskin. Namun"
                " daerah yang menanggung keduanya sekaligus tetap perlu diprioritaskan.</div>",
                unsafe_allow_html=True,
            )

with tab3:
    with st.container(border=True):
        st.markdown(
            "<h5>Tentang Sumber Data & Metodologi Analisis</h5>",
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <p><b>1. Sumber Data Utama:</b></p>
            <ul>
                <li><b>BPS (Badan Pusat Statistik):</b> persentase penduduk miskin, jumlah penduduk miskin, dan jumlah penduduk menurut kelompok umur per kabupaten/kota. Jumlah penduduk dan persentase usia produktif (15-64 tahun) dihitung dari tabel kelompok umur BPS.</li>
                <li><b>DIBI BNPB:</b> rekapitulasi kejadian bencana, korban, dan kerusakan rumah per kabupaten/kota. Kabupaten/kota tanpa catatan dianggap 0 kejadian.</li>
                <li><b>GADM Level 2:</b> batas poligon administratif kabupaten/kota untuk peta choropleth.</li>
            </ul>
            <p><b>2. Metodologi:</b></p>
            <ul>
                <li><b>K-Means:</b> fitur berupa logaritma kejadian bencana per 100 ribu penduduk, logaritma kerusakan rumah per 100 ribu penduduk, dan persentase kemiskinan, yang distandarkan sebelum pengelompokan. Jumlah klaster (3-5) dipilih dari silhouette score tertinggi. Nama klaster ditentukan dengan membandingkan rata-rata tiap klaster terhadap rata-rata nasional (selisih lebih dari 0,25 simpangan baku dianggap tinggi).</li>
                <li><b>Skor kerentanan:</b> 0,5 x indeks bencana per 100 ribu penduduk + 0,5 x indeks kemiskinan (skala min-max). Kuadran prioritas utama = kemiskinan dan bencana per penduduk sama-sama di atas median.</li>
                <li><b>Kemiskinan agregat</b> dihitung berbobot jumlah penduduk, bukan rata-rata sederhana antar-kabupaten/kota.</li>
                <li><b>Estimasi penduduk miskin</b> = jumlah penduduk x persentase penduduk miskin.</li>
            </ul>
            <p><b>3. Keterbatasan:</b></p>
            <ul>
                <li>Pencatatan DIBI tidak merata antar-daerah, sehingga kejadian rendah tidak selalu berarti aman.</li>
                <li>Korelasi tidak membuktikan hubungan sebab-akibat.</li>
                <li>Data kelompok umur menunjukkan siapa yang terpapar, bukan siapa yang benar-benar terdampak.</li>
            </ul>
            """,
            unsafe_allow_html=True,
        )

with st.container(border=True):
    st.markdown(
        "<h5>Eksplorasi Data Spasial & Atribut Wilayah</h5>", unsafe_allow_html=True
    )

    search_query = st.text_input("Cari Nama Kabupaten / Kota / Provinsi:", "")

    display_df = filtered_m[[
        "nama", "provinsi", "cluster_label", "pct_miskin", "kejadian",
        "kejadian_per100k", "rumah_rusak", "jumlah_jiwa", "penduduk_miskin_est",
        "usia_produktif_pct",
    ]]

    if search_query:
        display_df = display_df[
            display_df["nama"].str.contains(search_query, case=False, na=False)
            | display_df["provinsi"].str.contains(search_query, case=False, na=False)
        ]

    display_df = display_df.rename(columns={
        "nama": "Kab/Kota",
        "provinsi": "Provinsi",
        "cluster_label": "Kluster",
        "pct_miskin": "% Miskin",
        "kejadian": "Kejadian",
        "kejadian_per100k": "Kejadian/100 rb",
        "rumah_rusak": "Rumah Rusak",
        "jumlah_jiwa": "Penduduk",
        "penduduk_miskin_est": "Est. Penduduk Miskin",
        "usia_produktif_pct": "% Usia Produktif",
    }).round({"% Miskin": 2, "Kejadian/100 rb": 1, "Est. Penduduk Miskin": 0, "% Usia Produktif": 1})

    st.dataframe(display_df, hide_index=True, use_container_width=True, height=300)
