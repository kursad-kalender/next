import streamlit as st
import pandas as pd 
import numpy as np 
import requests
from st_aggrid import AgGrid, GridOptionsBuilder
import geopandas as gpd
from shapely.geometry import Point
import plotly.graph_objects as go

import httpx
import asyncio
import nest_asyncio
from folium.plugins import Fullscreen
import time
import folium
from collections import defaultdict
from streamlit_extras.switch_page_button import switch_page
import itertools
nest_asyncio.apply()

st.set_page_config(
    page_title="UrClimate Next",
    page_icon="🌏",
    layout="wide",
)
pd.set_option('future.no_silent_downcasting', True) # pandas sürümünden olduğuny tahmin ettiğim terminale print edilen hatayı muteleyen option. 
API_URL = st.secrets["API_URL"]
ENDPOINTS = ["point-fd-score", "point-ndd-score", "point-txge35-score", "point-high-fwi-days-score", "point-sfcwind-over-p99-score", "point-pr-over-p95-score"]

acute_threshold = 30
chronic_threshold = 40
#==================================
acute_risk_endpoints = [
    "point-high-fwi-days-score",
    "point-sfcwind-over-p99-score",
    "point-pr-over-p95-score"
]

chronic_risk_endpoints = [
    "point-ndd-score",
    "point-txge35-score",
    "point-fd-score"
]

endpoint_label_map = {
    "point-high-fwi-days-score": "Orman Yangını",
    "point-pr-over-p95-score": "Sel",
    "point-sfcwind-over-p99-score": "Fırtına",
    "point-ndd-score": "Kuraklık",
    "point-txge35-score": "Sıcak Hava Dalgası",
    "point-fd-score": "Deniz Seviyesi Yükselmesi"

}
term_map_short = {
    "Short Term": "KV",
    "Mid Term": "OV",
    "Long Term": "UV"
}
#==================================
# =================================
async def fetch_endpoint(client, url, params, asset_id, endpoint):
    try:
        response = await client.get(url, params=params)
        response.raise_for_status()
        return (asset_id, endpoint, response.json())
    except Exception as e:
        return (asset_id, endpoint, {"error": str(e)})
    
async def fetch_endpoints_for_asset(asset, endpoints, api_url, client):
    tasks = []
    for endpoint in endpoints:
        url = f"{api_url}{endpoint}"
        params = {"latitude": asset["latitude"], "longitude": asset["longitude"]}
        tasks.append(fetch_endpoint(client, url, params, asset["asset_id"], endpoint))
    return await asyncio.gather(*tasks)


def process_assets_with_async_progress(asset_df, API_URL):
    first_exceedance_hazard_records = []
    asset_records = asset_df.to_dict(orient="records")
    total_assets = len(asset_records)
    raw_scores = []

    hazard_records = {label: [] for label in endpoint_label_map.values()}
    asset_risk_summaries = []

    progress_bar = st.progress(0)
    status_text = st.empty()

    async def run_all():
        
        async with httpx.AsyncClient(timeout=30) as client:
            for i, asset in enumerate(asset_records):
                status_text.text(f"{i+1}/{total_assets} → İşleniyor: {asset['asset_id']}")
                responses = await fetch_endpoints_for_asset(asset, ENDPOINTS, API_URL, client)

                endpoint_data = {ep: data for _, ep, data in responses if data and "error" not in data}
                vulnerability = get_vulnerability(asset["nace_code"])

                for scenario_idx, scenario in enumerate(["ssp245", "ssp585"]):
                    acute_risks = []
                    chronic_risks = []

                    for endpoint, data in endpoint_data.items():
                        df = pd.DataFrame(data)
                        df["dates"] = pd.to_datetime(df["dates"], errors="coerce")
                        df = df.set_index("dates")

                        try:
                            df_scenario = df.iloc[:, [scenario_idx]]
                        except IndexError:
                            continue  # Veri yoksa geç

                        short_term_risk = df_scenario.loc["2021":"2040"].mean() * vulnerability
                        mid_term_risk   = df_scenario.loc["2041":"2060"].mean() * vulnerability
                        long_term_risk  = df_scenario.loc["2081":].mean() * vulnerability

                        df_avg = pd.DataFrame({
                            "Short Term": short_term_risk,
                            "Mid Term": mid_term_risk,
                            "Long Term": long_term_risk
                        })

                        if endpoint in acute_risk_endpoints:
                            threshold = acute_threshold
                            risks_list = acute_risks
                        else:
                            threshold = chronic_threshold
                            risks_list = chronic_risks

                        if threshold is not None:
                            yearly_avg = df_scenario.loc["2025":"2100"].resample("YE").mean().copy()
                            yearly_avg["Year"] = yearly_avg.index.year
                            above_thresh = yearly_avg[yearly_avg.iloc[:, 0] > threshold]

                            if not above_thresh.empty:
                                first_year = int(above_thresh.index.year[0])
                                first_value = above_thresh.iloc[0, 0]
                                first_value_str = f"{first_value:.1f}%"
                                if 2021 <= first_year <= 2040:
                                    belonging_term = "Short Term"
                                elif 2041 <= first_year <= 2060:
                                    belonging_term = "Mid Term"
                                elif first_year >= 2061:
                                    belonging_term = "Long Term"
                                else:
                                    belonging_term = None
                            else:
                                first_year = None
                                first_value_str = None

                            first_exceedance_hazard_records.append({
                                "Asset ID": asset["asset_id"],
                                "Loan ID": asset["loan_id"],
                                "Hazard": endpoint_label_map.get(endpoint, endpoint),
                                "Scenario": scenario,
                                "First Exceedance Year": first_year,
                                "Belonging Term": belonging_term,
                                "Exceedance Rate": first_value_str,
                                "Exposure Amount": asset["exposure_amount"]
                            })

                        terms_exceeding = [
                            term for term in ["Short Term", "Mid Term", "Long Term"]
                            if df_avg[term].values[0] > threshold
                        ]

                        if terms_exceeding:
                            label = endpoint_label_map.get(endpoint, endpoint)
                            risks_list.append(f"{label}({', '.join(terms_exceeding)})")


                        if endpoint in acute_risk_endpoints or endpoint in chronic_risk_endpoints:
                            label = endpoint_label_map.get(endpoint, endpoint)
                            hazard_records[label].append({
                                "Asset ID": asset["asset_id"],
                                "Latitude": asset["latitude"],
                                "Longitude": asset["longitude"],
                                "Scenario": scenario,
                                "Short Term": float(short_term_risk.values[0]),
                                "Mid Term": float(mid_term_risk.values[0]),
                                "Long Term": float(long_term_risk.values[0])
                            })

                    # HER senaryo için ayrı kayıt!
                    asset_risk_summaries.append({
                        "asset_id": asset["asset_id"],
                        "loan_id": asset["loan_id"],
                        "scenario": scenario,
                        "exposure_amount": asset["exposure_amount"],
                        "acute_risks": acute_risks,
                        "chronic_risks": chronic_risks,
                        "nace_code": asset["nace_code"],
                        "sector": get_sector_name(asset["nace_code"])
                    })

                    for endpoint, data in endpoint_data.items():
                        df = pd.DataFrame(data)
                        df["dates"] = pd.to_datetime(df["dates"], errors="coerce")
                        df = df.set_index("dates")

                        for scenario_idx, scenario in enumerate(["ssp245", "ssp585"]):
                            try:
                                df_scenario = df.iloc[:, [scenario_idx]].copy()
                            except IndexError:
                                continue

                            df_scenario.columns = ["score"]
                            df_scenario["Asset ID"] = asset["asset_id"]
                            df_scenario["Scenario"] = scenario
                            df_scenario["Endpoint"] = endpoint
                            df_scenario["Date"] = df_scenario.index
                            df_scenario["NACE Code"] = asset["nace_code"]
                            df_scenario["Exposure Amount"] = asset["exposure_amount"]

                            raw_scores.append(df_scenario.reset_index(drop=True))
                progress_bar.progress((i + 1) / total_assets)

        progress_bar.empty()
        status_text.empty()

    asyncio.run(run_all())

    return hazard_records, asset_risk_summaries, pd.DataFrame(first_exceedance_hazard_records), pd.concat(raw_scores, ignore_index=True)


def build_tooltip(joined_df, term="Short Term"):
    tooltip_df = (
        joined_df
        .drop(columns=["GID_2"])  # groupby sütununu çıkar
        .groupby(joined_df["GID_2"], group_keys=False)
        .apply(lambda g: "\n".join([
            f"{row['Asset ID']}: {row[term]:.2f}" for _, row in g.iterrows()
        ]))
        .reset_index(name="Contributors")
    )
    return tooltip_df

def prepare_district_map_data(districts, joined_df, term_col):
    tooltip_df = build_tooltip(joined_df, term=term_col)

    avg_df = (
        joined_df
        .groupby("GID_2", as_index=False)[term_col]
        .mean()
        .round(2)
        .rename(columns={term_col: f"{term_col} Average"})
    )

    merged = (
        districts
        .merge(tooltip_df, on="GID_2", how="left")
        .merge(avg_df, on="GID_2", how="left")
        .rename(columns={
            "NAME_1": "Province",
            "NAME_2": "District"
        })
    )

    return merged

def prepare_flag_based_map(term, flag_column, joined_scenario):
    try:
        joined_scenario[flag_column] = joined_scenario[flag_column].fillna(False)
        exceeding_assets = joined_scenario[joined_scenario[flag_column]]

        district_counts = (
            exceeding_assets
            .groupby("GID_2")
            .agg(
                exceed_count=("Asset ID", "count"),
                exceed_ids=("Asset ID", lambda x: ", ".join(x.astype(str)))
            )
            .reset_index()
        )

        districts_with_flags = districts.merge(district_counts, on="GID_2", how="left")
        districts_with_flags["exceed_count"] = districts_with_flags["exceed_count"].fillna(0).astype(int)
        districts_with_flags["exceed_ids"] = districts_with_flags["exceed_ids"].fillna("Yok")

        districts_with_flags = districts_with_flags.rename(columns={
            "NAME_1": "İl",
            "NAME_2": "İlçe",
            "exceed_count": "Aşan Asset Sayısı",
            "exceed_ids": "Aşan Asset ID'leri"
        })

        m = districts_with_flags.explore(
            column="Aşan Asset Sayısı",
            cmap="plasma", 
            legend=True,
            tooltip=["İl", "İlçe", "Aşan Asset Sayısı", "Aşan Asset ID'leri"],
            style_kwds={
                "weight": 0.3,
                "color": "#333333",
                "fillOpacity": 0.9
            },
            zoom_start=6,
            highlight=True,
            location=[35.0, 35.0],
            control_scale=False
        )

        m.options["minZoom"] = 5
        m.options["maxZoom"] = 8
        Fullscreen().add_to(m)

        return m

    except KeyError:
        st.warning(f"`{flag_column}` sütunu mevcut değil, bu harita için veri üretilemedi.")
        return None
    
def get_vulnerability(nace_code: str) -> float:
    nace_vulnerability_map = {
        "A": 0.8,   # Tarım, ormancılık ve balıkçılık
        "B": 0.6,   # Madencilik ve taş ocakçılığı
        "C": 0.5,   # İmalat
        "D": 0.4,   # Elektrik, gaz, buhar ve iklimlendirme
        "E": 0.7,   # Su temini, kanalizasyon vb.
        "F": 0.75,  # İnşaat
        "G": 0.65,  # Toptan ve perakende ticaret
        "H": 0.6,   # Ulaştırma ve depolama
        "I": 0.7    # Konaklama ve yiyecek hizmetleri, Turizm
    }

    if not isinstance(nace_code, str) or len(nace_code) == 0:
        return None  # veya varsayılan bir değer (örneğin 0.5)

    first_letter = nace_code.strip().upper()[0]
    return nace_vulnerability_map.get(first_letter, None)

def get_sector_name(nace_code: str) -> str:
    nace_sector_map = {
        "A": "Tarım, Ormancılık ve Balıkçılık",
        "B": "Madencilik ve Taş Ocakçılığı",
        "C": "İmalat",
        "D": "Elektrik, Gaz, Guhar ve İklimlendirme",
        "E": "Su Temini, Kanalizasyon, Atık Yönetimi",
        "F": "İnşaat",
        "G": "Toptan ve Perakende Ticaret",
        "H": "Ulaştırma ve Depolama",
        "I": "Konaklama ve Yiyecek Hizmetleri (Turizm)"
    }

    if not isinstance(nace_code, str) or len(nace_code) == 0:
        return "Bilinmeyen"

    first_letter = nace_code.strip().upper()[0]
    return nace_sector_map.get(first_letter, "Diğer veya Bilinmeyen")
# =================================

REQUIRED_COLUMNS = {"asset_id", "loan_id", "latitude", "longitude", "nace_code", "exposure_amount"}

st.title("UrClimate Next")
with st.container(border=True):
    if "asset_df" in st.session_state:
        asset_df = st.session_state.asset_df
        if st.button("Farklı bir CSV yükle"):
            st.session_state.clear()
            st.rerun()

        uploaded_columns = set(asset_df.columns)
        missing = REQUIRED_COLUMNS - uploaded_columns
        extra = uploaded_columns - REQUIRED_COLUMNS

        if missing or extra:
            if missing:
                st.error(f"❌ Eksik sütun(lar): {', '.join(missing)}")
            if extra:
                st.error(f"❌ Fazladan gelen geçersiz sütun(lar): {', '.join(extra)}")
            st.stop()

    else:
        uploaded_file = st.file_uploader("Varlık CSV dosyasını yükleyin", type=["csv"])
        if uploaded_file is None:
            st.info("Lütfen analiz için bir CSV dosyası yükleyin.")
            st.stop()
        try:
            asset_df = pd.read_csv(uploaded_file)

            uploaded_columns = set(asset_df.columns)
            missing = REQUIRED_COLUMNS - uploaded_columns
            extra = uploaded_columns - REQUIRED_COLUMNS

            if missing or extra:
                if missing:
                    st.error(f"❌ Eksik sütun(lar): {', '.join(missing)}")
                if extra:
                    st.error(f"❌ Fazladan gelen geçersiz sütun(lar): {', '.join(extra)}")
                st.stop()
            st.session_state.asset_df = asset_df    
            st.rerun()
        except Exception as e:
            st.error(f"CSV okunamadı: {e}")
            st.stop()
    


hazard_records = {label: [] for label in endpoint_label_map.values()}

    # =============
start_time = time.time() # API CALL BAŞLANGIC

if "hazard_records" not in st.session_state:
    with st.spinner("Veriler işleniyor..."):
        hazard_records, asset_risk_summaries, exceedance_df, raw_scores  = process_assets_with_async_progress(asset_df, API_URL)
        st.session_state.hazard_records = hazard_records
        st.session_state.asset_risk_summaries = asset_risk_summaries
        st.session_state.exceedance_df = exceedance_df
        st.session_state.raw_scores = raw_scores
else:
    hazard_records = st.session_state.hazard_records
    asset_risk_summaries = st.session_state.asset_risk_summaries
    exceedance_df = st.session_state.exceedance_df
    raw_scores = st.session_state.raw_scores

elapsed = time.time() - start_time

mins, secs = divmod(elapsed, 60)
st.markdown(f"<span style='color:#65F527'>✅ Veri çekme süresi: {int(mins)} dk {secs:.0f} sn</span>", unsafe_allow_html=True)

    # =============
with st.container(border=True): 
    st.page_link("pages/search.py", label="🔎 Assetiniz için detayları görmek ister misiniz?")
                                                                                                                                                           
def detect_term(term_text, term):
    return term in term_text if isinstance(term_text, str) else False


with st.container(border = True):
    st.write("## ⚠️ Maruz Kalan Varlıkların Tanımı")
    summary_df = pd.DataFrame(asset_risk_summaries)

    summary_df["acute_risks"] = summary_df["acute_risks"].apply(lambda lst: ", ".join(lst) if lst else " ")

    summary_df["chronic_risks"] = summary_df["chronic_risks"].apply(lambda lst: ", ".join(lst) if lst else " ")

    summary_df["is_exposed_short_term"] = summary_df.apply( # --> Short Termde herhangi bir hazarda maruz mu?
        lambda row: detect_term(row["acute_risks"], "Short Term") or detect_term(row["chronic_risks"], "Short Term"),
        axis=1
    )
    summary_df["is_exposed_mid_term"] = summary_df.apply( # --> Mid Termde herhangi bir hazarda maruz mu?
        lambda row: detect_term(row["acute_risks"], "Mid Term") or detect_term(row["chronic_risks"], "Mid Term"),
        axis=1
    )
    summary_df["is_exposed_long_term"] = summary_df.apply( # --> Long Termde herhangi bir hazarda maruz mu?
        lambda row: detect_term(row["acute_risks"], "Long Term") or detect_term(row["chronic_risks"], "Long Term"),
        axis=1
    )

    summary_df["is_exposed"] = summary_df.apply(lambda row: row["acute_risks"] != " " or row["chronic_risks"] != " ",axis=1) # --> Genel olarak term bakmadan maruz mu?

    summary_df.index = range(1, len(summary_df) + 1)
    summary_df.rename(columns={"asset_id": "Asset ID", "loan_id": "Loan ID","scenario": "Scenario", "nace_code":"NACE", "exposure_amount":"Exposure Amount","acute_risks": "Acute Risks", "chronic_risks": "Chronic Risks", "is_exposed" : "Is Exposed?", "sector": "Sector"}, inplace=True)

    term_map = {
        "Short Term": "is_exposed_short_term",
        "Mid Term": "is_exposed_mid_term",
        "Long Term": "is_exposed_long_term"
    }

    term_dfs = []
    for term_label, flag_col in term_map.items():
        df_term = summary_df.copy()
        df_term["Term"] = term_label
        df_term["is_exposed_term"] = df_term[flag_col]
        term_dfs.append(df_term)

    term_exposure_df = pd.concat(term_dfs, ignore_index=True)

    # sektöre göre aggregation, ama term bazlı.
    sector_grouped_by_term = (
        term_exposure_df.groupby(["Scenario", "Sector", "Term"])
        .agg(
            total_exposure=("Exposure Amount", "sum"),
            exposed_exposure=("Exposure Amount", lambda x: x[term_exposure_df.loc[x.index, "is_exposed_term"]].sum())
        )
        .assign(exposure_share=lambda df: df.exposed_exposure / df.total_exposure)
        .rename(columns={
            "total_exposure": "Toplam Miktar",
            "exposed_exposure": "Maruz Kalan Miktar",
            "exposure_share": "Maruziyet Yüzdesi"
        })
        .reset_index()
    )
    sector_to_nace = summary_df[["Sector", "NACE"]].drop_duplicates(subset="Sector").set_index("Sector")
    sector_grouped_by_term["NACE"] = sector_grouped_by_term["Sector"].map(sector_to_nace["NACE"])
    cols = ["NACE"] + [col for col in sector_grouped_by_term.columns if col != "NACE"]
    sector_grouped_by_term = sector_grouped_by_term[cols]


    columns_to_hide = ["is_exposed_short_term", "is_exposed_mid_term", "is_exposed_long_term"]
    summary_df_display = summary_df.drop(columns=columns_to_hide)
    st.dataframe(summary_df_display.style.format({"Exposure Amount": "{:,.0f}"}),use_container_width=True,height=500)

    scenarios = summary_df["Scenario"].unique()
    for scenario in scenarios:
        st.markdown(f"### Senaryo: `{scenario}`")
        df_scenario = summary_df[summary_df["Scenario"] == scenario]
        exposed_assets = df_scenario["Is Exposed?"].sum()
        total_assets = len(df_scenario)
        exposed_amount = df_scenario.loc[df_scenario["Is Exposed?"], "Exposure Amount"].sum()
        total_amount = df_scenario["Exposure Amount"].sum()
        exposure_rate = exposed_amount / total_amount if total_amount > 0 else 0

        colL, colLM, colRM, colR = st.columns([2, 2, 2, 2])
        with colL:
            st.markdown(f"""<div style='text-align: center; font-size: 18px;'> ➡️  <strong>Exposed Assets:</strong><br>{exposed_assets}/{total_assets}</div>""",unsafe_allow_html=True)
        with colLM:
            st.markdown(f"""<div style='text-align: center; font-size: 18px;'> ➡️  <strong>Exposed Amount:</strong><br>{exposed_amount:,.2f}</div>""",unsafe_allow_html=True)
        with colRM:
            st.markdown(f"""<div style='text-align: center; font-size: 18px;'> ➡️  <strong>Total Amount:</strong><br>{total_amount:,.2f}</div>""",unsafe_allow_html=True)
        with colR:
            st.markdown(f"""<div style='text-align: center; font-size: 18px;'> ➡️  <strong>Exposure Rate:</strong><br>{exposure_rate:.1%}</div>""",unsafe_allow_html=True)

    st.divider()

    with st.container(border=True):
        left, center, right = st.columns([0.3, 3, 0.3])  # Ortayı geniş tuttuk
        with center:
            st.markdown("<h3 style='text-align: center;'>📊 Senaryo Bazlı Sektörel Maruziyet Özeti</h3>", unsafe_allow_html=True)

            col1, col2, col3 = st.columns([2.5, 3, 0.3])
            with col2:
                scenario_label_map = {"SSP245": "ssp245", "SSP585": "ssp585"}
                scenario_choice = st.radio("Senaryo Seçiniz:", options=list(scenario_label_map.keys()), horizontal=True, label_visibility="collapsed")
                selected_scenario = scenario_label_map[scenario_choice]

            # Seçili senaryoyu yeşil renkle ortada göster
            st.markdown(f"<p style='text-align: center; font-size: 16px;'>🔎 Seçili Senaryo: <strong><span style='color:#32CD32'>{scenario_choice}</span></strong></p>", unsafe_allow_html=True)


            term_tabs = st.tabs(["Short Term", "Mid Term", "Long Term"])

            for term, tab in zip(["Short Term", "Mid Term", "Long Term"], term_tabs):
                with tab:
                    filtered = sector_grouped_by_term[
                        (sector_grouped_by_term["Scenario"] == selected_scenario) &
                        (sector_grouped_by_term["Term"] == term)
                    ]

                    # ==== DNSH ===================
                    dnsh_nace_codes = ["A", "B", "D"]
                    dnsh_filtered = filtered[
                        (filtered["NACE"].isin(dnsh_nace_codes)) &
                        (filtered["Maruziyet Yüzdesi"] > 0.4)
                    ]
                    if not dnsh_filtered.empty:
                        unique_naces = dnsh_filtered["NACE"].unique()
                        message = [f"{nace}: '{get_sector_name(nace)}'" for nace in unique_naces]
                        st.warning(
                            f"🔴 {selected_scenario.upper()} – {term} için DNSH sektörlerinde (%40+) maruziyet: {', '.join(message)}"
                        )
                    #==========================
                    nonzero = filtered[filtered["Maruz Kalan Miktar"] > 0]

                    # ==== PIE CHART ====
                    if nonzero.empty:
                        pie_fig = go.Figure(data=[go.Pie(
                            labels=["Tehlike Yok"],
                            values=[1],
                            hole=0.4,
                            textinfo="label",
                            marker_colors=["#E0E0E0"]
                        )])
                        pie_fig.update_layout(
                            title={"text": f"{term} İçin Maruz Kalma Yok", "x": 0.5, "xanchor": "center"},
                            showlegend=False,
                            annotations=[dict(text="Tehlike Yok", x=0.5, y=0.5, font_size=18, showarrow=False)]
                        )
                    else:
                        pie_fig = go.Figure(data=[go.Pie(
                            labels=nonzero["Sector"],
                            values=nonzero["Maruz Kalan Miktar"],
                            hole=0.3,
                            textinfo="label+percent",
                            textposition='outside',
                            hovertemplate="%{label}<br>Miktar: %{value:,.0f} €<extra></extra>"
                        )])
                        pie_fig.update_layout(
                            title={"text": f"{term} - {selected_scenario.upper()} İçin Sektörel Maruz Kalma", "x": 0.5, "xanchor": "center"},
                            showlegend=True,
                            legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=-0.4,
                                xanchor="center",
                                x=0.5
                            ),
                            margin=dict(l=20, r=20, t=40, b=80),
                            height=450
                        )

                    # ==== BAR CHART ====
                    sectors = filtered["Sector"].tolist()
                    total_values = filtered["Toplam Miktar"].tolist()
                    exposed_values = filtered["Maruz Kalan Miktar"].tolist()

                    bar_fig = go.Figure()
                    bar_fig.add_trace(go.Bar(
                        x=sectors,
                        y=total_values,
                        name="Toplam Miktar",
                        marker_color="indianred"
                    ))
                    bar_fig.add_trace(go.Bar(
                        x=sectors,
                        y=exposed_values,
                        name="Maruz Kalan Miktar",
                        marker_color="lightskyblue"
                    ))
                    bar_fig.update_layout(
                        barmode="group",
                        title={
                            "text": f"{term} – {selected_scenario.upper()} Senaryosu için Miktarlar",
                            "x": 0.5,
                            "xanchor": "center"
                        },
                        xaxis_title="Sektör",
                        yaxis_title="Miktar (€)",
                        height=500
                    )
                    
                    colPieChart, colBarGraph = st.columns(2)
                    with colPieChart:
                        st.plotly_chart(pie_fig, use_container_width=True)
                    with colBarGraph:
                        st.plotly_chart(bar_fig, use_container_width=True)

                    st.markdown(f"<h4 style='text-align: center;'>📌 {selected_scenario.upper()} - {term} için Sektörel Yüzdelik Maruziyet</h4>", unsafe_allow_html=True)
                    for sector, total, exposed in zip(sectors, total_values, exposed_values):
                        if total > 0:
                            percentage = (exposed / total) * 100
                        else:
                            percentage = 0

                        st.markdown(
                            f"""
                            <div style="text-align: center;">
                                🔹 <strong>{sector}:</strong> <strong><span style="color:#28a745;">%{percentage:.1f}</span></strong> 
                                (<strong><span style="color:#dc3545;">{exposed:,.0f}</span></strong>   / {total:,.0f})
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    st.write(" ")

hazard_dfs = {
    h_type: {
        sc.upper(): pd.DataFrame([r for r in recs if r.get("Scenario", "").upper() == sc.upper()])
        for sc in {"ssp245", "ssp585"}
        if any(r.get("Scenario", "").upper() == sc.upper() for r in recs)
    }
    for h_type, recs in hazard_records.items() if recs
}
st.session_state.hazard_df = hazard_dfs


with st.container(border=True):
    st.write("## 🌍 Vadesel Tehlike Skorları")

    if "selected_scenario" not in st.session_state:
        st.session_state.selected_scenario = "SSP245"
    if "selected_term" not in st.session_state:
        st.session_state.selected_term = "Short Term"
    if "selected_hazard" not in st.session_state:
        st.session_state.selected_hazard = list(hazard_dfs.keys())[0]

    col1, col2, col3 = st.columns(3)
    with col1:
        st.session_state.selected_scenario = st.selectbox("Senaryo Seçin", ["SSP245", "SSP585"], index=["SSP245", "SSP585"].index(st.session_state.selected_scenario))
    with col2:
        st.session_state.selected_term = st.selectbox("Zaman Aralığı", ["Short Term", "Mid Term", "Long Term"], index=["Short Term", "Mid Term", "Long Term"].index(st.session_state.selected_term))
    with col3:
        st.session_state.selected_hazard = st.selectbox("Tehlike Tipi", list(hazard_dfs.keys()), index=list(hazard_dfs.keys()).index(st.session_state.selected_hazard))

    hazard = st.session_state.selected_hazard
    scenario = st.session_state.selected_scenario.upper()
    term = st.session_state.selected_term

    df = hazard_dfs[hazard].get(scenario, pd.DataFrame())

    if df.empty:
        st.warning("Seçilen tehlike ve senaryo için veri bulunamadı.")
        st.stop()

    all_terms = ["Short Term", "Mid Term", "Long Term"]
    terms_to_drop = [t for t in all_terms if t != term]
    term_filtered_df = df.drop(columns=terms_to_drop, errors="ignore").copy()

    desired_order = ["Asset ID", "Scenario", "Latitude", "Longitude", term]
    cols = [col for col in desired_order if col in term_filtered_df.columns]
    term_filtered_df = term_filtered_df[cols]

    #st.dataframe(term_filtered_df, use_container_width=True) # sanırım basmama gerek yok bunu.

    @st.cache_data
    def load_districts():
        return gpd.read_file("gadm_turkey/gadm41_TUR_2.shp")

    districts = load_districts()
    geometry = [Point(xy) for xy in zip(df["Longitude"], df["Latitude"])]
    points = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    joined = gpd.sjoin(points, districts, how="left", predicate="within")
    joined_scenario = joined[joined["Scenario"].str.upper() == scenario.upper()]

    for loop_term in ["Short Term", "Mid Term", "Long Term"]:
        if hazard in ("Orman Yangını", "Fırtına", "Sel"):
            flag = f"{loop_term} Acute Flag"
            if flag not in joined_scenario.columns:
                joined_scenario[flag] = joined_scenario[loop_term] > acute_threshold
        else:
            flag = f"{loop_term} Chronic Flag"
            if flag not in joined_scenario.columns:
                joined_scenario[flag] = joined_scenario[loop_term] > chronic_threshold

    grouped = joined_scenario.groupby("GID_2")[[term]].mean().reset_index()
    districts_grouped = districts.merge(grouped, on="GID_2", how="left")
    districts_map_data = prepare_district_map_data(districts_grouped, joined_scenario, term)

    rename_dict = {
        "Province": "İl",
        "District": "İlçe",
        "Contributors": "Katkıda Bulunanlar",
        f"{term} Average": "Ortalama Skor"
    }
    districts_map_data = districts_map_data.rename(columns=rename_dict)

    m = districts_map_data.explore(
        column="Ortalama Skor",
        tooltip=["İl", "İlçe", "Ortalama Skor", "Katkıda Bulunanlar"],
        cmap="YlOrRd",
        legend=True,
        highlight=True,
        zoom_start=6,
        location=[35.0, 35.0],
        control_scale=False
    )
    m.options["minZoom"] = 5
    m.options["maxZoom"] = 8
    Fullscreen().add_to(m)

    
    with st.expander(f"### İlçe Bazında Vadesel Tehlike Skor Dağılımı Haritaları ({hazard})", expanded=False):
        with st.spinner(f"{term} haritası yükleniyor..."):
            st.write(f"#### {term}")
            st.components.v1.html(m._repr_html_(), height=500, width =1800, scrolling=False)
            st.info("INFO - Kayıtlı varlıkların hani ilçeye düştükleri ve bu ilçeye düşen varlıkların seçili risk bazında ortalama değerlerini ısı haritasıyla gösterir. ")

    if hazard in ("Orman Yangını", "Sel", "Fırtına"):
        threshold_type = "Akut"
        threshold_value = acute_threshold
        flag_type = "Acute"
    else:
        threshold_type = "Kronik"
        threshold_value = chronic_threshold
        flag_type = "Chronic"

    
    with st.expander(f"### İlçe Bazında Risk Gösteren Varlık Haritaları ({hazard} için {threshold_type} Eşik: >{threshold_value})", expanded=False):
        with st.spinner(f"{term} haritası yükleniyor..."):
            st.write(f"#### {term}")
            flag_type = "Acute" if hazard in ("Orman Yangını", "Sel", "Fırtına") else "Chronic"
            flag_col = f"{term} {flag_type} Flag"
            risk_map = prepare_flag_based_map(term, flag_col, joined_scenario)
            if risk_map:
                st.components.v1.html(risk_map._repr_html_(), height=500, width = 1800, scrolling=False)
                st.info("INFO - İlçelere düşen varlıkların hangilerinin akut - kronik eşikleri geçtiğini belirler ve eşiği geçen varlıkların ID'sini, toplam eşik geçen varlık sayısıyla birlikte gösterir.")
            else:
                st.warning("Seçilen koşullar için riskli varlık haritası oluşturulamadı.")
    # =========================================

# ==============================================
with st.container(border=True):
    st.write(f"## 🌍 Vadesel Tehlikeye Bağlı Maruziyet")
    with st.expander(f"### Tehlikelerin İlk Oluşumlarına Göre Meydana Gelecek Vadesel Zararlar", expanded=False):
        with st.spinner("Analiz Yükleniyor..."):
            scenario_options = ["SSP245", "SSP585"]
            selected_scenario = st.radio("Senaryo Seçimi", scenario_options, horizontal=True)
            selected_scenario = selected_scenario.lower()  # lowercase for filtering
            exceedance_df = exceedance_df[exceedance_df["Scenario"].str.lower() == selected_scenario].copy()
            exceedance_df["Exposure Amount"] = pd.to_numeric(exceedance_df["Exposure Amount"], errors="coerce")

            
            all_assets = asset_df[["asset_id", "exposure_amount"]].drop_duplicates()
            all_assets = (
                asset_df[["asset_id", "exposure_amount"]]
                .drop_duplicates()
                .rename(columns={"asset_id": "Asset ID", "exposure_amount": "Exposure Amount"})
            )
            hazards = exceedance_df["Hazard"].unique()
            terms = ["Short Term", "Mid Term", "Long Term"]
            combos = pd.DataFrame(itertools.product(all_assets["Asset ID"], hazards, terms),
                            columns=["Asset ID", "Hazard", "Belonging Term"])
            combos = combos.merge(all_assets, on="Asset ID", how="left")
            exceedance_df["is_exceeding"] = True
            merged = combos.merge(
                exceedance_df[["Asset ID", "Hazard", "Belonging Term", "is_exceeding"]],
                on=["Asset ID", "Hazard", "Belonging Term"],
                how="left"
            )
            merged["is_exceeding"] = merged["is_exceeding"].fillna(False)
            merged = merged.infer_objects(copy=False)
            summary_by_hazard_term = (
                merged.groupby(["Hazard", "Belonging Term"], as_index=False)
                .agg(
                    total_exposure=("Exposure Amount", "sum"),
                    exposed_exposure=("Exposure Amount", lambda x: x[merged.loc[x.index, "is_exceeding"]].sum())
                )
            )
            summary_by_hazard_term["exposure_share"] = summary_by_hazard_term["exposed_exposure"] / summary_by_hazard_term["total_exposure"]
            summary_by_hazard_term["exposure_share"] = (summary_by_hazard_term["exposure_share"] * 100).round(1).astype(str) + "%"
            summary_by_hazard_term["time_band"] = summary_by_hazard_term["Belonging Term"].map(term_map_short)
            summary_by_hazard_term = summary_by_hazard_term.rename(columns = {"total_exposure": "Toplam Miktar","exposed_exposure": "Toplam Maruz Miktar","exposure_share": "Maruziyet Yüzdesi","time_band": "Vade"})
            #st.write(summary_by_hazard_term) # bunu da basıp kalabalık yapmayacağım

            import altair as alt
            bar_chart = alt.Chart(summary_by_hazard_term).mark_bar().encode(
                x=alt.X("Vade:N", title="Zaman Bandı", sort=["KV", "OV", "UV"]),
                y=alt.Y("Toplam Maruz Miktar:Q", title="Maruz Kalan Miktar (€)"),
                color=alt.Color("Hazard:N", title="Tehlike Türü"),
                tooltip=[
                    alt.Tooltip("Hazard:N", title="Tehlike"),
                    alt.Tooltip("Vade:N", title="Zaman Bandı"),
                    alt.Tooltip("Toplam Maruz Miktar:Q", title="Maruz Kalan (€)", format=",.0f")
                ]
            ).properties(
                width=700,
                height=400,
                title="Zaman Bandına Göre Tehlikelerin İlk Ortaya Çıkışına Bağlı Maruziyet"
            )

            st.altair_chart(bar_chart, use_container_width=True)
            st.info("INFO - Her bir varlığa etki eden 6 tehlike türüne sahibiz. 2015-2100 yıllarına ssp245 ve ssp585 senaryolarına ait tehlike skorları kullanılarak, bu 6 tehlike türünün belirlenen eşik değerlerini aştıkları ilk yıl hesaplanır (Örn: 2052). Hesaplanan yılın Kısa-Orta-Uzun vadeden hangisine dahil olduğu belirlenir, ve maruziyet hesabı yapılabilmesi için vade bazlı toplama işlemi yapılır. ")
 # ===================================================================    


 # ==================== Modül 7 Full Assumption ====================
    # pd_multiplier --> PD MUltiplier --> Temerrüt Olasılığı Çarpanı --> Mevcut PD Değeri bununla çarplır. Daha yüksek risk skoru = daha yüksek PD
    # lgd_add_pp --> LGD Add per Point --> LGD'ye eklenecek yüzde puan --> LGD'ye doğrudan +%X eklenir.
    # haircut_multiplier --> Haircut Multiplier --> Saç Traşı Çarpanı --> Teminat değerindeki düşüş oranı çarpanla artırılır.

    # pd_adj --> PD Ayarı --> Ekstra PD Artışı (çarpan cinsinden)
    # lgd_adj_pp --> LGD Ek Puan --> LGD'ye +%X eklemesi
    # haircut_adj --> Haircut Ayarı --> Teminat saç traşı için ekstra artış (çarpan olarak)

    # Exposure Amount Default == Exposure Amount? 
    # PD == Probability of Default
    # LGD == Loss Given Default
    # Expected Credit Loss_base = EAD * PD_baseline * LGD_baseline 
    # Expected Credit Loss_stressed = EAD * PD_stressed * LGD_stressed

    ## Yıllık değerlerle hesaplamaya başlayacağım, oradan kısa orta uzun vade ortalamaya geçmek kolay olacak.
PD_baseline = 3
LGD_baseline = 45
Haircut_baseline = 10

risk_score_bands = [
    {"score_min": 0,  "score_max": 10.999,  "pd_multiplier": 1,   "lgd_add_pp": 0,  "haircut_multiplier": 1},
    {"score_min": 11, "score_max": 25.999,  "pd_multiplier": 1.15,"lgd_add_pp": 2,  "haircut_multiplier": 1.05},
    {"score_min": 26, "score_max": 50.999,  "pd_multiplier": 1.3, "lgd_add_pp": 5,  "haircut_multiplier": 1.1},
    {"score_min": 51, "score_max": 75.999,  "pd_multiplier": 1.6, "lgd_add_pp": 8,  "haircut_multiplier": 1.2},
    {"score_min": 76, "score_max": 100.0, "pd_multiplier": 2,   "lgd_add_pp": 12, "haircut_multiplier": 1.3},
]

hazard_overlay = [
    {"hazard_type": "Sel", "pd_adj": 0.1, "lgd_adj_pp": 1, "haircut_adj": 0.05},
    {"hazard_type": "Fırtına", "pd_adj": 0.1, "lgd_adj_pp": 1, "haircut_adj": 0.05},
    {"hazard_type": "Sıcak Hava Dalgası", "pd_adj": 0.05, "lgd_adj_pp": 0, "haircut_adj": 0.01},
    {"hazard_type": "Kuraklık", "pd_adj": 0.05, "lgd_adj_pp": 0, "haircut_adj": 0.01},
    {"hazard_type": "Orman Yangını", "pd_adj": 0.05, "lgd_adj_pp": 2, "haircut_adj": 0.03},
    {"hazard_type": "Deniz Seviyesi Yükselmesi", "pd_adj": 0, "lgd_adj_pp": 3, "haircut_adj": 0.08}
]

sector_overlay = [
    {"nace_section": "A", "sector_pd_adj": 0.1, "sector_lgd_adj": 2, "sector_haircut_adj": 0.03},
    {"nace_section": "B", "sector_pd_adj": 0.05, "sector_lgd_adj": 1, "sector_haircut_adj": 0.02},
    {"nace_section": "C", "sector_pd_adj": 0, "sector_lgd_adj": 0, "sector_haircut_adj": 0.02},
    {"nace_section": "D", "sector_pd_adj": 0.05, "sector_lgd_adj": 1, "sector_haircut_adj": 0.03},
    {"nace_section": "E", "sector_pd_adj": 0, "sector_lgd_adj": 0, "sector_haircut_adj": 0}, 
    {"nace_section": "F", "sector_pd_adj": 0.05, "sector_lgd_adj": 2, "sector_haircut_adj": 0.05},
    {"nace_section": "G", "sector_pd_adj": 0, "sector_lgd_adj": 0, "sector_haircut_adj": 0.01},
    {"nace_section": "H", "sector_pd_adj": 0, "sector_lgd_adj": 0, "sector_haircut_adj": 0.01},
    {"nace_section": "I", "sector_pd_adj": 0.05, "sector_lgd_adj": 1, "sector_haircut_adj": 0.02}
]

with st.container(border=True):
    st.write(f"### 🔥 Modül 7 Boundless")
    raw_scores = raw_scores[["Asset ID","Date", "Endpoint", "Scenario", "NACE Code", "Exposure Amount", "score"]]
    raw_scores["score"] = raw_scores.apply(lambda row: row["score"] * get_vulnerability(row["NACE Code"]), axis=1)
    raw_scores["Date"] = raw_scores["Date"].dt.year 
    raw_scores["Endpoint"] = raw_scores["Endpoint"].map(endpoint_label_map)

    # RICK SCORE BANDS
    raw_scores_copy = raw_scores.copy()
    risk_bands_df = pd.DataFrame(risk_score_bands)
    bins = [0, 11, 26, 51, 76, 101]  
    labels = list(range(len(risk_score_bands)))
    raw_scores_copy["band_index"] = pd.cut(raw_scores_copy["score"],bins=bins,labels=labels,right=False).astype(int)
    raw_scores_copy = raw_scores_copy.merge(risk_bands_df,left_on="band_index",right_index=True,how="left")
    raw_scores_copy = raw_scores_copy.drop(columns=["score_min", "score_max", "band_index"], errors="ignore")

    # HAZARD OVERLAY
    hazard_overlay_df = pd.DataFrame(hazard_overlay)
    raw_scores_copy = raw_scores_copy.merge(hazard_overlay_df, how="left", left_on="Endpoint", right_on="hazard_type")
    raw_scores_copy = raw_scores_copy.drop(columns=["hazard_type"])

    # SECTOR OVERLAY
    sector_overlay_df = pd.DataFrame(sector_overlay)
    raw_scores_copy = raw_scores_copy.merge(sector_overlay_df, how="left", left_on="NACE Code", right_on="nace_section")
    raw_scores_copy = raw_scores_copy.drop(columns=["nace_section"])

    # FINAL VALUES (NIHAI KATSAYILAR)
    # PD_baseline * xlsx'lerden gelenler.
    raw_scores_copy["PD_stressed"] = PD_baseline * (raw_scores_copy["pd_multiplier"] + raw_scores_copy["pd_adj"] + raw_scores_copy["sector_pd_adj"])
    # LGD_baseline + xlsx'lerden gelenler.
    raw_scores_copy["LGD_stressed"] = LGD_baseline + (raw_scores_copy["lgd_add_pp"] + raw_scores_copy["lgd_adj_pp"] + raw_scores_copy["sector_lgd_adj"])
    # Haircut_baseline + xlsx'lerden gelenler.
    raw_scores_copy["Haircut_stressed"] = Haircut_baseline + (raw_scores_copy["haircut_multiplier"] + raw_scores_copy["haircut_adj"] + raw_scores_copy["sector_haircut_adj"])
    raw_scores_copy = raw_scores_copy.drop(columns=["pd_multiplier", "pd_adj", "sector_pd_adj", "lgd_add_pp", "lgd_adj_pp", "sector_lgd_adj", "haircut_multiplier", "haircut_adj", "sector_haircut_adj"], errors="ignore")

    # ECL_base ve ECL_stressed hesaplamaları
    raw_scores_copy["Exposure Amount"] = pd.to_numeric(raw_scores_copy["Exposure Amount"], errors="coerce")
    raw_scores_copy["ECL_base"] = raw_scores_copy["Exposure Amount"] * (PD_baseline/100) * (LGD_baseline/100)
    raw_scores_copy["ECL_stressed"] = raw_scores_copy["Exposure Amount"] * (raw_scores_copy["PD_stressed"]/100) * (raw_scores_copy["LGD_stressed"]/100)
    raw_scores_copy["ΔECL"] = raw_scores_copy["ECL_stressed"] - raw_scores_copy["ECL_base"]
    raw_scores_copy["ECL_base"] = raw_scores_copy["ECL_base"].apply(lambda x: f"{x:,.0f}".replace(",", ".")) # formatting
    raw_scores_copy["ECL_stressed"] = raw_scores_copy["ECL_stressed"].apply(lambda x: f"{x:,.0f}".replace(",", ".")) # formatting
    raw_scores_copy["ΔECL"] = raw_scores_copy["ΔECL"].apply(lambda x: f"{x:,.0f}".replace(",", ".")) # formatting

    # TEMINAT HESAPLAMALARI
    raw_scores_copy["Teminat Kaybı"] = (raw_scores_copy["Haircut_stressed"]/100) * (raw_scores_copy["Exposure Amount"] / 2)
    raw_scores_copy["Net Teminat Kaybı"] = (raw_scores_copy["Exposure Amount"] / 2) - raw_scores_copy["Teminat Kaybı"]
    raw_scores_copy["Teminat Kaybı"] = raw_scores_copy["Teminat Kaybı"].apply(lambda x: f"{x:,.0f}".replace(",", ".")) # formatting
    raw_scores_copy["Net Teminat Kaybı"] = raw_scores_copy["Net Teminat Kaybı"].apply(lambda x: f"{x:,.0f}".replace(",", ".")) # formatting
    raw_scores_copy = raw_scores_copy.drop(columns=["Exposure Amount"], errors="ignore")

    pivoted_scores = raw_scores.pivot_table(index=["Asset ID", "Date"], columns=["Endpoint", "Scenario" ], values="score").reset_index()
    pivoted_scores.columns.name = None
    pivoted_scores.columns = pd.MultiIndex.from_tuples([(endpoint, scenario.upper()) for endpoint, scenario in pivoted_scores.columns])

    row = pivoted_scores.loc[(pivoted_scores["Asset ID"] == "TR006") & (pivoted_scores["Date"] == 2038), ("Sel", "SSP245")]
    #sel_all = pivoted_scores[[("Sel", "SSP245"), ("Sel", "SSP585")]]
    # 🠕🠕🠕🠕🠕 Pivot DF'den satır bazlı veri çekmek için format 🠕🠕🠕🠕🠕 

    with st.expander(f"#### HxV = Risk (Senaryo Bazlı Tüm Assetler için)", expanded=False):
        #st.write(pivoted_scores)
        #st.write(raw_scores_copy)
        st.write("In Progress...")
# ==================== ======================= ====================



