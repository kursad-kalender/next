import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

st.set_page_config(
    page_title="UrClimate Next",
    page_icon="🌏",
    layout="wide",
)

endpoint_label_map = {
    "point-high-fwi-days-score": "Orman Yangını",
    "point-sfcwind-over-p99-score": "Fırtına",
    "point-pr-over-p95-score": "Sel",
    "point-ndd-score": "Kuraklık",
    "point-txge35-score": "Sıcak Hava Dalgası",
    "point-fd-score": "Deniz Seviyesi Yükselmesi"
}

if "hazard_df" in st.session_state:
    hazard_dict = st.session_state.hazard_df
    final_list = []

    for hazard_name, scenarios in hazard_dict.items():
        scenario_dfs = []
        shared_data = None

        for scenario_name, df in scenarios.items():
            if df.empty:
                continue

            base_cols = ["Asset ID", "Latitude", "Longitude"]
            values_cols = ["Short Term", "Mid Term", "Long Term"]

            common_data = df[base_cols].copy()
            values_data = df[values_cols].copy()

            values_data.columns = pd.MultiIndex.from_product([[scenario_name], values_data.columns])

            if shared_data is None:
                shared_data = common_data.copy()
                shared_data.insert(0, "Hazard", hazard_name)
                shared_data.set_index(["Hazard", "Asset ID", "Latitude", "Longitude"], inplace=True)

            values_data.index = shared_data.index
            scenario_dfs.append(values_data)

        if shared_data is not None and scenario_dfs:
            merged = pd.concat([shared_data] + scenario_dfs, axis=1)
            final_list.append(merged)

    if final_list:
        df_final = pd.concat(final_list)
        new_cols = []
        for col in df_final.columns:
            if isinstance(col, tuple):
                if col[1] == "" and col[0] not in ["Latitude", "Longitude"]:
                    new_cols.append((col[0], ""))
                else:
                    new_cols.append(col)
            else:
                new_cols.append((col, ""))

        df_final.columns = pd.MultiIndex.from_tuples(new_cols)



        with st.container(border=True):
            colTitle, colButton = st.columns([8, 1])
            with colTitle:
                st.write("### Asset ID Arama - Tüm Tehlikeler")
            with colButton:
                st.page_link("final.py", label = "Anasayfaya Geri Dön")
            asset_ids = sorted(df_final.index.get_level_values("Asset ID").unique().tolist())

            keyword = st.text_input("Asset ID girin.").strip()
            filtered_ids = [aid for aid in asset_ids if keyword.lower() in aid.lower()] if keyword else []

            if filtered_ids:
                selected_id = st.selectbox("Eşleşenleri seçin:", options=filtered_ids)
                if selected_id:
                    matching_rows = df_final[df_final.index.get_level_values("Asset ID") == selected_id]
                    if not matching_rows.empty:
                        with st.expander(f"## Arama Sonuçları", expanded=True):
                            st.dataframe(matching_rows)
                        
                        st.divider()
                        with st.container(border=True):
                            if "raw_scores" in st.session_state:
                                raw_scores = st.session_state.raw_scores
                                raw_filtered = raw_scores[raw_scores["Asset ID"] == selected_id]
                                raw_filtered = raw_filtered[["Asset ID","Date", "Endpoint", "Scenario",  "score"]]
                                raw_filtered["Date"] = raw_filtered["Date"].dt.year # date sütununu sadece yıla çeviriyorum.
                                raw_filtered["Endpoint"] = raw_filtered["Endpoint"].map(endpoint_label_map) # Endpoint sütununu mapliyorum ki hazard namelere geçelim.
                                # Pivot: Senaryo başlık, skor değer, diğerleri index
                                pivot_df = raw_filtered.pivot_table(
                                    index=["Asset ID", "Date", "Endpoint"],
                                    columns="Scenario",
                                    values="score"
                                ).reset_index()
                                pivot_df.columns.name = None
                                #st.dataframe(pivot_df, use_container_width=True)

                                # ==== Çizgi Grafiği Tüm Hazardlar için st.tabs====
                                hazard_names = pivot_df["Endpoint"].unique().tolist()
                                st.write(f"## Yıllık Tehlike Değerlerinin Senaryo Karşılaştırılması")
                                tabs = st.tabs(hazard_names)
                                for tab, selected_hazard in zip(tabs, hazard_names):
                                    with tab:
                                        filtered = pivot_df[pivot_df["Endpoint"] == selected_hazard]

                                        if filtered.empty:
                                            st.warning(f"{selected_hazard} için veri bulunamadı.")
                                            continue

                                        fig = go.Figure()
                                        fig.add_trace(go.Scatter(
                                            x=filtered["Date"],
                                            y=filtered["ssp245"],
                                            mode="lines+markers",
                                            name="SSP245"
                                        ))
                                        fig.add_trace(go.Scatter(
                                            x=filtered["Date"],
                                            y=filtered["ssp585"],
                                            mode="lines+markers",
                                            name="SSP585",
                                            line=dict(color="red"),  
                                            marker=dict(color="red")
                                        ))
                                        # Trend çizgisi - SSP245
                                        x_vals = filtered["Date"].values
                                        y_vals_245 = filtered["ssp245"].values
                                        if len(x_vals) >= 2:  # en az 2 nokta olmalı
                                            coef_245 = np.polyfit(x_vals, y_vals_245, 1)
                                            trend_245 = np.poly1d(coef_245)
                                            fig.add_trace(go.Scatter(
                                                x=x_vals,
                                                y=trend_245(x_vals),
                                                mode="lines",
                                                name="Trend SSP245",
                                                line=dict(dash="dot", color="blue", width = 2)
                                            ))

                                        # Trend çizgisi - SSP585
                                        y_vals_585 = filtered["ssp585"].values
                                        if len(x_vals) >= 2:
                                            coef_585 = np.polyfit(x_vals, y_vals_585, 1)
                                            trend_585 = np.poly1d(coef_585)
                                            fig.add_trace(go.Scatter(
                                                x=x_vals,
                                                y=trend_585(x_vals),
                                                mode="lines",
                                                name="Trend SSP585",
                                                line=dict(dash="dot", color="darkred", width = 2)
                                            ))

                                        fig.update_layout(
                                            title = f"{selected_id} - {selected_hazard} | 2015-2100 Tehlike Değerleri",
                                            xaxis_title="Yıl",
                                            yaxis_title="Tehlike Skoru",
                                            hovermode="x unified"
                                        )

                                        st.plotly_chart(fig, use_container_width=True)
                                # ====== Çizgi Grafiği End =======   
                    else:
                        st.warning(f"{selected_id} için sonuç bulunamadı.")
            elif keyword:  # sadece kullanıcı bir şey yazdıysa ama eşleşme olmadıysa
                st.warning("Aradığınız kelimeyi içeren bir asset bulunamadı.")
    else:
        st.warning("No data after merging hazard scenarios.")
else:
    st.warning("No data in session state.")

