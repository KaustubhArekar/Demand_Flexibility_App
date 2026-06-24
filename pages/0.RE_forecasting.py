import streamlit as st
import pandas as pd
import numpy as np
import io
import os
import pytz
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Lazy imports for heavy dependencies ──────────────────────────────────────
def _import_ml():
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_squared_error
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Input
    import tensorflow as tf
    tf.random.set_seed(455)
    np.random.seed(455)
    return MinMaxScaler, mean_squared_error, Sequential, LSTM, Dense, Input

def _import_pvlib():
    import pvlib
    from pvlib import location, irradiance
    return location, irradiance

# ── Page config ───────────────────────────────────────────────────────────────
st.title("RE Generation Forecasting")
st.write(
    "Train LSTM models on historical weather data to forecast hourly solar and wind "
    "generation for each plant. The aggregated output is a ready-to-use **RE.xlsx** "
    "for the Demand Flexibility Analysis Model."
)
st.info(
    "⚠️ LSTM training is compute-intensive. Expect **2–5 minutes per plant** depending "
    "on server load. Do not refresh the page during training.",
    icon="⏱️"
)

# ════════════════════════════════════════════════════════════════════════════
# STAGE 1 — Plant Configuration
# ════════════════════════════════════════════════════════════════════════════
st.divider()
st.subheader("Stage 1: Configure Plants")

col_np, col_yr = st.columns([1, 2])
with col_np:
    num_plants = st.number_input(
        "Number of plants",
        min_value=1, max_value=20, value=1, step=1,
        help="Total number of solar/wind plants to include in the forecast"
    )

with col_yr:
    col_ts, col_te, col_fc = st.columns(3)
    with col_ts:
        tstart = st.number_input("Training start year", min_value=1990, max_value=2020, value=2001)
    with col_te:
        tend   = st.number_input("Training end year",   min_value=2000, max_value=2023, value=2022)
    with col_fc:
        forecast_year = st.number_input("Forecast year", min_value=2023, max_value=2035, value=2023)

st.caption(
    "Training data: years **{} – {}** | Forecast target: **{}**".format(
        int(tstart), int(tend), int(forecast_year)
    )
)

# ════════════════════════════════════════════════════════════════════════════
# STAGE 2 — Per-plant File Upload & Parameters
# ════════════════════════════════════════════════════════════════════════════
st.divider()
st.subheader("Stage 2: Upload Plant Data")

st.markdown("""
Each plant requires **three files**:

| File | Required columns | Notes |
|---|---|---|
| **Weather CSV** (`*Plant.csv`) | `Timestamp`, `Temperature`, `GHI`, `Wind Speed` | Hourly rows, `Timestamp` parseable as datetime |
| **Solar Info CSV** (`Info_Solar.csv`) | `Parameter`, `Value` | See parameter table below |
| **Wind Info CSV** (`Info_Wind.csv`) | `Parameter`, `Value` | See parameter table below |
""")

with st.expander("📋 Info_Solar.csv parameter reference"):
    st.table(pd.DataFrame({
        "Row (Parameter)": ["Latitude", "Longitude", "Pnom (kW)", "nDRT (derating factor)",
                            "alpha (%/°C)", "gamma", "A_PV (m²)"],
        "Example Value":   ["23.5",      "72.8",     "500",        "0.85",
                            "-0.004",    "1.0",       "2.0"],
        "Description": [
            "Site latitude in decimal degrees",
            "Site longitude in decimal degrees",
            "Rated DC power of PV array (kW)",
            "Derating factor accounting for soiling, shading, wiring losses (0–1)",
            "Temperature coefficient of power (%/°C, typically negative)",
            "Mounting factor (1 = free-standing)",
            "Surface area of PV module (m²)",
        ]
    }))

with st.expander("📋 Info_Wind.csv parameter reference"):
    st.table(pd.DataFrame({
        "Row (Parameter)": ["Latitude", "Longitude", "Prtdwn (kW)", "Vctin (m/s)",
                            "Vrtdwn (m/s)", "Vctout (m/s)"],
        "Example Value":   ["23.5",      "72.8",      "2000",         "2.5",
                            "12.0",       "25.0"],
        "Description": [
            "Site latitude",
            "Site longitude",
            "Rated power of wind turbine (kW)",
            "Cut-in wind speed — generation starts above this",
            "Rated wind speed — full power above this",
            "Cut-out wind speed — turbine shuts down above this",
        ]
    }))

# ── Build per-plant upload widgets ───────────────────────────────────────────
plant_uploads = {}   # {plant_idx: {"weather": file, "solar_info": file, "wind_info": file, "name": str}}

for i in range(int(num_plants)):
    with st.expander(f"Plant {i+1}", expanded=(i == 0)):
        plant_name = st.text_input(
            "Plant name / ID",
            value=f"Plant_{i+1}",
            key=f"pname_{i}"
        )
        plant_type = st.selectbox(
            "Plant type",
            options=["Solar", "Wind", "Solar + Wind"],
            key=f"ptype_{i}"
        )
        uc1, uc2, uc3 = st.columns(3)
        with uc1:
            w_file = st.file_uploader(
                "Weather CSV (*Plant.csv)",
                type="csv", key=f"weather_{i}"
            )
        with uc2:
            s_file = st.file_uploader(
                "Solar Info CSV",
                type="csv", key=f"solar_{i}",
                disabled=(plant_type == "Wind")
            )
        with uc3:
            wnd_file = st.file_uploader(
                "Wind Info CSV",
                type="csv", key=f"wind_{i}",
                disabled=(plant_type == "Solar")
            )
        plant_uploads[i] = {
            "name": plant_name,
            "type": plant_type,
            "weather": w_file,
            "solar_info": s_file,
            "wind_info": wnd_file,
        }

# ── Upload validation ─────────────────────────────────────────────────────────
def _uploads_complete(uploads):
    for i, p in uploads.items():
        if p["weather"] is None:
            return False, f"Plant {i+1} ({p['name']}): weather CSV missing"
        if p["type"] in ("Solar", "Solar + Wind") and p["solar_info"] is None:
            return False, f"Plant {i+1} ({p['name']}): Solar Info CSV missing"
        if p["type"] in ("Wind", "Solar + Wind") and p["wind_info"] is None:
            return False, f"Plant {i+1} ({p['name']}): Wind Info CSV missing"
    return True, ""

uploads_ok, upload_msg = _uploads_complete(plant_uploads)
if not uploads_ok:
    st.warning(f"⚠️ {upload_msg}")

# ════════════════════════════════════════════════════════════════════════════
# STAGE 3 — Training & Forecasting
# ════════════════════════════════════════════════════════════════════════════
st.divider()
st.subheader("Stage 3: Train & Forecast")

# LSTM hyperparameters (advanced expander)
with st.expander("⚙️ Advanced LSTM settings"):
    ac1, ac2, ac3 = st.columns(3)
    with ac1:
        n_steps  = st.number_input("Look-back steps (n_steps)", min_value=5,  max_value=50, value=10)
        epochs_n = st.number_input("Training epochs",           min_value=1,  max_value=50, value=5)
    with ac2:
        lstm_units  = st.number_input("LSTM units",   min_value=32, max_value=256, value=125, step=16)
        batch_size  = st.number_input("Batch size",   min_value=8,  max_value=128, value=32,  step=8)
    with ac3:
        st.markdown("**Default values match the reference notebook.** Increase epochs for better accuracy at the cost of longer training time.")

run_btn = st.button(
    "🚀 Run Forecast",
    disabled=not uploads_ok,
    use_container_width=True,
    type="primary"
)

# ── Core helper functions ─────────────────────────────────────────────────────
def split_sequence(sequence, n_steps):
    X, y = [], []
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence) - 1:
            break
        X.append(sequence[i:end_ix])
        y.append(sequence[end_ix])
    return np.array(X), np.array(y)


def train_lstm(train_scaled, n_steps, lstm_units, epochs_n, batch_size):
    MinMaxScaler, mean_squared_error, Sequential, LSTM, Dense, Input = _import_ml()
    model = Sequential([
        Input(shape=(n_steps, 1)),
        LSTM(units=lstm_units, activation="tanh"),
        Dense(units=1)
    ])
    model.compile(optimizer="RMSprop", loss="mse")
    X_train, y_train = split_sequence(train_scaled, n_steps)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    model.fit(X_train, y_train, epochs=int(epochs_n), batch_size=int(batch_size), verbose=0)
    return model


def predict_series(model, full_series_scaled, test_len, n_steps, scaler):
    inputs = full_series_scaled[len(full_series_scaled) - test_len - n_steps:]
    inputs = inputs.reshape(-1, 1)
    X_test, _ = split_sequence(inputs, n_steps)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    predicted = model.predict(X_test, verbose=0)
    return scaler.inverse_transform(predicted)


def get_transposition_factor(lat, lon, forecast_year):
    location_mod, irradiance_mod = _import_pvlib()
    tz   = pytz.timezone("Asia/Kolkata")
    site = location_mod.Location(lat, lon, tz=tz)

    dates = pd.date_range(
        f"{int(forecast_year)}-01-01",
        f"{int(forecast_year)}-12-31",
        freq="D"
    )

    Final_irr = pd.DataFrame()
    for d in dates:
        times = pd.date_range(d, freq="10min", periods=6 * 24, tz=tz)
        clearsky       = site.get_clearsky(times)
        solar_position = site.get_solarposition(times=times)
        POA = irradiance_mod.get_total_irradiance(
            surface_tilt=20, surface_azimuth=0,
            dni=clearsky["dni"], ghi=clearsky["ghi"], dhi=clearsky["dhi"],
            solar_zenith=solar_position["apparent_zenith"],
            solar_azimuth=solar_position["azimuth"]
        )
        day_df = pd.DataFrame({"GHI": clearsky["ghi"], "POA": POA["poa_global"]})
        day_df = day_df.reset_index(drop=True)
        Final_irr = pd.concat([Final_irr, day_df], axis=1)

    Final_irr = Final_irr.transpose()
    GHI_mat = Final_irr.iloc[0::2].reset_index(drop=True)
    POA_mat = Final_irr.iloc[1::2].reset_index(drop=True)
    TF_10min = (POA_mat / GHI_mat).set_index(dates.strftime("%d/%m/%Y"))

    # Aggregate 10-min → hourly
    TF_hr = pd.DataFrame()
    for x in range(0, 144, 6):
        col = TF_10min.iloc[:, x:x + 6].mean(axis=1).to_frame()
        TF_hr = pd.concat([TF_hr, col], axis=1)
    TF_hr.columns = range(24)

    # Flatten day × hour → 8760 row series
    row_wise = [val for _, row in TF_hr.iterrows() for val in row]
    return pd.DataFrame(row_wise, columns=["Value"])


def compute_solar_power(predicted_GHI, predicted_Temp, TF_hr_new, info_solar):
    Pnom   = float(info_solar.iloc[2].iloc[0])
    nDRT   = float(info_solar.iloc[3].iloc[0])
    alpha  = float(info_solar.iloc[4].iloc[0])
    A_PV   = float(info_solar.iloc[6].iloc[0])

    Tc_STC  = 25
    GT_STC  = 1
    U0, U1, WS = 29, 0, 1
    Tau_alpha   = 0.9
    nmp_STC     = Pnom / (A_PV * GT_STC) / 100

    ghi_df  = pd.DataFrame(predicted_GHI, columns=["Value"])
    ghi_df[ghi_df < 1] = 0
    temp_df = pd.DataFrame(predicted_Temp, columns=["Value"])
    TF_hr_new = TF_hr_new.fillna(1).reset_index(drop=True)

    estimated_POA = TF_hr_new * ghi_df
    GT  = estimated_POA * 3600 * 10 / (1000 * 3600)
    Ta  = temp_df

    Tc  = Ta + estimated_POA * ((Tau_alpha * (1 - nmp_STC)) / (U0 + WS * U1))
    Pdc = Pnom * nDRT * (GT / GT_STC) * (1 + alpha * (Tc - Tc_STC))
    Pac = 0.95 * Pdc
    Pac.columns = ["AC Power (kW)"]
    return Pac


def compute_wind_power(predicted_WS, info_wind):
    Prtdwn = float(info_wind.iloc[2].iloc[0])
    Vctin  = float(info_wind.iloc[3].iloc[0])
    Vrtdwn = float(info_wind.iloc[4].iloc[0])
    Vctout = float(info_wind.iloc[5].iloc[0])

    rho, Cp, Pi = 1.225, 0.4, np.pi
    D    = np.sqrt((4 * Prtdwn) / (Pi * rho * (Vrtdwn ** 3) * Cp))
    A_WT = Pi * ((D / 2) ** 2)

    n_rows = len(predicted_WS)
    Pwn = np.zeros(n_rows)
    for i in range(n_rows):
        ws = float(predicted_WS[i, 0])
        if ws < Vctin or ws > Vctout:
            Pwn[i] = 0
        elif ws > Vrtdwn:
            Pwn[i] = Prtdwn / 1000
        else:
            Pwn[i] = (0.5 * rho * A_WT * (ws ** 3)) / 1000
    return pd.DataFrame(Pwn, columns=["Wind Power (kW)"])


# ── Main training loop ────────────────────────────────────────────────────────
if run_btn:
    MinMaxScaler, _, Sequential, LSTM, Dense, Input = _import_ml()

    final_solar = pd.DataFrame()
    final_wind  = pd.DataFrame()
    plant_results = {}   # store per-plant metrics for display

    overall_progress = st.progress(0, text="Starting…")
    log_area = st.empty()

    for idx, plant in plant_uploads.items():
        pname = plant["name"]
        ptype = plant["type"]
        overall_progress.progress(
            idx / int(num_plants),
            text=f"Processing {pname} ({idx+1}/{int(num_plants)})…"
        )

        log_area.info(f"📡 **{pname}** — reading weather data…")

        # ── Load weather CSV ─────────────────────────────────────────────
        try:
            dataset = pd.read_csv(plant["weather"], parse_dates=["Timestamp"])
        except Exception as e:
            st.error(f"{pname}: failed to read weather CSV — {e}")
            continue

        dataset = dataset.sort_values("Timestamp").set_index("Timestamp")

        required_cols = {"Temperature", "GHI", "Wind Speed"}
        if not required_cols.issubset(dataset.columns):
            st.error(f"{pname}: weather CSV must contain columns: {required_cols}. Found: {list(dataset.columns)}")
            continue

        train_data = dataset.loc[str(int(tstart)):str(int(tend))]
        test_data  = dataset.loc[str(int(tend + 1)):]

        if len(test_data) == 0:
            st.error(f"{pname}: no test data found after year {int(tend)}. Check your Timestamp range.")
            continue

        # ── Scale ────────────────────────────────────────────────────────
        scalers, trained_models, predicted = {}, {}, {}
        features_map = {
            "Temperature": "Temperature",
            "GHI":         "GHI",
            "Wind Speed":  "Wind Speed"
        }

        sc_bar = st.progress(0, text=f"{pname} — scaling…")
        for fi, (feat_key, feat_col) in enumerate(features_map.items()):
            sc = MinMaxScaler(feature_range=(0, 1))
            train_vals = train_data[feat_col].values.reshape(-1, 1)
            train_scaled = sc.fit_transform(train_vals)
            scalers[feat_key] = sc

            sc_bar.progress((fi * 2 + 1) / 8, text=f"{pname} — training LSTM for {feat_col}…")
            model = train_lstm(train_scaled, int(n_steps), int(lstm_units), int(epochs_n), int(batch_size))
            trained_models[feat_key] = model

            full_series = dataset[feat_col].values.reshape(-1, 1)
            full_scaled = sc.transform(full_series)
            pred = predict_series(model, full_scaled, len(test_data), int(n_steps), sc)
            predicted[feat_key] = pred
            sc_bar.progress((fi * 2 + 2) / 8, text=f"{pname} — {feat_col} done")

        sc_bar.progress(1.0, text=f"{pname} — weather forecast complete ✓")

        pred_GHI  = predicted["GHI"]
        pred_Temp = predicted["Temperature"]
        pred_WS   = predicted["Wind Speed"]
        pred_GHI[pred_GHI < 1] = 0
        pred_WS[pred_WS < 0]   = 0

        plant_solar_power = None
        plant_wind_power  = None

        # ── Solar power ──────────────────────────────────────────────────
        if ptype in ("Solar", "Solar + Wind"):
            log_area.info(f"☀️ **{pname}** — computing transposition factor & solar power…")
            try:
                info_solar = pd.read_csv(plant["solar_info"]).set_index("Parameter")
                lat = float(info_solar.iloc[0].iloc[0])
                lon = float(info_solar.iloc[1].iloc[0])
                TF  = get_transposition_factor(lat, lon, forecast_year)
                plant_solar_power = compute_solar_power(pred_GHI, pred_Temp, TF, info_solar)
                final_solar = pd.concat([final_solar, plant_solar_power], axis=1)
            except Exception as e:
                st.error(f"{pname}: solar computation failed — {e}")

        # ── Wind power ───────────────────────────────────────────────────
        if ptype in ("Wind", "Solar + Wind"):
            log_area.info(f"💨 **{pname}** — computing wind power…")
            try:
                info_wind = pd.read_csv(plant["wind_info"]).set_index("Parameter")
                plant_wind_power = compute_wind_power(pred_WS, info_wind)
                final_wind = pd.concat([final_wind, plant_wind_power], axis=1)
            except Exception as e:
                st.error(f"{pname}: wind computation failed — {e}")

        plant_results[pname] = {
            "type":    ptype,
            "pred_GHI":  pred_GHI,
            "pred_Temp": pred_Temp,
            "pred_WS":   pred_WS,
            "solar_power": plant_solar_power,
            "wind_power":  plant_wind_power,
        }

    # ── Aggregate ────────────────────────────────────────────────────────────
    overall_progress.progress(1.0, text="Aggregating results…")
    log_area.empty()

    # Determine canonical length from whichever DataFrame actually has data
    if not final_solar.empty:
        n_slots = len(final_solar)
    elif not final_wind.empty:
        n_slots = len(final_wind)
    else:
        st.error("No generation data was produced. Check plant files and try again.")
        st.stop()

    if not final_solar.empty:
        solar_agg = final_solar.sum(axis=1).values / 1000   # kW → MW
    else:
        solar_agg = np.zeros(n_slots)

    if not final_wind.empty:
        wind_agg = final_wind.sum(axis=1).values / 1000     # kW → MW
    else:
        wind_agg = np.zeros(n_slots)

    # Safety trim — align to shortest in case of partial plant failures
    n_slots = min(len(solar_agg), len(wind_agg))
    solar_agg = solar_agg[:n_slots]
    wind_agg  = wind_agg[:n_slots]

    RE_output = pd.DataFrame({
        "Solar": solar_agg,
        "Wind":  wind_agg,
    })

    st.session_state["RE_forecast_output"]  = RE_output
    st.session_state["RE_plant_results"]    = plant_results
    st.session_state["RE_forecast_year"]    = int(forecast_year)
    st.session_state["RE_num_plants"]       = int(num_plants)

    st.success(
        f"✅ Forecast complete! Aggregated Solar: **{RE_output['Solar'].sum():.1f} MWh** | "
        f"Wind: **{RE_output['Wind'].sum():.1f} MWh** across {int(num_plants)} plant(s)."
    )
    st.rerun()


# ════════════════════════════════════════════════════════════════════════════
# RESULTS — shown persistently from session state
# ════════════════════════════════════════════════════════════════════════════
if "RE_forecast_output" in st.session_state:
    RE_output     = st.session_state["RE_forecast_output"]
    plant_results = st.session_state["RE_plant_results"]
    fy            = st.session_state["RE_forecast_year"]

    st.divider()
    st.subheader(f"Results — Forecast Year {fy}")

    # ── KPI metrics ──────────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Solar Generation",  f"{RE_output['Solar'].sum():,.1f} MWh")
    m2.metric("Total Wind Generation",   f"{RE_output['Wind'].sum():,.1f} MWh")
    m3.metric("Peak Solar Hour",         f"{RE_output['Solar'].max():.2f} MW")
    m4.metric("Peak Wind Hour",          f"{RE_output['Wind'].max():.2f} MW")

    # ── Aggregated generation chart ──────────────────────────────────────────
    st.markdown("#### Aggregated Hourly Generation Profile")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=RE_output.index, y=RE_output["Solar"],
        name="Solar (MW)", mode="lines",
        line=dict(color="orange", width=1),
        hovertemplate="Hour %{x}<br>Solar: %{y:.3f} MW<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=RE_output.index, y=RE_output["Wind"],
        name="Wind (MW)", mode="lines",
        line=dict(color="steelblue", width=1),
        hovertemplate="Hour %{x}<br>Wind: %{y:.3f} MW<extra></extra>"
    ))
    fig.update_layout(
        xaxis_title=f"Hour of {fy}",
        yaxis_title="Generation (MW)",
        hovermode="x unified",
        template="plotly_white",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Heatmaps ─────────────────────────────────────────────────────────────
    st.markdown("#### Generation Heatmaps (Hour of Day × Day of Year)")

    daily_slots = 24
    n_days = len(RE_output) // daily_slots

    def build_heatmap_matrix(series, n_days, daily_slots):
        mat = np.full((n_days, daily_slots), np.nan)
        for d in range(n_days):
            start = d * daily_slots
            end   = start + daily_slots
            if end <= len(series):
                mat[d, :] = series.values[start:end]
        return mat

    hc1, hc2 = st.columns(2)
    with hc1:
        solar_mat = build_heatmap_matrix(RE_output["Solar"], n_days, daily_slots)
        fig_s = go.Figure(go.Heatmap(
            z=solar_mat, colorscale="YlOrRd",
            colorbar=dict(title="MW"),
            hovertemplate="Day %{y}<br>Hour %{x}<br>Solar: %{z:.3f} MW<extra></extra>"
        ))
        fig_s.update_layout(
            title="Solar Generation", xaxis_title="Hour of Day",
            yaxis_title="Day of Year", height=400, template="plotly_white"
        )
        st.plotly_chart(fig_s, use_container_width=True)

    with hc2:
        wind_mat = build_heatmap_matrix(RE_output["Wind"], n_days, daily_slots)
        fig_w = go.Figure(go.Heatmap(
            z=wind_mat, colorscale="Blues",
            colorbar=dict(title="MW"),
            hovertemplate="Day %{y}<br>Hour %{x}<br>Wind: %{z:.3f} MW<extra></extra>"
        ))
        fig_w.update_layout(
            title="Wind Generation", xaxis_title="Hour of Day",
            yaxis_title="Day of Year", height=400, template="plotly_white"
        )
        st.plotly_chart(fig_w, use_container_width=True)

    # ── Monthly summary table ────────────────────────────────────────────────
    st.markdown("#### Monthly Generation Summary")

    import calendar
    monthly_rows = []
    cumulative_hour = 0
    for m in range(1, 13):
        days_in_month = calendar.monthrange(fy, m)[1]
        hours = days_in_month * 24
        end_h = min(cumulative_hour + hours, len(RE_output))
        slice_ = RE_output.iloc[cumulative_hour:end_h]
        monthly_rows.append({
            "Month":              calendar.month_name[m],
            "Solar (MWh)":        round(slice_["Solar"].sum(), 1),
            "Wind (MWh)":         round(slice_["Wind"].sum(), 1),
            "Total RE (MWh)":     round(slice_["Solar"].sum() + slice_["Wind"].sum(), 1),
            "Peak Solar (MW)":    round(slice_["Solar"].max(), 3),
            "Peak Wind (MW)":     round(slice_["Wind"].max(), 3),
        })
        cumulative_hour += hours

    monthly_df = pd.DataFrame(monthly_rows)
    st.dataframe(monthly_df, hide_index=True, use_container_width=True)

    # ── Download section ─────────────────────────────────────────────────────
    st.divider()
    st.subheader("Download Outputs")

    dc1, dc2 = st.columns(2)

    with dc1:
        st.markdown("**RE.xlsx** — ready for the Demand Flexibility Model")
        st.caption("Drop this file directly into the DF model as the Renewable Energy input.")
        buf_re = io.BytesIO()
        with pd.ExcelWriter(buf_re, engine="openpyxl") as writer:
            RE_output.to_excel(writer, index=False, sheet_name="RE")
        st.download_button(
            "⬇️ Download RE.xlsx",
            data=buf_re.getvalue(),
            file_name=f"RE_{fy}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

    with dc2:
        st.markdown("**Full results CSV** — hourly Solar + Wind with timestamp")
        RE_with_ts = RE_output.copy()
        try:
            RE_with_ts.insert(
                0, "Timestamp",
                pd.date_range(f"{fy}-01-01", periods=len(RE_output), freq="h")
            )
        except Exception:
            pass
        buf_csv = io.BytesIO()
        RE_with_ts.to_csv(buf_csv, index=False)
        st.download_button(
            "⬇️ Download forecast CSV",
            data=buf_csv.getvalue(),
            file_name=f"RE_forecast_{fy}.csv",
            mime="text/csv",
            use_container_width=True
        )

    # ── Feed to DF model button ──────────────────────────────────────────────
    st.divider()
    if st.button("📤 Use this forecast as RE input in the DF Model", use_container_width=True):
        st.session_state["re"] = RE_output
        st.success(
            "✅ RE forecast loaded into the DF Model session. "
            "Go to the main page and re-run the analysis — "
            "no need to re-upload RE.xlsx manually."
        )