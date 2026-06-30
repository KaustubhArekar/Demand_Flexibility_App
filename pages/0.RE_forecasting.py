import streamlit as st
import pandas as pd
import numpy as np
import io
import pytz
import calendar
import plotly.graph_objects as go

# ── Lazy imports for heavy dependencies ──────────────────────────────────────
def _import_ml():
    from sklearn.preprocessing import MinMaxScaler
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Input
    import tensorflow as tf
    tf.random.set_seed(455)
    np.random.seed(455)
    return MinMaxScaler, Sequential, LSTM, Dense, Input

def _import_pvlib():
    from pvlib import location, irradiance
    return location, irradiance

EXPECTED_LEN = 8760

st.title("RE Generation Forecasting")
st.write(
    "Forecast hourly weather parameters for a target year using LSTM models, "
    "review the forecast, then estimate plant-wise and aggregated solar and wind "
    "generation. Output is a ready-to-use **RE.xlsx** for the Demand Flexibility Model."
)

# ════════════════════════════════════════════════════════════════════════════
# Core ANN/LSTM helper functions
# ════════════════════════════════════════════════════════════════════════════
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
    _, Sequential, LSTM, Dense, Input = _import_ml()
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


def predict_series(model, full_series_scaled, test_len, n_steps, scaler, expected_len=EXPECTED_LEN):
    inputs = full_series_scaled[len(full_series_scaled) - test_len - n_steps:]
    inputs = inputs.reshape(-1, 1)
    X_test, _ = split_sequence(inputs, n_steps)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    predicted = model.predict(X_test, verbose=0)
    predicted = scaler.inverse_transform(predicted)
    # Defensive cap — keep every forecasted weather series at exactly expected_len
    if len(predicted) > expected_len:
        predicted = predicted[:expected_len]
    return predicted


def get_transposition_factor(lat, lon, forecast_year):
    location_mod, irradiance_mod = _import_pvlib()
    tz   = pytz.timezone("Asia/Kolkata")
    site = location_mod.Location(lat, lon, tz=tz)

    dates = pd.date_range(f"{int(forecast_year)}-01-01", f"{int(forecast_year)}-12-31", freq="D")

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
        day_df = pd.DataFrame({"GHI": clearsky["ghi"], "POA": POA["poa_global"]}).reset_index(drop=True)
        Final_irr = pd.concat([Final_irr, day_df], axis=1)

    Final_irr = Final_irr.transpose()
    GHI_mat = Final_irr.iloc[0::2].reset_index(drop=True)
    POA_mat = Final_irr.iloc[1::2].reset_index(drop=True)
    TF_10min = (POA_mat / GHI_mat).set_index(dates.strftime("%d/%m/%Y"))

    TF_hr = pd.DataFrame()
    for x in range(0, 144, 6):
        col = TF_10min.iloc[:, x:x + 6].mean(axis=1).to_frame()
        TF_hr = pd.concat([TF_hr, col], axis=1)
    TF_hr.columns = range(24)

    row_wise = [val for _, row in TF_hr.iterrows() for val in row]
    return pd.DataFrame(row_wise, columns=["Value"])


def compute_solar_power(predicted_GHI, predicted_Temp, TF_hr_new, info_solar):
    """Pnom is read directly in MW (per the updated Info_Solar.csv convention).
    Output Pac is in MW — same unit as Pnom, since the formula scales linearly."""
    Pnom   = float(info_solar.iloc[2].iloc[0])   # MW
    nDRT   = float(info_solar.iloc[3].iloc[0])
    alpha  = float(info_solar.iloc[4].iloc[0])
    A_PV   = float(info_solar.iloc[6].iloc[0])

    Tc_STC, GT_STC = 25, 1
    U0, U1, WS = 29, 0, 1
    Tau_alpha   = 0.9
    nmp_STC     = Pnom / (A_PV * GT_STC) / 100

    ghi_df  = pd.DataFrame(predicted_GHI, columns=["Value"])
    ghi_df[ghi_df < 1] = 0
    temp_df = pd.DataFrame(predicted_Temp, columns=["Value"])
    TF_hr_new = TF_hr_new.fillna(1).reset_index(drop=True)

    estimated_POA = TF_hr_new * ghi_df
    # GT must be expressed in the same units as GT_STC (1 kW/m^2, standard test
    # conditions). estimated_POA (derived from GHI) is in W/m^2, so divide by
    # 1000 to convert W/m^2 -> kW/m^2. Previously this used /100, which left GT
    # roughly 10x too large at peak sun, inflating Pdc well past Pnom.
    GT = estimated_POA / 1000
    Ta = temp_df

    Tc  = Ta + estimated_POA * ((Tau_alpha * (1 - nmp_STC)) / (U0 + WS * U1))
    Pdc = Pnom * nDRT * (GT / GT_STC) * (1 + alpha * (Tc - Tc_STC))
    Pac = 0.95 * Pdc   # MW
    Pac.columns = ["AC Power (MW)"]
    return Pac


def compute_wind_power(predicted_WS, info_wind, expected_len=EXPECTED_LEN):
    """Prtdwn is read in MW (per the updated Info_Wind.csv convention) and
    converted internally to Watts, since the turbine swept-area formula
    (D = sqrt(4P / (pi*rho*v^3*Cp))) is dimensionally an SI/Watts formula.
    Output Pwn is in MW."""
    Prtdwn_MW = float(info_wind.iloc[2].iloc[0])   # MW, as entered by user
    Prtdwn_W  = Prtdwn_MW * 1_000_000              # convert to W for the physics formula
    Vctin  = float(info_wind.iloc[3].iloc[0])
    Vrtdwn = float(info_wind.iloc[4].iloc[0])
    Vctout = float(info_wind.iloc[5].iloc[0])

    rho, Cp, Pi = 1.225, 0.4, np.pi
    D    = np.sqrt((4 * Prtdwn_W) / (Pi * rho * (Vrtdwn ** 3) * Cp))
    A_WT = Pi * ((D / 2) ** 2)

    n_rows = min(len(predicted_WS), expected_len)
    Pwn = np.zeros(n_rows)
    for i in range(n_rows):
        ws = float(predicted_WS[i])
        if ws < Vctin or ws > Vctout:
            Pwn[i] = 0
        elif ws > Vrtdwn:
            Pwn[i] = Prtdwn_MW                                          # MW directly
        else:
            Pwn[i] = (0.5 * rho * A_WT * (ws ** 3)) / 1_000_000          # W → MW
    return pd.DataFrame(Pwn, columns=["Wind Power (MW)"])


def build_heatmap_matrix(series, n_days, daily_slots=24):
    mat = np.full((n_days, daily_slots), np.nan)
    vals = np.asarray(series)
    for d in range(n_days):
        start, end = d * daily_slots, d * daily_slots + daily_slots
        if end <= len(vals):
            mat[d, :] = vals[start:end]
    return mat


# ════════════════════════════════════════════════════════════════════════════
# STAGE 1 — Configure plants & upload weather files
# ════════════════════════════════════════════════════════════════════════════
st.divider()
st.subheader("Stage 1: Plant Weather Data")

col_np, col_yr = st.columns([1, 2])
with col_np:
    num_plants = st.number_input(
        "Number of sites", min_value=1, max_value=20, value=1, step=1,
        key="re_num_plants_input",
        help="Total number of solar/wind plant sites needing weather forecasts. "
             "If two plants share one weather station, you can still upload the "
             "same file for both."
    )

with col_yr:
    col_ts, col_te, col_fc = st.columns(3)
    with col_ts:
        tstart = st.number_input("Training start year", min_value=1990, max_value=2020, value=2001, key="re_tstart")
    with col_te:
        tend = st.number_input("Training end year", min_value=2000, max_value=2023, value=2022, key="re_tend")
    with col_fc:
        forecast_year = st.number_input("Forecast year", min_value=2023, max_value=2035, value=2023, key="re_fyear")

st.caption(f"Training data: years **{int(tstart)} – {int(tend)}** | Forecast target: **{int(forecast_year)}**")

st.markdown(
    "Upload one **weather CSV** per plant site. Each file needs hourly rows with "
    "columns: `Timestamp`, `Temperature`, `GHI`, `Wind Speed`. Plant capacity and "
    "turbine/module specs are **not** needed yet — those are entered in Stage 3, "
    "after you've reviewed the weather forecast."
)

weather_uploads = {}
for i in range(int(num_plants)):
    with st.expander(f"📡 Site {i+1} — Weather Data", expanded=(i == 0)):
        site_name = st.text_input("Site name", value=f"Site_{i+1}", key=f"re_sitename_{i}")
        w_file = st.file_uploader(
            "Weather CSV (Timestamp, Temperature, GHI, Wind Speed)",
            type="csv", key=f"re_weather_{i}"
        )
        weather_uploads[i] = {"name": site_name, "weather": w_file}

weather_ok = all(w["weather"] is not None for w in weather_uploads.values())
if not weather_ok:
    st.warning("⚠️ Upload a weather CSV for every site before forecasting.")

with st.expander("⚙️ Advanced LSTM settings"):
    ac1, ac2 = st.columns(2)
    with ac1:
        n_steps  = st.number_input("Look-back steps (n_steps)", min_value=5, max_value=50, value=10, key="re_nsteps")
        epochs_n = st.number_input("Training epochs", min_value=1, max_value=50, value=5, key="re_epochs")
    with ac2:
        lstm_units = st.number_input("LSTM units", min_value=32, max_value=256, value=125, step=16, key="re_units")
        batch_size = st.number_input("Batch size", min_value=8, max_value=128, value=32, step=8, key="re_batch")

# ════════════════════════════════════════════════════════════════════════════
# STAGE 2 — Forecast weather parameters
# ════════════════════════════════════════════════════════════════════════════
st.divider()
st.subheader("Stage 2: Forecast Weather Parameters")
st.caption(
    "Trains an LSTM per site for Temperature, GHI, and Wind Speed, then forecasts "
    f"the full target year ({int(EXPECTED_LEN)} hourly slots)."
)

forecast_btn = st.button(
    "🌦️ Forecast Weather", type="primary", use_container_width=True,
    disabled=not weather_ok
)

if forecast_btn:
    MinMaxScaler, _, _, _, _ = _import_ml()
    weather_forecasts = {}   # {site_idx: {"name":..., "Temperature":arr, "GHI":arr, "Wind Speed":arr}}

    overall_bar = st.progress(0, text="Starting weather forecast…")
    log_area = st.empty()

    features = ["Temperature", "GHI", "Wind Speed"]

    for idx, site in weather_uploads.items():
        sname = site["name"]
        overall_bar.progress(idx / int(num_plants), text=f"Forecasting {sname} ({idx+1}/{int(num_plants)})…")

        try:
            dataset = pd.read_csv(site["weather"], parse_dates=["Timestamp"])
        except Exception as e:
            st.error(f"{sname}: failed to read weather CSV — {e}")
            continue

        dataset = dataset.sort_values("Timestamp")
        dup = dataset["Timestamp"].duplicated().sum()
        if dup > 0:
            st.warning(f"⚠️ {sname}: removed {dup} duplicate timestamp row(s).")
            dataset = dataset.drop_duplicates(subset="Timestamp", keep="first")
        dataset = dataset.set_index("Timestamp")

        required_cols = {"Temperature", "GHI", "Wind Speed"}
        if not required_cols.issubset(dataset.columns):
            st.error(f"{sname}: weather CSV must contain columns {required_cols}. Found: {list(dataset.columns)}")
            continue

        train_data = dataset.loc[str(int(tstart)):str(int(tend))]
        test_data  = dataset.loc[str(int(tend + 1)):]
        if len(test_data) == 0:
            st.error(f"{sname}: no data found after year {int(tend)}. Check the Timestamp range.")
            continue

        # ── Diagnostic: warn clearly if the post-training-year data is incomplete ──
        if len(test_data) < EXPECTED_LEN:
            actual_days = len(test_data) / 24
            last_ts = test_data.index.max()
            st.warning(
                f"⚠️ **{sname}**: weather data after year {int(tend)} only covers "
                f"{len(test_data)} hours (~{actual_days:.0f} days), not a full year "
                f"({EXPECTED_LEN} hours). Data ends at **{last_ts}**. "
                f"The remaining slots will be filled using the same calendar hours "
                f"from year {int(tend)} of your training data (seasonal pattern), "
                f"not a flat constant — but this is a fallback, not a true forecast "
                f"for that period. Upload weather data covering the full target year "
                f"for an accurate result."
            )

        site_forecast = {"name": sname}
        sc_bar = st.progress(0, text=f"{sname} — training…")

        for fi, feat in enumerate(features):
            log_area.info(f"📡 **{sname}** — training LSTM for **{feat}**…")
            sc = MinMaxScaler(feature_range=(0, 1))
            train_vals = train_data[feat].values.reshape(-1, 1)
            train_scaled = sc.fit_transform(train_vals)

            model = train_lstm(train_scaled, int(n_steps), int(lstm_units), int(epochs_n), int(batch_size))

            full_series = dataset[feat].values.reshape(-1, 1)
            full_scaled = sc.transform(full_series)
            pred = predict_series(model, full_scaled, len(test_data), int(n_steps), sc, EXPECTED_LEN)
            pred = pred.flatten()

            if feat == "GHI":
                pred = np.where(pred < 1, 0, pred)
            if feat == "Wind Speed":
                pred = np.where(pred < 0, 0, pred)

            # ── Fill any shortfall using the same calendar hours from the final
            #    training year, instead of freezing on one repeated constant.
            #    This preserves daily/seasonal shape for the missing period
            #    rather than producing a visibly flat tail.
            if len(pred) < EXPECTED_LEN:
                shortfall = EXPECTED_LEN - len(pred)
                last_train_year_data = dataset.loc[str(int(tend)):str(int(tend))][feat].values
                if len(last_train_year_data) > 0:
                    # Tile the last training year's pattern to cover the shortfall,
                    # starting from where the real forecast left off
                    start_pos = len(pred) % len(last_train_year_data)
                    filler = np.resize(
                        np.roll(last_train_year_data, -start_pos), shortfall
                    )
                else:
                    filler = np.full(shortfall, pred[-1] if len(pred) else 0)
                pred = np.concatenate([pred, filler])

            site_forecast[feat] = pred
            sc_bar.progress((fi + 1) / len(features), text=f"{sname} — {feat} done")

        site_forecast["real_forecast_len"] = len(test_data)
        weather_forecasts[idx] = site_forecast

    overall_bar.progress(1.0, text="Weather forecast complete ✓")
    log_area.empty()

    st.session_state["re_weather_forecasts"] = weather_forecasts
    st.session_state["re_forecast_year"]     = int(forecast_year)
    st.session_state["re_site_names"]        = {i: w["name"] for i, w in weather_uploads.items()}
    st.success(f"✅ Weather forecast complete for {len(weather_forecasts)} site(s).")
    st.rerun()


# ── Display weather forecast results (persisted) ─────────────────────────────
if "re_weather_forecasts" in st.session_state:
    weather_forecasts = st.session_state["re_weather_forecasts"]
    fy = st.session_state["re_forecast_year"]

    st.markdown("#### Weather Forecast Review")
    st.caption(f"Forecast year: {fy} | {len(weather_forecasts)} site(s)")

    site_options = {i: wf["name"] for i, wf in weather_forecasts.items()}
    review_site = st.selectbox(
        "Select site to review",
        options=list(site_options.keys()),
        format_func=lambda i: site_options[i],
        key="re_review_site"
    )

    wf = weather_forecasts[review_site]
    real_len = wf.get("real_forecast_len", EXPECTED_LEN)

    rk1, rk2, rk3 = st.columns(3)
    rk1.metric("Avg Temperature", f"{wf['Temperature'].mean():.1f} °C")
    rk2.metric("Avg GHI", f"{wf['GHI'].mean():.1f} W/m²")
    rk3.metric("Avg Wind Speed", f"{wf['Wind Speed'].mean():.2f} m/s")

    if real_len < EXPECTED_LEN:
        st.error(
            f"⚠️ Only **{real_len}** of {EXPECTED_LEN} hours ({real_len/24:.0f} days) "
            f"are a true LSTM forecast for {wf['name']}. The remaining "
            f"{EXPECTED_LEN - real_len} hours (shown shaded below) are filled using "
            f"the seasonal pattern from your last training year — not a real forecast. "
            f"Upload weather data covering the full target year to fix this."
        )

    fig_w = go.Figure()
    fig_w.add_trace(go.Scatter(y=wf["Temperature"], name="Temperature (°C)", line=dict(color="firebrick", width=1)))
    fig_w.add_trace(go.Scatter(y=wf["GHI"], name="GHI (W/m²)", line=dict(color="orange", width=1), yaxis="y2"))
    fig_w.add_trace(go.Scatter(y=wf["Wind Speed"], name="Wind Speed (m/s)", line=dict(color="steelblue", width=1), yaxis="y3"))
    if real_len < EXPECTED_LEN:
        fig_w.add_vrect(
            x0=real_len, x1=EXPECTED_LEN,
            fillcolor="lightgray", opacity=0.35, line_width=0,
            annotation_text="Seasonal fallback (not a real forecast)",
            annotation_position="top left"
        )
        fig_w.add_vline(
            x=real_len, line_dash="dash", line_color="gray",
            annotation_text="Real forecast ends here", annotation_position="top right"
        )
    fig_w.update_layout(
        title=f"Forecasted Weather — {wf['name']} ({fy})",
        xaxis_title="Hour of year",
        yaxis=dict(title="Temperature (°C)", side="left"),
        yaxis2=dict(title="GHI (W/m²)", overlaying="y", side="right"),
        yaxis3=dict(title="Wind Speed (m/s)", overlaying="y", side="right", anchor="free", position=0.97),
        hovermode="x unified", template="plotly_white", height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_w, use_container_width=True)

    with st.expander("📊 Weather heatmaps (Hour of day × Day of year)"):
        n_days = len(wf["Temperature"]) // 24
        hc1, hc2, hc3 = st.columns(3)
        with hc1:
            mat = build_heatmap_matrix(wf["Temperature"], n_days)
            f = go.Figure(go.Heatmap(z=mat, colorscale="RdYlBu_r", colorbar=dict(title="°C")))
            f.update_layout(title="Temperature", xaxis_title="Hour", yaxis_title="Day", height=350, template="plotly_white")
            st.plotly_chart(f, use_container_width=True)
        with hc2:
            mat = build_heatmap_matrix(wf["GHI"], n_days)
            f = go.Figure(go.Heatmap(z=mat, colorscale="YlOrRd", colorbar=dict(title="W/m²")))
            f.update_layout(title="GHI", xaxis_title="Hour", yaxis_title="Day", height=350, template="plotly_white")
            st.plotly_chart(f, use_container_width=True)
        with hc3:
            mat = build_heatmap_matrix(wf["Wind Speed"], n_days)
            f = go.Figure(go.Heatmap(z=mat, colorscale="Blues", colorbar=dict(title="m/s")))
            f.update_layout(title="Wind Speed", xaxis_title="Hour", yaxis_title="Day", height=350, template="plotly_white")
            st.plotly_chart(f, use_container_width=True)

    with st.expander("⬇️ Download forecasted weather data"):
        wdf = pd.DataFrame({
            "Hour": np.arange(len(wf["Temperature"])),
            "Temperature": wf["Temperature"],
            "GHI": wf["GHI"],
            "Wind Speed": wf["Wind Speed"],
        })
        buf = io.BytesIO()
        wdf.to_csv(buf, index=False)
        st.download_button(
            f"Download {wf['name']} weather forecast CSV",
            data=buf.getvalue(), file_name=f"{wf['name']}_weather_{fy}.csv",
            mime="text/csv", key=f"re_wdl_{review_site}"
        )

    # ════════════════════════════════════════════════════════════════════════
    # STAGE 3 — Plant specs (capacity, turbine/module info)
    # ════════════════════════════════════════════════════════════════════════
    st.divider()
    st.subheader("Stage 3: Plant Specifications")
    st.caption(
        "Now that the weather forecast looks correct, assign solar and/or wind plant "
        "specifications to each weather site. One weather site can host a Solar plant, "
        "a Wind plant, or both."
    )

    plant_specs = {}
    for idx, wf in weather_forecasts.items():
        with st.expander(f"🏭 {wf['name']} — Plant Configuration", expanded=(idx == 0)):
            ptype = st.selectbox(
                "Plant type at this site", options=["Solar", "Wind", "Solar + Wind", "None"],
                key=f"re_ptype_{idx}"
            )

            solar_info_file, wind_info_file = None, None
            pc1, pc2 = st.columns(2)
            with pc1:
                if ptype in ("Solar", "Solar + Wind"):
                    st.markdown("**☀️ Solar Info CSV**")
                    solar_info_file = st.file_uploader(
                        "Parameter, Value — Latitude, Longitude, Pnom (MW), nDRT, alpha, gamma, A_PV (m²)",
                        type="csv", key=f"re_solarinfo_{idx}"
                    )
            with pc2:
                if ptype in ("Wind", "Solar + Wind"):
                    st.markdown("**💨 Wind Info CSV**")
                    wind_info_file = st.file_uploader(
                        "Parameter, Value — Latitude, Longitude, Prtdwn (MW), Vctin, Vrtdwn, Vctout",
                        type="csv", key=f"re_windinfo_{idx}"
                    )

            plant_specs[idx] = {
                "type": ptype,
                "solar_info": solar_info_file,
                "wind_info": wind_info_file,
            }

    def _specs_complete(specs):
        for i, p in specs.items():
            if p["type"] in ("Solar", "Solar + Wind") and p["solar_info"] is None:
                return False, f"{weather_forecasts[i]['name']}: Solar Info CSV missing"
            if p["type"] in ("Wind", "Solar + Wind") and p["wind_info"] is None:
                return False, f"{weather_forecasts[i]['name']}: Wind Info CSV missing"
        return True, ""

    specs_ok, specs_msg = _specs_complete(plant_specs)
    if not specs_ok:
        st.warning(f"⚠️ {specs_msg}")

    # ════════════════════════════════════════════════════════════════════════
    # STAGE 4 — Estimate generation
    # ════════════════════════════════════════════════════════════════════════
    st.divider()
    st.subheader("Stage 4: Estimate Generation")

    estimate_btn = st.button(
        "⚡ Estimate Generation", type="primary", use_container_width=True,
        disabled=not specs_ok
    )

    if estimate_btn:
        final_solar = pd.DataFrame()
        final_wind  = pd.DataFrame()
        gen_progress = st.progress(0, text="Calculating plant generation…")
        n_active = sum(1 for p in plant_specs.values() if p["type"] != "None")
        done = 0

        for idx, spec in plant_specs.items():
            if spec["type"] == "None":
                continue
            sname = weather_forecasts[idx]["name"]
            wf = weather_forecasts[idx]

            if spec["type"] in ("Solar", "Solar + Wind"):
                try:
                    info_solar = pd.read_csv(spec["solar_info"]).set_index("Parameter")
                    lat = float(info_solar.iloc[0].iloc[0])
                    lon = float(info_solar.iloc[1].iloc[0])
                    Pnom_check = float(info_solar.iloc[2].iloc[0])
                    TF  = get_transposition_factor(lat, lon, fy)
                    p_solar = compute_solar_power(wf["GHI"], wf["Temperature"], TF, info_solar)
                    p_solar = p_solar.reset_index(drop=True)

                    # Sanity check: peak AC output should not meaningfully exceed
                    # nominal DC capacity (a small overshoot from temperature/
                    # irradiance effects is physically possible, but >10% over
                    # Pnom signals a units or formula issue worth investigating).
                    peak_val = p_solar.iloc[:, 0].max()
                    if peak_val > 1.10 * Pnom_check:
                        st.warning(
                            f"⚠️ {sname}: peak solar output ({peak_val:.1f} MW) exceeds "
                            f"nominal DC capacity ({Pnom_check:.1f} MW) by more than 10%. "
                            f"Check that Pnom in the Solar Info CSV is in MW and the "
                            f"weather GHI data looks reasonable."
                        )

                    p_solar.columns = [f"{sname}_solar"]
                    final_solar = pd.concat([final_solar, p_solar], axis=1)
                except Exception as e:
                    st.error(f"{sname}: solar generation calc failed — {e}")

            if spec["type"] in ("Wind", "Solar + Wind"):
                try:
                    info_wind = pd.read_csv(spec["wind_info"]).set_index("Parameter")
                    p_wind = compute_wind_power(wf["Wind Speed"], info_wind, EXPECTED_LEN)
                    p_wind = p_wind.reset_index(drop=True)
                    p_wind.columns = [f"{sname}_wind"]
                    final_wind = pd.concat([final_wind, p_wind], axis=1)
                except Exception as e:
                    st.error(f"{sname}: wind generation calc failed — {e}")

            done += 1
            gen_progress.progress(done / max(n_active, 1), text=f"{sname} done ({done}/{n_active})")

        gen_progress.progress(1.0, text="Generation estimate complete ✓")

        # Reset index defensively, align both series to EXPECTED_LEN
        final_solar = final_solar.reset_index(drop=True)
        final_wind  = final_wind.reset_index(drop=True)

        if not final_solar.empty and len(final_solar) != EXPECTED_LEN:
            final_solar = final_solar.iloc[:EXPECTED_LEN]
        if not final_wind.empty and len(final_wind) != EXPECTED_LEN:
            final_wind = final_wind.iloc[:EXPECTED_LEN]

        # Both compute_solar_power and compute_wind_power now return MW directly
        # (Pnom and Prtdwn are read in MW from the updated Info_Solar/Info_Wind CSVs),
        # so no further unit conversion is needed here.
        solar_agg = final_solar.sum(axis=1).values if not final_solar.empty else np.zeros(EXPECTED_LEN)
        wind_agg  = final_wind.sum(axis=1).values  if not final_wind.empty  else np.zeros(EXPECTED_LEN)

        n_slots = min(len(solar_agg), len(wind_agg))
        RE_output = pd.DataFrame({
            "Solar": solar_agg[:n_slots],
            "Wind":  wind_agg[:n_slots],
        })

        st.session_state["RE_forecast_output"] = RE_output
        st.session_state["RE_per_plant_solar"] = final_solar  # already MW, per plant
        st.session_state["RE_per_plant_wind"]  = final_wind
        st.success(
            f"✅ Generation estimate complete! Total Solar: **{RE_output['Solar'].sum():.1f} MWh** | "
            f"Total Wind: **{RE_output['Wind'].sum():.1f} MWh**"
        )
        st.rerun()


# ════════════════════════════════════════════════════════════════════════════
# RESULTS — Aggregated generation (persisted)
# ════════════════════════════════════════════════════════════════════════════
if "RE_forecast_output" in st.session_state:
    RE_output = st.session_state["RE_forecast_output"]
    fy = st.session_state.get("re_forecast_year", 2023)

    st.divider()
    st.subheader(f"Results — Forecast Year {fy}")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Solar Generation", f"{RE_output['Solar'].sum():,.1f} MWh")
    m2.metric("Total Wind Generation",  f"{RE_output['Wind'].sum():,.1f} MWh")
    m3.metric("Peak Solar Hour",        f"{RE_output['Solar'].max():.2f} MW")
    m4.metric("Peak Wind Hour",         f"{RE_output['Wind'].max():.2f} MW")

    st.markdown("#### Aggregated Hourly Generation Profile")
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=RE_output["Solar"], name="Solar (MW)", line=dict(color="orange", width=1)))
    fig.add_trace(go.Scatter(y=RE_output["Wind"], name="Wind (MW)", line=dict(color="steelblue", width=1)))
    fig.update_layout(
        xaxis_title=f"Hour of {fy}", yaxis_title="Generation (MW)",
        hovermode="x unified", template="plotly_white", height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Per-plant breakdown
    if "RE_per_plant_solar" in st.session_state and not st.session_state["RE_per_plant_solar"].empty:
        with st.expander("🏭 Per-plant Solar breakdown"):
            st.dataframe(st.session_state["RE_per_plant_solar"].describe().T, use_container_width=True)
    if "RE_per_plant_wind" in st.session_state and not st.session_state["RE_per_plant_wind"].empty:
        with st.expander("🏭 Per-plant Wind breakdown"):
            st.dataframe(st.session_state["RE_per_plant_wind"].describe().T, use_container_width=True)

    st.markdown("#### Generation Heatmaps (Hour of Day × Day of Year)")
    n_days = len(RE_output) // 24
    hc1, hc2 = st.columns(2)
    with hc1:
        mat = build_heatmap_matrix(RE_output["Solar"], n_days)
        f = go.Figure(go.Heatmap(z=mat, colorscale="YlOrRd", colorbar=dict(title="MW")))
        f.update_layout(title="Solar Generation", xaxis_title="Hour of Day", yaxis_title="Day of Year", height=400, template="plotly_white")
        st.plotly_chart(f, use_container_width=True)
    with hc2:
        mat = build_heatmap_matrix(RE_output["Wind"], n_days)
        f = go.Figure(go.Heatmap(z=mat, colorscale="Blues", colorbar=dict(title="MW")))
        f.update_layout(title="Wind Generation", xaxis_title="Hour of Day", yaxis_title="Day of Year", height=400, template="plotly_white")
        st.plotly_chart(f, use_container_width=True)

    st.markdown("#### Monthly Generation Summary")
    monthly_rows = []
    cum_hr = 0
    for m in range(1, 13):
        days = calendar.monthrange(fy, m)[1]
        hrs = days * 24
        end_h = min(cum_hr + hrs, len(RE_output))
        sl = RE_output.iloc[cum_hr:end_h]
        monthly_rows.append({
            "Month": calendar.month_name[m],
            "Solar (MWh)": round(sl["Solar"].sum(), 1),
            "Wind (MWh)": round(sl["Wind"].sum(), 1),
            "Total RE (MWh)": round(sl["Solar"].sum() + sl["Wind"].sum(), 1),
            "Peak Solar (MW)": round(sl["Solar"].max(), 3),
            "Peak Wind (MW)": round(sl["Wind"].max(), 3),
        })
        cum_hr += hrs
    st.dataframe(pd.DataFrame(monthly_rows), hide_index=True, use_container_width=True)

    st.divider()
    st.subheader("Download Outputs")
    dc1, dc2 = st.columns(2)
    with dc1:
        st.markdown("**RE.xlsx** — ready for the Demand Flexibility Model")
        buf_re = io.BytesIO()
        with pd.ExcelWriter(buf_re, engine="openpyxl") as writer:
            RE_output.to_excel(writer, index=False, sheet_name="RE")
        st.download_button(
            "⬇️ Download RE.xlsx", data=buf_re.getvalue(),
            file_name=f"RE_{fy}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
    with dc2:
        st.markdown("**Full results CSV** — hourly Solar + Wind")
        buf_csv = io.BytesIO()
        RE_output.to_csv(buf_csv, index=False)
        st.download_button(
            "⬇️ Download forecast CSV", data=buf_csv.getvalue(),
            file_name=f"RE_forecast_{fy}.csv", mime="text/csv",
            use_container_width=True
        )

    st.divider()
    if st.button("📤 Use this forecast as RE input in the DF Model", use_container_width=True):
        st.session_state["re"] = RE_output
        st.success(
            "✅ RE forecast loaded into the DF Model session. "
            "Go to the main page and re-run the analysis — no need to re-upload RE.xlsx."
        )