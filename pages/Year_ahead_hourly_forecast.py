import streamlit as st
import pandas as pd
import numpy as np
import math
import io
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ════════════════════════════════════════════════════════════════════════════
# ANN — pure numpy implementation (from notebook)
# ════════════════════════════════════════════════════════════════════════════

def initialise_parameters(layer_dimensions):
    np.random.seed(1)
    parameters = {}
    for l in range(1, len(layer_dimensions)):
        # He initialisation for ReLU hidden layers
        parameters['W' + str(l)] = (
            np.random.randn(layer_dimensions[l], layer_dimensions[l - 1])
            * np.sqrt(2.0 / layer_dimensions[l - 1])
        )
        parameters['b' + str(l)] = np.zeros((layer_dimensions[l], 1))
    return parameters


def linear_forward(X, W, b):
    z = np.dot(W, X) + b
    cache = (X, W, b)
    return z, cache


def forward_activation(a_prev, W, b, activation):
    z, linear_cache = linear_forward(a_prev, W, b)
    if activation == 'relu':
        g = np.where(z > 0, z, 0)
        activation_cache = z
    elif activation == 'tanh':
        g = np.tanh(z)
        activation_cache = z
    else:   # linear (output layer)
        g = z
        activation_cache = z
    cache = (linear_cache, activation_cache)
    return g, cache


def forward_propagation(X, parameters):
    layers = len(parameters) // 2
    caches = []
    A = X
    for l in range(1, layers):
        A, cache = forward_activation(A, parameters['W' + str(l)],
                                      parameters['b' + str(l)], 'relu')
        caches.append(cache)
    Al, cache = forward_activation(A, parameters['W' + str(layers)],
                                   parameters['b' + str(layers)], 'linear')
    caches.append(cache)
    return Al, caches


def compute_cost(output, y):
    y = np.array(y)
    weights = 1 + 10 * (y / y.max())
    diff = output - y
    cost = (1 / len(y.flatten())) * np.sum(weights * np.square(diff))
    return cost


def relu_backward(dAl, z):
    dZ = np.array(dAl, copy=True)
    dZ[z <= 0] = 0
    return dZ


def linear_backward(dZ, cache):
    a_prev, W, b = cache
    m = a_prev.shape[1]
    dW = np.dot(dZ, a_prev.T) / m
    db = np.squeeze(np.sum(dZ, axis=1, keepdims=True) / m)
    dA_prev = np.dot(W.T, dZ)
    return dA_prev, dW, db


def model_backward_propagation(Al, y, caches):
    grads = {}
    L = len(caches)
    dAl = (Al - y) / Al.shape[1]

    # Last layer (linear activation — gradient passes through directly)
    current_cache = caches[-1]
    aa_prev = current_cache[0][0]
    ww = current_cache[0][1]
    grads['dA' + str(L)], grads['dW' + str(L)], grads['db' + str(L)] = \
        linear_backward(dAl, current_cache[0])

    # Hidden layers (ReLU)
    for l in reversed(range(L - 1)):
        current_cache = caches[l]
        dAl_prev = grads['dA' + str(l + 2)]
        dZ = relu_backward(dAl_prev, current_cache[1])
        dA_prev, dW, db = linear_backward(dZ, current_cache[0])
        grads['dA' + str(l + 1)] = dA_prev
        grads['dW' + str(l + 1)] = dW
        grads['db' + str(l + 1)] = db
    return grads


def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2
    for l in range(L):
        parameters['W' + str(l + 1)] -= learning_rate * grads['dW' + str(l + 1)]
        db = learning_rate * grads['db' + str(l + 1)]
        parameters['b' + str(l + 1)] -= db.reshape(parameters['b' + str(l + 1)].shape)
    return parameters


def ann_train(X_train, y_train, layer_dims, learning_rate, num_iterations):
    """Train ANN and return (parameters, cost_history)."""
    parameters = initialise_parameters(layer_dims)
    costs = []
    for i in range(num_iterations):
        Al, caches = forward_propagation(X_train, parameters)
        cost = compute_cost(Al, y_train)
        costs.append(cost)
        grads = model_backward_propagation(Al, y_train, caches)
        parameters = update_parameters(parameters, grads, learning_rate)
    return parameters, np.array(costs)


def ann_predict(X, parameters):
    output, _ = forward_propagation(X, parameters)
    return output[0]


def compute_metrics(y_actual, y_pred):
    rmse  = np.sqrt(np.mean((y_actual - y_pred) ** 2))
    nrmse = 100 * rmse / (y_actual.max() - y_actual.min() + 1e-10)
    mask  = (np.abs(y_actual) + np.abs(y_pred)) > 1e-6
    smape = np.mean(
        np.abs(y_actual[mask] - y_pred[mask])
        / ((np.abs(y_actual[mask]) + np.abs(y_pred[mask])) / 2)
    ) * 100 if mask.any() else 0.0
    mae   = np.mean(np.abs(y_actual - y_pred))
    r2    = 1 - np.sum((y_actual - y_pred) ** 2) / (
        np.sum((y_actual - y_actual.mean()) ** 2) + 1e-10)
    return {"RMSE": round(rmse, 4),
            "nRMSE (%)": round(nrmse, 2),
            "sMAPE (%)": round(smape, 2),
            "MAE": round(mae, 4),
            "R²": round(r2, 4)}


# ════════════════════════════════════════════════════════════════════════════
# PAGE
# ════════════════════════════════════════════════════════════════════════════
st.title("ANN Time-Series Forecasting")
st.write(
    "A generic year-ahead forecasting tool using a custom numpy ANN. "
    "Upload historical data, tune the model, evaluate on a held-out test set, "
    "then generate a full-year forecast."
)

USE_CASE_INFO = {
    "State Electricity Demand": {
        "desc": "Year-ahead hourly state demand forecast. "
                "Columns = years of historical data (sorted hourly values). "
                "The last column is the target year to forecast.",
        "x_label": "Hour of year",
        "y_label": "Demand (MW)",
        "layer_default": "5,5,1",
        "lr_default": 100.0,
        "iter_default": 5000,
    },
    "IEX Market Price (MCP)": {
        "desc": "Year-ahead DAM MCP forecast. "
                "Columns = years of sorted MCP values (₹/MWh). "
                "Last column is the target year.",
        "x_label": "Hour of year",
        "y_label": "MCP (₹/MWh)",
        "layer_default": "2,2,1",
        "lr_default": 100.0,
        "iter_default": 3000,
    },
    "Custom / Generic": {
        "desc": "Any time-series data. "
                "Columns = feature years / input variables (sorted values). "
                "Last column is the target variable to forecast.",
        "x_label": "Sample index",
        "y_label": "Value",
        "layer_default": "4,4,1",
        "lr_default": 0.01,
        "iter_default": 3000,
    },
}

# ── Stage 1: Use case & data upload ─────────────────────────────────────────
st.divider()
st.subheader("Stage 1: Select Use Case & Upload Data")

uc_col, info_col = st.columns([1, 2])
with uc_col:
    use_case = st.selectbox("Use case", options=list(USE_CASE_INFO.keys()))

info = USE_CASE_INFO[use_case]
with info_col:
    st.info(f"**{use_case}:** {info['desc']}")

with st.expander("📋 Expected file format", expanded=False):
    st.markdown("""
**CSV or Excel file with the following structure:**

| Feature 1| Feature 2 | Feature 3 | ... |  (target) |
|----------|-----------|-----------|-----|-----------|
| 450.2    | 461.5     | 478.0     | ... | 495.1     |
| 423.8    | 438.2     | 451.3     | ... | 467.4     |
| ...      | ...       | ...       | ... | ...       |

- Each **column** represents one input feature.
- Feature can be prevous year values, datetime based features like hour, month, day-of-week etc.
- Each **row** is one time slot (sorted chronologically — e.g., hour 1 to hour 8760).
- The **last column** is the target variable the model will learn to predict.
- No index column needed; column headers should be year labels or feature names.
- **All columns except the last** are used as input features (X).
- **Last column** is the target (Y).
    """)

uploaded_file = st.file_uploader(
    "Upload historical data file",
    type=["csv", "xlsx"],
    key="ann_data_upload"
)

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            raw_data = pd.read_csv(uploaded_file)
        else:
            raw_data = pd.read_excel(uploaded_file)
        st.session_state["ann_raw_data"] = raw_data
        st.success(f"✅ Loaded: {raw_data.shape[0]} rows × {raw_data.shape[1]} columns")
        st.dataframe(raw_data.head(5), use_container_width=True)
    except Exception as e:
        st.error(f"Failed to read file: {e}")
    


        
        

# ── Stage 2: Model configuration ────────────────────────────────────────────
st.divider()
st.subheader("Stage 2: Configure & Train Model")

if "ann_raw_data" not in st.session_state:
    st.warning("⚠️ Upload data above before configuring the model.")
    st.stop()

raw_data = st.session_state["ann_raw_data"]
n_features = raw_data.shape[1] - 1   # all columns except last
n_samples  = raw_data.shape[0]

st.caption(
    f"Detected **{n_features}** input feature(s) and **{n_samples}** samples. "
    f"Input layer size is fixed at **{n_features}** neurons."
)

cfg1, cfg2, cfg3 = st.columns(3)

with cfg1:
    st.markdown("**Neural Network Architecture**")
    layer_str = st.text_input(
        "Layer sizes (comma-separated)",
        value=info["layer_default"],
        help=f"First value must be {n_features} (input features). "
             "Last value must be 1 (scalar output). "
             f"Example: {n_features},8,4,1"
    )
    train_split = st.slider(
        "Training split (%)", min_value=50, max_value=90, value=70, step=5,
        help="Percentage of data used for training; remainder is the test set"
    )

with cfg2:
    st.markdown("**Training Hyperparameters**")
    learning_rate = st.number_input(
        "Learning rate", min_value=0.100, max_value=100000.0,
        value=info["lr_default"], step=10.0, format="%.2f",
        help="Step size for gradient descent. Too high → divergence. Too low → slow convergence."
    )
    num_iterations = st.number_input(
        "Iterations", min_value=100, max_value=50000,
        value=info["iter_default"], step=500,
        help="Number of gradient descent steps. More iterations = longer training but potentially better fit."
    )

with cfg3:
    st.markdown("**Display**")
    cost_smooth = st.slider(
        "Cost curve smoothing (every N iters)", min_value=1, max_value=200,
        value=10, help="Plot one cost point every N iterations to reduce noise"
    )
    x_label = st.text_input("X-axis label", value=info["x_label"])
    y_label = st.text_input("Y-axis label", value=info["y_label"])

# ── Parse and validate layer dimensions ─────────────────────────────────────
try:
    user_layers = [int(x.strip()) for x in layer_str.split(",")]
    if user_layers[0] != n_features:
        st.warning(
            f"⚠️ First layer size should match number of input features ({n_features}). "
            f"Auto-correcting from {user_layers[0]} to {n_features}."
        )
        user_layers[0] = n_features
    if user_layers[-1] != 1:
        st.warning("⚠️ Last layer size must be 1 (scalar output). Auto-correcting.")
        user_layers[-1] = 1
    layer_dims = user_layers
    st.caption(f"Network: `{' → '.join(str(x) for x in layer_dims)}`")
except Exception:
    st.error("Invalid layer format. Use comma-separated integers, e.g. `5,8,4,1`")
    st.stop()

# ── Train button ─────────────────────────────────────────────────────────────
train_btn = st.button("Train Model", type="primary", use_container_width=True)

if train_btn:
    with st.spinner("Preparing data…"):
        X_all = raw_data.iloc[:, :-1].values.astype(float)
        Y_all = raw_data.iloc[:, -1].values.astype(float)

        # Store raw max for de-normalisation later
        X_max = X_all.max(axis=0)
        X_max[X_max == 0] = 1   # avoid div by zero
        Y_max = Y_all.max()
        if Y_max == 0:
            Y_max = 1.0

        Xn = X_all / X_max
        Yn = Y_all / Y_max

        n_train = int((train_split / 100) * n_samples)
        X_train = Xn[:n_train, :].T          # shape: (features, n_train)
        y_train = Yn[:n_train].reshape(1, -1) # shape: (1, n_train)
        X_test  = Xn[n_train:, :].T
        y_test  = Yn[n_train:].reshape(1, -1)

        st.session_state["ann_X_max"]   = X_max
        st.session_state["ann_Y_max"]   = Y_max
        st.session_state["ann_Xn"]      = Xn
        st.session_state["ann_Yn"]      = Yn
        st.session_state["ann_X_train"] = X_train
        st.session_state["ann_y_train"] = y_train
        st.session_state["ann_X_test"]  = X_test
        st.session_state["ann_y_test"]  = y_test
        st.session_state["ann_Y_all"]   = Y_all
        st.session_state["ann_n_train"] = n_train
        st.session_state["ann_layer_dims"] = layer_dims
        st.session_state["ann_x_label"] = x_label
        st.session_state["ann_y_label"] = y_label

    progress_bar = st.progress(0, text="Training ANN…")

    # Chunked training for progress updates
    chunk = max(1, int(num_iterations) // 20)
    parameters = initialise_parameters(layer_dims)
    all_costs = []

    for start_iter in range(0, int(num_iterations), chunk):
        end_iter = min(start_iter + chunk, int(num_iterations))
        iters_this_chunk = end_iter - start_iter
        params_chunk, costs_chunk = ann_train(
            X_train, y_train, layer_dims,
            learning_rate, iters_this_chunk
        )
        # Warm-start: use last params as init for next chunk
        if start_iter == 0:
            parameters = params_chunk
        else:
            # Re-train from current parameters
            for i in range(iters_this_chunk):
                Al, caches = forward_propagation(X_train, parameters)
                cost = compute_cost(Al, y_train)
                all_costs.append(cost)
                grads = model_backward_propagation(Al, y_train, caches)
                parameters = update_parameters(parameters, grads, learning_rate)

        if start_iter == 0:
            all_costs.extend(costs_chunk.tolist())

        pct = end_iter / int(num_iterations)
        progress_bar.progress(pct, text=f"Training… {end_iter}/{int(num_iterations)} iterations")

    progress_bar.progress(1.0, text="Training complete ✓")

    # Final clean training pass to get full cost history
    with st.spinner("Computing final cost curve…"):
        parameters, costs = ann_train(
            X_train, y_train, layer_dims, learning_rate, int(num_iterations)
        )

    st.session_state["ann_parameters"] = parameters
    st.session_state["ann_costs"]       = costs
    st.session_state["ann_lr"]          = learning_rate
    st.session_state["ann_iters"]       = int(num_iterations)
    st.session_state["ann_smooth"]      = cost_smooth
    st.success("✅ Training complete!")
    st.rerun()


# ════════════════════════════════════════════════════════════════════════════
# RESULTS — Training diagnostics + test performance
# ════════════════════════════════════════════════════════════════════════════
if "ann_parameters" not in st.session_state:
    st.stop()

parameters  = st.session_state["ann_parameters"]
costs       = st.session_state["ann_costs"]
X_train     = st.session_state["ann_X_train"]
y_train     = st.session_state["ann_y_train"]
X_test      = st.session_state["ann_X_test"]
y_test      = st.session_state["ann_y_test"]
Y_all       = st.session_state["ann_Y_all"]
Y_max       = st.session_state["ann_Y_max"]
n_train     = st.session_state["ann_n_train"]
x_label     = st.session_state["ann_x_label"]
y_label     = st.session_state["ann_y_label"]
smooth      = st.session_state.get("ann_smooth", 10)

st.divider()
st.subheader("Training Results")

# ── KPI row ──────────────────────────────────────────────────────────────────
final_train_cost = costs[-1]
train_pred = ann_predict(X_train, parameters)
test_pred  = ann_predict(X_test,  parameters)

y_train_actual = y_train[0] * Y_max
y_test_actual  = y_test[0]  * Y_max
train_pred_actual = train_pred * Y_max
test_pred_actual  = test_pred  * Y_max

train_metrics = compute_metrics(y_train_actual, train_pred_actual)
test_metrics  = compute_metrics(y_test_actual,  test_pred_actual)

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Final Cost (MSE)",  f"{final_train_cost:.6f}")
k2.metric("Test nRMSE",        f"{test_metrics['nRMSE (%)']:.2f}%")
k3.metric("Test sMAPE",        f"{test_metrics['sMAPE (%)']:.2f}%")
k4.metric("Test R²",           f"{test_metrics['R²']:.4f}")
k5.metric("Test MAE",          f"{test_metrics['MAE']:.4f}")

# ── Two-column layout: cost curve + test chart ───────────────────────────────
diag1, diag2 = st.columns([1, 1])

with diag1:
    st.markdown("#### Cost Curve Over Training")
    smooth_int = max(1, int(smooth))
    idx   = np.arange(0, len(costs), smooth_int)
    c_smooth = costs[idx]

    fig_cost = go.Figure()
    fig_cost.add_trace(go.Scatter(
        x=idx, y=c_smooth,
        mode="lines",
        line=dict(color="crimson", width=2),
        hovertemplate="Iteration %{x}<br>Cost: %{y:.6f}<extra></extra>"
    ))
    fig_cost.update_layout(
        xaxis_title="Iteration",
        yaxis_title="MSE Cost",
        template="plotly_white",
        height=380,
        margin=dict(t=30)
    )
    st.plotly_chart(fig_cost, use_container_width=True)
    st.caption(
        f"Learning rate: `{st.session_state['ann_lr']}` | "
        f"Iterations: `{st.session_state['ann_iters']}` | "
        f"Architecture: `{' → '.join(str(x) for x in st.session_state['ann_layer_dims'])}`"
    )

with diag2:
    st.markdown("#### Test Set: Actual vs Predicted")

    n_test = len(y_test_actual)
    x_range_test = np.arange(n_train, n_train + n_test)

    fig_test = go.Figure()
    fig_test.add_trace(go.Scatter(
        x=x_range_test, y=y_test_actual,
        mode="lines",
        name="Actual",
        line=dict(color="steelblue", width=1.5),
        hovertemplate=f"{x_label} %{{x}}<br>Actual: %{{y:.3f}}<extra></extra>"
    ))
    fig_test.add_trace(go.Scatter(
        x=x_range_test, y=test_pred_actual,
        mode="lines",
        name="Predicted",
        line=dict(color="orange", width=1.5, dash="dot"),
        hovertemplate=f"{x_label} %{{x}}<br>Predicted: %{{y:.3f}}<extra></extra>"
    ))
    fig_test.update_layout(
        xaxis_title=x_label,
        yaxis_title=y_label,
        template="plotly_white",
        height=380,
        margin=dict(t=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified"
    )
    st.plotly_chart(fig_test, use_container_width=True)

# ── Full dataset overlay: train + test ───────────────────────────────────────
st.markdown("#### Full Dataset: Actual vs Model Output")

full_pred_actual = np.concatenate([train_pred_actual, test_pred_actual])
full_range = np.arange(len(full_pred_actual))

fig_full = go.Figure()
fig_full.add_trace(go.Scatter(
    x=full_range, y=Y_all,
    mode="lines",
    name="Actual",
    line=dict(color="steelblue", width=1),
    hovertemplate=f"{x_label} %{{x}}<br>Actual: %{{y:.3f}}<extra></extra>"
))
fig_full.add_trace(go.Scatter(
    x=full_range, y=full_pred_actual,
    mode="lines",
    name="Model output",
    line=dict(color="orange", width=1, dash="dot"),
    hovertemplate=f"{x_label} %{{x}}<br>Model: %{{y:.3f}}<extra></extra>"
))
# Shaded region to separate train / test
fig_full.add_vrect(
    x0=n_train, x1=len(full_range),
    fillcolor="lightyellow", opacity=0.3, line_width=0,
    annotation_text="Test region", annotation_position="top left"
)
fig_full.add_vline(
    x=n_train, line_dash="dash", line_color="gray",
    annotation_text=f"Train/Test split ({int(st.session_state.get('ann_X_train', X_train).shape[1])} samples)",
    annotation_position="top right"
)
fig_full.update_layout(
    xaxis_title=x_label,
    yaxis_title=y_label,
    template="plotly_white",
    height=380,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    hovermode="x unified"
)
st.plotly_chart(fig_full, use_container_width=True)

# ── Detailed metrics table ───────────────────────────────────────────────────
st.markdown("#### Detailed Error Metrics")
metrics_df = pd.DataFrame({
    "Metric": list(train_metrics.keys()),
    "Training Set": list(train_metrics.values()),
    "Test Set": list(test_metrics.values()),
})
st.dataframe(metrics_df, hide_index=True, use_container_width=True)

# ── Scatter: actual vs predicted (test only) ──────────────────────────────────
with st.expander("📈 Actual vs Predicted scatter (test set)"):
    fig_scatter = go.Figure()
    fig_scatter.add_trace(go.Scatter(
        x=y_test_actual, y=test_pred_actual,
        mode="markers",
        marker=dict(color="steelblue", size=4, opacity=0.6),
        hovertemplate="Actual: %{x:.3f}<br>Predicted: %{y:.3f}<extra></extra>"
    ))
    # Perfect prediction line
    mn = min(y_test_actual.min(), test_pred_actual.min())
    mx = max(y_test_actual.max(), test_pred_actual.max())
    fig_scatter.add_trace(go.Scatter(
        x=[mn, mx], y=[mn, mx],
        mode="lines",
        line=dict(color="crimson", dash="dash"),
        name="Perfect fit"
    ))
    fig_scatter.update_layout(
        xaxis_title=f"Actual {y_label}",
        yaxis_title=f"Predicted {y_label}",
        template="plotly_white",
        height=400
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# Stage 3: Year-ahead forecast
# ════════════════════════════════════════════════════════════════════════════
st.divider()
st.subheader("Stage 3: Year-Ahead Forecast")
st.write(
    "Provide the input features for the forecast year (all columns except the target). "
    "This should be a single-column or multi-column file with the same number of rows "
    "as the training data and **one fewer column** (no target column)."
)

fc1, fc2 = st.columns([2, 1])
with fc1:
    forecast_file = st.file_uploader(
        "Upload forecast input file (feature columns only, no target column)",
        type=["csv", "xlsx"],
        key="ann_forecast_upload"
    )
with fc2:
    forecast_label = st.text_input(
        "Forecast series label",
        value=f"Forecast ({raw_data.columns[-1] if 'ann_raw_data' in st.session_state else 'Year N+1'})"
    )

if forecast_file is not None:
    try:
        if forecast_file.name.endswith(".csv"):
            fc_data = pd.read_csv(forecast_file)
        else:
            fc_data = pd.read_excel(forecast_file)

        X_max = st.session_state["ann_X_max"]

        if fc_data.shape[1] != n_features:
            st.error(
                f"Forecast file has {fc_data.shape[1]} column(s) but model expects "
                f"{n_features} input feature(s). Check your file."
            )
            st.stop()

        X_fc = fc_data.values.astype(float)
        # Normalise using training max
        X_fc_n = X_fc / X_max
        fc_pred_n = ann_predict(X_fc_n.T, parameters)
        fc_pred   = fc_pred_n * Y_max

        st.success(
            f"✅ Forecast complete — {len(fc_pred)} time slots | "
            f"Peak: **{fc_pred.max():.2f}** | Mean: **{fc_pred.mean():.2f}** | "
            f"Total: **{fc_pred.sum():.1f}**"
        )

        # Chart
        fig_fc = go.Figure()
        # Overlay last year of actual data for context
        fig_fc.add_trace(go.Scatter(
            x=np.arange(len(Y_all)),
            y=Y_all,
            mode="lines",
            name="Historical (actual)",
            line=dict(color="steelblue", width=1),
            opacity=0.5
        ))
        fig_fc.add_trace(go.Scatter(
            x=np.arange(len(Y_all), len(Y_all) + len(fc_pred)),
            y=fc_pred,
            mode="lines",
            name=forecast_label,
            line=dict(color="darkorange", width=2),
            hovertemplate=f"{x_label} %{{x}}<br>{y_label}: %{{y:.3f}}<extra></extra>"
        ))
        fig_fc.add_vline(
            x=len(Y_all), line_dash="dash", line_color="gray",
            annotation_text="Forecast starts", annotation_position="top right"
        )
        fig_fc.update_layout(
            xaxis_title=x_label,
            yaxis_title=y_label,
            template="plotly_white",
            height=420,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode="x unified"
        )
        st.plotly_chart(fig_fc, use_container_width=True)

        # Monthly summary (if 8760 rows)
        if len(fc_pred) == 8760:
            import calendar
            st.markdown("#### Monthly Forecast Summary")
            monthly_rows = []
            h = 0
            for m in range(1, 13):
                days = calendar.monthrange(2025, m)[1]
                hrs  = days * 24
                sl   = fc_pred[h:h + hrs]
                monthly_rows.append({
                    "Month":       calendar.month_name[m],
                    f"Total ({y_label})":  round(sl.sum(), 1),
                    f"Peak ({y_label})":   round(sl.max(), 2),
                    f"Mean ({y_label})":   round(sl.mean(), 2),
                    f"Min ({y_label})":    round(sl.min(), 2),
                })
                h += hrs
            st.dataframe(pd.DataFrame(monthly_rows), hide_index=True, use_container_width=True)

        # Downloads
        dl1, dl2 = st.columns(2)
        with dl1:
            fc_df = pd.DataFrame({x_label: np.arange(len(fc_pred)), y_label: fc_pred})
            buf_csv = io.BytesIO()
            fc_df.to_csv(buf_csv, index=False)
            st.download_button(
                "⬇️ Download forecast CSV",
                data=buf_csv.getvalue(),
                file_name="ann_forecast.csv",
                mime="text/csv",
                use_container_width=True
            )
        with dl2:
            buf_xl = io.BytesIO()
            with pd.ExcelWriter(buf_xl, engine="openpyxl") as writer:
                fc_df.to_excel(writer, index=False, sheet_name="Forecast")
                pd.DataFrame(monthly_rows).to_excel(writer, index=False, sheet_name="Monthly Summary") if len(fc_pred) == 8760 else None
            st.download_button(
                "⬇️ Download forecast Excel",
                data=buf_xl.getvalue(),
                file_name="ann_forecast.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

        # Push to DF model demand session state
        if st.button("📤 Use this forecast as Demand input in the DF Model", use_container_width=True):
            demand_df = pd.DataFrame({"demand": fc_pred})
            st.session_state["demand"] = demand_df
            st.success(
                "✅ Forecast loaded as demand input. Go to the main page, "
                "re-upload other files and click Process & Initialize Data — "
                "demand is already set."
            )

    except Exception as e:
        st.error(f"Forecast failed: {e}")