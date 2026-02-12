"""Interactive web-based heating optimizer simulator.

Streamlit app wrapping the existing simulator logic with interactive controls
and Plotly charts. Includes comparison mode to overlay two scenarios.

Usage:
    streamlit run scripts/heating/simulator_app.py
"""

import sys
from datetime import datetime, time
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

# Ensure project root is importable
_project_root = str(Path(__file__).resolve().parents[2])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from scripts.heating.optimizer import DailyHeatingSchedule, HeatingOptimizer  # noqa: E402
from scripts.heating.simulator import (  # noqa: E402
    diurnal_temperature,
    estimate_current_room_temp,
    generate_forecast,
)
from scripts.heating.thermal_model import ThermalModel  # noqa: E402

MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "heating"
MODEL_FILE = "thermal_model.pkl"

WEATHER_CONDITIONS = [
    "cloudy",
    "sunny",
    "clear",
    "partlycloudy",
    "overcast",
    "rainy",
    "snowy",
]


def run_simulation(
    temp_min: float,
    temp_max: float,
    condition: str,
    target_warm_time: time,
    target_off_time: time,
    target_temp: float,
    min_bedroom_temp: float,
    min_daytime_temp: float,
    k_override: float | None = None,
    heating_rate_override: float | None = None,
) -> tuple[DailyHeatingSchedule, list[dict], float, float, ThermalModel]:
    """Run optimizer with given params."""
    model = ThermalModel(model_dir=str(MODEL_PATH))
    model.load(filename=MODEL_FILE)

    if k_override is not None:
        model.k = k_override
    if heating_rate_override is not None:
        model.mean_heating_rate = heating_rate_override

    clock_time = time(4, 0)
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    simulated_now = today.replace(hour=clock_time.hour, minute=clock_time.minute)

    forecast = generate_forecast(today, temp_min, temp_max, condition)

    outside_temp = diurnal_temperature(
        clock_time.hour + clock_time.minute / 60, temp_min, temp_max
    )

    optimizer = HeatingOptimizer(model)

    # Two-pass estimation (same logic as simulator.py)
    bedroom_temp = estimate_current_room_temp(
        model, clock_time, temp_min, target_temp, target_warm_time, target_off_time
    )
    pass1 = optimizer.calculate_optimal_schedule(
        target_warm_time=target_warm_time,
        target_night_time=target_off_time,
        target_temp=target_temp,
        min_overnight_temp=min_bedroom_temp,
        min_daytime_temp=min_daytime_temp,
        current_temps={"bedroom": bedroom_temp},
        outside_temp=outside_temp,
        weather_forecast=forecast,
        current_time=simulated_now,
    )
    computed_off = pass1.switch_off_time or target_off_time
    if computed_off != target_off_time:
        bedroom_temp = estimate_current_room_temp(
            model, clock_time, temp_min, target_temp, target_warm_time, computed_off
        )

    schedule = optimizer.calculate_optimal_schedule(
        target_warm_time=target_warm_time,
        target_night_time=target_off_time,
        target_temp=target_temp,
        min_overnight_temp=min_bedroom_temp,
        min_daytime_temp=min_daytime_temp,
        current_temps={"bedroom": bedroom_temp},
        outside_temp=outside_temp,
        weather_forecast=forecast,
        current_time=simulated_now,
    )

    return schedule, forecast, bedroom_temp, outside_temp, model


def build_temperature_chart(
    schedule: DailyHeatingSchedule,
    forecast: list[dict],
    temp_min: float,
    temp_max: float,
    target_temp: float,
    min_bedroom_temp: float,
    min_daytime_temp: float,
    target_warm_time: time,
    target_off_time: time,
    schedule_b: DailyHeatingSchedule | None = None,
    forecast_b: list[dict] | None = None,
    temp_min_b: float | None = None,
    temp_max_b: float | None = None,
) -> go.Figure:
    """Build Plotly temperature chart with heating bands."""
    fig = go.Figure()

    hours = [hp.hour for hp in schedule.hours]
    room_temps = [hp.expected_room_temp for hp in schedule.hours]
    outside_temps = [diurnal_temperature(float(h), temp_min, temp_max) for h in hours]

    # Heating ON band (Scenario A)
    on_hour = schedule.switch_on_time.hour
    off_hour = schedule.switch_off_time.hour if schedule.switch_off_time else 24
    fig.add_vrect(
        x0=on_hour,
        x1=off_hour,
        fillcolor="rgba(251, 146, 60, 0.15)",
        layer="below",
        line_width=0,
        annotation_text="Heating ON",
        annotation_position="top left",
        annotation_font_color="#F59E0B",
        annotation_font_size=11,
    )

    # Indoor temp (A)
    fig.add_trace(
        go.Scatter(
            x=hours,
            y=room_temps,
            name="Indoor (A)" if schedule_b else "Indoor",
            line=dict(color="#F59E0B", width=3),
            mode="lines",
        )
    )

    # Outdoor temp (A)
    fig.add_trace(
        go.Scatter(
            x=hours,
            y=outside_temps,
            name="Outdoor (A)" if schedule_b else "Outdoor",
            line=dict(color="#3B82F6", width=2, dash="dash"),
            mode="lines",
        )
    )

    # Target temp threshold
    fig.add_hline(
        y=target_temp,
        line=dict(color="#10B981", width=1, dash="dot"),
        annotation_text=f"Target {target_temp}\u00b0C",
        annotation_position="bottom right",
        annotation_font_color="#10B981",
        annotation_font_size=10,
    )

    # Min overnight threshold (nighttime hours only)
    for h in hours:
        if h < target_warm_time.hour or h >= target_off_time.hour:
            fig.add_shape(
                type="line",
                x0=h,
                x1=h + 1,
                y0=min_bedroom_temp,
                y1=min_bedroom_temp,
                line=dict(color="#EF4444", width=1, dash="dot"),
            )

    # Min daytime threshold (daytime hours only)
    for h in hours:
        if target_warm_time.hour <= h < target_off_time.hour:
            fig.add_shape(
                type="line",
                x0=h,
                x1=h + 1,
                y0=min_daytime_temp,
                y1=min_daytime_temp,
                line=dict(color="#F87171", width=1, dash="dot"),
            )

    # Switch markers
    fig.add_vline(
        x=on_hour,
        line=dict(color="#F59E0B", width=1, dash="dash"),
        annotation_text=f"ON {schedule.switch_on_time.strftime('%H:%M')}",
        annotation_position="top left",
        annotation_font_size=10,
        annotation_font_color="#F59E0B",
    )
    if schedule.switch_off_time:
        fig.add_vline(
            x=off_hour,
            line=dict(color="#F59E0B", width=1, dash="dash"),
            annotation_text=f"OFF {schedule.switch_off_time.strftime('%H:%M')}",
            annotation_position="top right",
            annotation_font_size=10,
            annotation_font_color="#F59E0B",
        )

    # Scenario B overlay
    if schedule_b is not None:
        hours_b = [hp.hour for hp in schedule_b.hours]
        room_temps_b = [hp.expected_room_temp for hp in schedule_b.hours]
        t_min_b = temp_min_b if temp_min_b is not None else temp_min
        t_max_b = temp_max_b if temp_max_b is not None else temp_max
        outside_temps_b = [
            diurnal_temperature(float(h), t_min_b, t_max_b) for h in hours_b
        ]

        on_hour_b = schedule_b.switch_on_time.hour
        off_hour_b = (
            schedule_b.switch_off_time.hour if schedule_b.switch_off_time else 24
        )
        fig.add_vrect(
            x0=on_hour_b,
            x1=off_hour_b,
            fillcolor="rgba(139, 92, 246, 0.10)",
            layer="below",
            line_width=0,
        )

        fig.add_trace(
            go.Scatter(
                x=hours_b,
                y=room_temps_b,
                name="Indoor (B)",
                line=dict(color="#8B5CF6", width=3, dash="dash"),
                mode="lines",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=hours_b,
                y=outside_temps_b,
                name="Outdoor (B)",
                line=dict(color="#6366F1", width=2, dash="dot"),
                mode="lines",
            )
        )

    fig.update_layout(
        xaxis=dict(
            title="Hour of Day",
            tickmode="array",
            tickvals=list(range(0, 24, 2)),
            ticktext=[f"{h:02d}:00" for h in range(0, 24, 2)],
            range=[-0.5, 23.5],
        ),
        yaxis=dict(title="Temperature (\u00b0C)"),
        height=450,
        margin=dict(l=50, r=20, t=30, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )

    return fig


def render_metrics(
    schedule: DailyHeatingSchedule, model: ThermalModel, label: str = ""
) -> None:
    """Render key metrics using st.metric."""
    prefix = f"**{label}** " if label else ""
    if prefix:
        st.markdown(prefix)

    off_str = (
        schedule.switch_off_time.strftime("%H:%M")
        if schedule.switch_off_time
        else "CONTINUOUS"
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Switch ON", schedule.switch_on_time.strftime("%H:%M"))
    c2.metric("Switch OFF", off_str)
    c3.metric("Setpoint", f"{schedule.optimal_setpoint}\u00b0C")

    c4, c5, c6 = st.columns(3)
    c4.metric("Gas Usage", f"~{schedule.expected_gas_usage:.1f} kWh")
    burner_str = (
        f"{schedule.expected_burner_hours:.1f} hrs"
        if schedule.expected_burner_hours is not None
        else "-"
    )
    c5.metric("Burner Hours", burner_str)
    c6.metric("Solar Gain", f"+{schedule.solar_contribution:.1f}\u00b0C")


def render_hourly_table(
    schedule: DailyHeatingSchedule, temp_min: float, temp_max: float
) -> None:
    """Render hourly plan as a dataframe."""
    rows = []
    for hp in schedule.hours:
        outside_h = diurnal_temperature(hp.hour, temp_min, temp_max)
        rows.append(
            {
                "Hour": f"{hp.hour:02d}:00",
                "State": hp.system_state.upper(),
                "Setpoint": f"{hp.setpoint}\u00b0C" if hp.setpoint else "-",
                "Room Temp": f"{hp.expected_room_temp:.1f}\u00b0C",
                "Modulation": (
                    f"{hp.expected_modulation:.0f}%"
                    if hp.system_state == "on"
                    else "-"
                ),
                "Outside": f"{outside_h:.1f}\u00b0C",
            }
        )
    st.dataframe(rows, use_container_width=True, hide_index=True)


def sidebar_controls(key_prefix: str = "a") -> dict:
    """Render sidebar controls and return params dict."""
    params = {}

    st.sidebar.subheader("Weather" if key_prefix == "a" else "Weather (B)")
    params["temp_min"] = st.sidebar.slider(
        "Temp Min (\u00b0C)",
        -15.0,
        15.0,
        -5.0 if key_prefix == "a" else 0.0,
        0.5,
        key=f"{key_prefix}_temp_min",
    )
    params["temp_max"] = st.sidebar.slider(
        "Temp Max (\u00b0C)",
        -10.0,
        25.0,
        3.0 if key_prefix == "a" else 8.0,
        0.5,
        key=f"{key_prefix}_temp_max",
    )
    params["condition"] = st.sidebar.selectbox(
        "Condition",
        WEATHER_CONDITIONS,
        index=0 if key_prefix == "a" else 1,
        key=f"{key_prefix}_condition",
    )

    if key_prefix == "a":
        st.sidebar.subheader("Schedule")
        warm_time = st.sidebar.time_input(
            "Target Warm Time", time(8, 0), key=f"{key_prefix}_warm"
        )
        off_time = st.sidebar.time_input(
            "Preferred Off Time", time(23, 0), key=f"{key_prefix}_off"
        )
        params["target_warm_time"] = warm_time
        params["target_off_time"] = off_time
        params["target_temp"] = st.sidebar.slider(
            "Target Temp (\u00b0C)", 18.0, 23.0, 20.0, 0.5, key=f"{key_prefix}_target"
        )
        params["min_bedroom_temp"] = st.sidebar.slider(
            "Min Night Temp (\u00b0C)",
            15.0,
            20.0,
            18.0,
            0.5,
            key=f"{key_prefix}_min_night",
        )
        params["min_daytime_temp"] = st.sidebar.slider(
            "Min Day Temp (\u00b0C)",
            18.0,
            22.0,
            20.0,
            0.5,
            key=f"{key_prefix}_min_day",
        )

    st.sidebar.subheader(
        "Model Overrides" if key_prefix == "a" else "Model Overrides (B)"
    )
    use_k = st.sidebar.checkbox(
        "Override k", value=False, key=f"{key_prefix}_use_k"
    )
    if use_k:
        params["k_override"] = st.sidebar.slider(
            "k (cooling constant)",
            0.003,
            0.010,
            0.0064,
            0.0001,
            format="%.4f",
            key=f"{key_prefix}_k",
        )
    else:
        params["k_override"] = None

    use_hr = st.sidebar.checkbox(
        "Override heating rate",
        value=False,
        key=f"{key_prefix}_use_hr",
    )
    if use_hr:
        params["heating_rate_override"] = st.sidebar.slider(
            "Heating rate (\u00b0C/hr)",
            0.5,
            2.0,
            1.0,
            0.05,
            key=f"{key_prefix}_hr",
        )
    else:
        params["heating_rate_override"] = None

    return params


def main() -> None:
    st.set_page_config(
        page_title="Heating Optimizer Simulator",
        page_icon="\U0001f321\ufe0f",
        layout="wide",
    )

    st.title("\U0001f321\ufe0f Heating Optimizer Simulator")

    # --- Sidebar ---
    params_a = sidebar_controls("a")

    st.sidebar.markdown("---")
    compare = st.sidebar.checkbox("Compare mode (Scenario B)", value=False)

    params_b = None
    if compare:
        st.sidebar.markdown("---")
        params_b = sidebar_controls("b")

    # --- Run simulation A ---
    schedule_a, forecast_a, bedroom_a, outside_a, model_a = run_simulation(
        temp_min=params_a["temp_min"],
        temp_max=params_a["temp_max"],
        condition=params_a["condition"],
        target_warm_time=params_a.get("target_warm_time", time(8, 0)),
        target_off_time=params_a.get("target_off_time", time(23, 0)),
        target_temp=params_a.get("target_temp", 20.0),
        min_bedroom_temp=params_a.get("min_bedroom_temp", 18.0),
        min_daytime_temp=params_a.get("min_daytime_temp", 20.0),
        k_override=params_a.get("k_override"),
        heating_rate_override=params_a.get("heating_rate_override"),
    )

    # --- Run simulation B ---
    schedule_b = None
    forecast_b = None
    if params_b is not None:
        schedule_b, forecast_b, _, _, model_b = run_simulation(
            temp_min=params_b["temp_min"],
            temp_max=params_b["temp_max"],
            condition=params_b["condition"],
            target_warm_time=params_a.get("target_warm_time", time(8, 0)),
            target_off_time=params_a.get("target_off_time", time(23, 0)),
            target_temp=params_a.get("target_temp", 20.0),
            min_bedroom_temp=params_a.get("min_bedroom_temp", 18.0),
            min_daytime_temp=params_a.get("min_daytime_temp", 20.0),
            k_override=params_b.get("k_override"),
            heating_rate_override=params_b.get("heating_rate_override"),
        )

    # --- Metrics ---
    if compare and schedule_b:
        col_a, col_b = st.columns(2)
        with col_a:
            render_metrics(schedule_a, model_a, label="Scenario A")
        with col_b:
            render_metrics(schedule_b, model_b, label="Scenario B")
    else:
        render_metrics(schedule_a, model_a)

    # --- Model info ---
    info = model_a.get_model_info()
    tau = f"{1 / model_a.k:.0f}h" if model_a.k > 0 else "-"
    st.caption(
        f"Model: k={model_a.k:.4f} (\u03c4={tau}) | "
        f"heating rate={model_a.mean_heating_rate:.2f}\u00b0C/h | "
        f"trained={info['last_trained'] or 'defaults'}"
    )

    # --- Chart ---
    fig = build_temperature_chart(
        schedule=schedule_a,
        forecast=forecast_a,
        temp_min=params_a["temp_min"],
        temp_max=params_a["temp_max"],
        target_temp=params_a.get("target_temp", 20.0),
        min_bedroom_temp=params_a.get("min_bedroom_temp", 18.0),
        min_daytime_temp=params_a.get("min_daytime_temp", 20.0),
        target_warm_time=params_a.get("target_warm_time", time(8, 0)),
        target_off_time=params_a.get("target_off_time", time(23, 0)),
        schedule_b=schedule_b,
        forecast_b=forecast_b,
        temp_min_b=params_b["temp_min"] if params_b else None,
        temp_max_b=params_b["temp_max"] if params_b else None,
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Reasoning ---
    with st.expander("Reasoning", expanded=False):
        if compare and schedule_b:
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Scenario A**")
                for r in schedule_a.reasoning:
                    st.markdown(f"- {r}")
            with col_b:
                st.markdown("**Scenario B**")
                for r in schedule_b.reasoning:
                    st.markdown(f"- {r}")
        else:
            for r in schedule_a.reasoning:
                st.markdown(f"- {r}")

    # --- Hourly table ---
    with st.expander("Hourly Plan", expanded=False):
        if compare and schedule_b:
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Scenario A**")
                render_hourly_table(
                    schedule_a, params_a["temp_min"], params_a["temp_max"]
                )
            with col_b:
                st.markdown("**Scenario B**")
                render_hourly_table(
                    schedule_b, params_b["temp_min"], params_b["temp_max"]
                )
        else:
            render_hourly_table(schedule_a, params_a["temp_min"], params_a["temp_max"])


if __name__ == "__main__":
    main()
