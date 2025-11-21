# modules/ui_analysis.py
import streamlit as st
import plotly.graph_objects as go
import database
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAXResults

# ✅ Reutilizamos lógica de EDA / tendencias desde ui_info
from .ui_info import load_info_df, compute_eda_aggregates, line_chart, _axis, THEME_PALETTE


# --- Función para Cargar y Usar el Modelo de Forecast ---
@st.cache_resource
def load_forecast_model():
    """Carga el modelo SARIMA guardado."""
    try:
        model = SARIMAXResults.load("crime_forecaster.pkl")
        return model
    except FileNotFoundError:
        st.error(
            "Error: Archivo 'crime_forecaster.pkl' no encontrado. "
            "Por favor, ejecuta 'train_models.py' primero."
        )
        return None
    except Exception as e:
        st.error(f"Error al cargar el modelo de forecast: {e}")
        return None


@st.cache_data
def get_forecast(_model, steps=7):
    """Genera una predicción de 'steps' días."""
    if _model:
        try:
            forecast = _model.get_forecast(steps=steps)
            forecast_df = forecast.summary_frame(alpha=0.05)
            forecast_df = forecast_df.rename(
                columns={
                    "mean": "Predicción Promedio",
                    "mean_ci_lower": "Límite Inferior (95%)",
                    "mean_ci_upper": "Límite Superior (95%)",
                }
            )
            return forecast_df
        except Exception as e:
            st.warning(f"Error al generar predicción: {e}")
            return pd.DataFrame()
    return pd.DataFrame()


# --- Render principal para la página de Analysis ---
def render():
    # Título y descripción
    st.title("🕓 Análisis de Tendencia y Pronóstico")
    st.markdown("Análisis histórico y pronóstico a corto y mediano plazo.")

    # --- Carga de Datos y Modelos ---
    df_tendencia = database.get_historical_tendency()
    model_sarima = load_forecast_model()

    # --- Gráfico Principal: Tendencia y Pronóstico a 5 Meses ---
    st.header("Pronóstico de Tendencia a 5 Meses")
    st.markdown(
        "Esta gráfica muestra la tendencia histórica (últimos 6 meses) "
        "y una predicción del número total de crímenes para los próximos 5 meses."
    )

    if not df_tendencia.empty:
        df_tendencia["fecha"] = pd.to_datetime(df_tendencia["fecha"])
        df_tendencia = df_tendencia.sort_values("fecha")

        monthly_actual = (
            df_tendencia.set_index("fecha")["total_delitos"]
            .resample("M")
            .sum()
            .tail(6)
        )
        recent_daily = (
            df_tendencia.set_index("fecha")["total_delitos"].tail(180)
        )
        recent_daily_df = recent_daily.reset_index()
        recent_daily_df["media_7d"] = (
            recent_daily_df["total_delitos"].rolling(7).mean()
        )
        last_ma_value = (
            recent_daily_df["media_7d"].dropna().iloc[-1]
            if not recent_daily_df["media_7d"].dropna().empty
            else None
        )
        monthly_forecast = pd.DataFrame()

        color_hist_line = THEME_PALETTE[1]
        color_hist_bar = THEME_PALETTE[0]
        color_forecast = "#CBD5F5"

        fig_daily = go.Figure()
        fig_monthly = go.Figure()

        # Barras de casos diarios para mostrar variabilidad
        if not recent_daily_df.empty:
            fig_daily.add_trace(
                go.Bar(
                    x=recent_daily_df["fecha"],
                    y=recent_daily_df["total_delitos"],
                    name="Casos diarios observados",
                    marker_color=color_hist_bar,
                    opacity=0.28,
                    hovertemplate="%{x|%d %b %Y}<br>Total: %{y:,.0f}<extra></extra>",
                )
            )

            # Línea de media móvil 7 días para resaltar tendencia
            fig_daily.add_trace(
                go.Scatter(
                    x=recent_daily_df["fecha"],
                    y=recent_daily_df["media_7d"],
                    mode="lines",
                    name="Media móvil 7 días",
                    line=dict(color=color_hist_line, width=3),
                    hovertemplate="%{x|%d %b %Y}<br>Media 7d: %{y:,.0f}<extra></extra>",
                )
            )

        # Barras para totales mensuales recientes
        if not monthly_actual.empty:
            fig_monthly.add_trace(
                go.Bar(
                    x=monthly_actual.index,
                    y=monthly_actual.values,
                    name="Total mensual (histórico)",
                    marker_color=color_hist_bar,
                    opacity=0.35,
                    hovertemplate="%{x|%b %Y}<br>Total: %{y:,.0f}<extra></extra>",
                )
            )

        forecast_success = False

        # 2. Obtener y mostrar el pronóstico
        if model_sarima:
            # Pronóstico a 150 días (5 meses)
            df_forecast = get_forecast(model_sarima, steps=150)

            if not df_forecast.empty:
                forecast_success = True
                # 🔒 Corrección para evitar números negativos en pronóstico y bandas
                for col in [
                    "Predicción Promedio",
                    "Límite Inferior (95%)",
                    "Límite Superior (95%)",
                ]:
                    df_forecast[col] = df_forecast[col].clip(lower=0)

                df_forecast.index = pd.to_datetime(df_forecast.index)
                first_forecast_value = df_forecast["Predicción Promedio"].iloc[0]
                monthly_forecast = df_forecast.resample("M").sum().iloc[:5]

                forecast_start = df_forecast.index.min()
                forecast_end = df_forecast.index.max()

                # Línea de pronóstico (media diaria)
                fig_daily.add_trace(
                    go.Scatter(
                        x=df_forecast.index,
                        y=df_forecast["Predicción Promedio"],
                        mode="lines",
                        name="Pronóstico diario",
                        line=dict(dash="dash", color=color_forecast, width=3),
                        hovertemplate="%{x|%d %b %Y}<br>Pronóstico: %{y:,.0f}<extra></extra>",
                    )
                )
                if not monthly_forecast.empty:
                    fig_monthly.add_trace(
                        go.Bar(
                            x=monthly_forecast.index,
                            y=monthly_forecast["Predicción Promedio"],
                            name="Total mensual (pronosticado)",
                            marker_color=color_forecast,
                            opacity=0.45,
                            hovertemplate="%{x|%b %Y}<br>Total: %{y:,.0f}<extra></extra>",
                            error_y=dict(
                                type="data",
                                array=(
                                    (
                                        monthly_forecast["Límite Superior (95%)"]
                                        - monthly_forecast["Predicción Promedio"]
                                    ).clip(lower=0)
                                ),
                                arrayminus=(
                                    (
                                        monthly_forecast["Predicción Promedio"]
                                        - monthly_forecast["Límite Inferior (95%)"]
                                    ).clip(lower=0)
                                ),
                                color=color_forecast,
                                thickness=1.5,
                            ),
                        )
                    )

                if (
                    last_ma_value is not None
                    and first_forecast_value is not None
                    and last_ma_value > 0
                ):
                    delta_pct = (
                        (first_forecast_value - last_ma_value) / last_ma_value
                    ) * 100
                    annotation_y = max(
                        first_forecast_value, last_ma_value
                    )
                    fig_daily.add_annotation(
                        x=forecast_start,
                        y=annotation_y,
                        yshift=20,
                        text=f"{delta_pct:+.1f}% vs último promedio semanal",
                        showarrow=False,
                        font=dict(color=color_forecast, size=12),
                        bgcolor="rgba(15,23,42,0.6)",
                        bordercolor=color_forecast,
                        borderwidth=1,
                    )
            else:
                st.error("No se pudo generar el pronóstico.")
        else:
            st.warning("No se pudo cargar el modelo de pronóstico.")

        fig_daily.update_layout(
            template="plotly_dark",
            xaxis_title="Fecha",
            yaxis_title="Total de delitos (diario)",
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
            margin=dict(t=60, b=30, l=40, r=20),
            hovermode="x unified",
        )
        fig_daily.update_xaxes(showgrid=False)

        fig_monthly.update_layout(
            template="plotly_dark",
            xaxis_title="Mes",
            yaxis_title="Total de delitos (mensual)",
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
            margin=dict(t=40, b=30, l=40, r=20),
            hovermode="x unified",
        )
        fig_monthly.update_xaxes(showgrid=False)

        st.subheader("Detalle diario con pronóstico de 5 meses")
        st.plotly_chart(fig_daily, use_container_width=True)

        if fig_monthly.data:
            st.subheader("Comparativo mensual histórico vs pronosticado")
            st.plotly_chart(fig_monthly, use_container_width=True)

        if forecast_success and not monthly_forecast.empty:
            forecast_summary = monthly_forecast["Predicción Promedio"]
            last_hist_value = (
                monthly_actual.iloc[-1] if not monthly_actual.empty else None
            )
            peak_month = forecast_summary.idxmax()
            cols = st.columns(3)
            prox_value = forecast_summary.iloc[0]
            delta_text = None
            if last_hist_value is not None and last_hist_value > 0:
                delta = prox_value - last_hist_value
                delta_pct = (delta / last_hist_value) * 100
                delta_text = f"{delta:+,.0f} casos ({delta_pct:+.1f}%)"
            cols[0].metric(
                f"Próximo mes ({forecast_summary.index[0].strftime('%b %Y')})",
                f"{prox_value:,.0f}",
                delta=delta_text,
            )
            cols[1].metric(
                "Promedio próximos 5 meses",
                f"{forecast_summary.mean():,.0f}",
            )
            cols[2].metric(
                "Mes con mayor riesgo",
                f"{forecast_summary.max():,.0f}",
                delta=peak_month.strftime("%b %Y"),
            )

            with st.expander("Ver tabla con el pronóstico mensual a 5 meses"):
                forecast_table = monthly_forecast.copy()
                forecast_table.index = forecast_table.index.strftime("%b %Y")
                st.dataframe(
                    forecast_table[
                        [
                            "Predicción Promedio",
                            "Límite Inferior (95%)",
                            "Límite Superior (95%)",
                        ]
                    ].style.format("{:.0f}")
                )
    else:
        st.warning("No se pudieron cargar los datos de tendencia.")

    # =========================
    # TENDENCIAS EN EL TIEMPO
    # (antes estaban en ui_info)
    # =========================
    st.header("Tendencias en el Tiempo")

    # Cargamos el dataset completo para tendencias globales
    base_df = load_info_df(year_min=2015, limit=None)
    if base_df.empty:
        st.warning("No hay datos suficientes para mostrar tendencias históricas.")
        return

    agg = compute_eda_aggregates(base_df)

    # Tendencia Anual por Alcaldía (Top 8)
    if not agg["yearly_mun_top"].empty:
        st.subheader("Tendencia Anual por Alcaldía (8 principales)")
        st.altair_chart(
            line_chart(
                agg["yearly_mun_top"],
                "anio_inicio",
                "count",
                color_field="alcaldia_std",
                title="Tendencia Anual por Alcaldía (8 principales)",
                width=950,
                height=420,
                x_axis=_axis("Año"),
                y_axis=_axis("Número de casos"),
            ),
            use_container_width=True,
        )

    # Tendencia Anual: Violento vs No Violento
    if not agg["yearly_viol"].empty:
        st.subheader("Tendencia Anual: Violento vs No Violento")
        st.altair_chart(
            line_chart(
                agg["yearly_viol"],
                "anio_inicio",
                "count",
                color_field="violence_type",
                title="Tendencia Anual: Violento vs No Violento",
                width=950,
                height=420,
                x_axis=_axis("Año"),
                y_axis=_axis("Número de casos"),
            ),
            use_container_width=True,
        )
