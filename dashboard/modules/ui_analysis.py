import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import database # Importamos nuestro módulo
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAXResults # Para cargar el modelo

st.set_page_config(page_title="Análisis de Tendencias", page_icon="🕓", layout="wide")

# --- Función para Cargar y Usar el Modelo de Forecast ---
@st.cache_resource
def load_forecast_model():
    """Carga el modelo SARIMA guardado."""
    try:
        model = SARIMAXResults.load('crime_forecaster.pkl')
        return model
    except FileNotFoundError:
        st.error("Error: Archivo 'crime_forecaster.pkl' no encontrado. Por favor, ejecuta 'train_models.py' primero.")
        return None
    except Exception as e:
        st.error(f"Error al cargar el modelo de forecast: {e}")
        return None

def get_forecast(model, steps=7):
    """Genera una predicción de 'steps' días."""
    if model:
        try:
            forecast = model.get_forecast(steps=steps)
            forecast_df = forecast.summary_frame(alpha=0.05) # Intervalo de confianza del 95%
            return forecast_df
        except Exception as e:
            st.warning(f"Error al generar predicción: {e}")
            return pd.DataFrame()
    return pd.DataFrame()


# --- Título de la Página ---
st.title("🕓 Análisis de Tendencia y Pronóstico")
st.markdown("Análisis histórico y pronóstico a corto plazo (short-term future: Forecast).")

# --- Carga de Datos y Modelos ---
df_tendencia = database.get_historical_tendency()
model_sarima = load_forecast_model()

# --- Gráfico Principal: Tendencia y Pronóstico ---
st.subheader("Historical trend and forecast")

if not df_tendencia.empty:
    fig = go.Figure()
    
    # 1. Línea histórica (solo los últimos 180 días para que se vea mejor)
    df_tendencia_reciente = df_tendencia.tail(180)
    fig.add_trace(go.Scatter(
        x=df_tendencia_reciente['fecha'], 
        y=df_tendencia_reciente['total_delitos'], 
        mode='lines', 
        name='Tendencia Histórica'
    ))
    
    # 2. Obtener y mostrar el pronóstico
    if model_sarima:
        df_forecast = get_forecast(model_sarima, steps=7) # Pronóstico a 7 días
        
        if not df_forecast.empty:
            # Línea de pronóstico (media)
            fig.add_trace(go.Scatter(
                x=df_forecast.index, 
                y=df_forecast['mean'], 
                mode='lines+markers', 
                name='Pronóstico', 
                line=dict(dash='dot', color='red')
            ))
            # Banda de confianza (como en tu mockup)
            fig.add_trace(go.Scatter(
                x=df_forecast.index,
                y=df_forecast['mean_ci_upper'],
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                name='Límite Superior'
            ))
            fig.add_trace(go.Scatter(
                x=df_forecast.index,
                y=df_forecast['mean_ci_lower'],
                mode='lines',
                line=dict(width=0),
                fillcolor='rgba(255, 0, 0, 0.2)', # Sombreado rojo
                fill='tonexty', # Rellenar hasta la traza anterior
                showlegend=False,
                name='Límite Inferior'
            ))

    fig.update_layout(template="plotly_dark", xaxis_title="Fecha", yaxis_title="Total de Delitos")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("No se pudieron cargar los datos de tendencia.")


# --- EL RESTO DE GRÁFICOS (permanecen igual) ---
st.divider()
st.header("Distribución de Delitos")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Distribución por Tipo de Delito")
    df_categoria = database.get_distribution_by_category()
    if not df_categoria.empty:
        fig_pie = px.pie(
            df_categoria.head(10), # Top 10
            names='categoria_delito', 
            values='total', 
            title="Distribución de Delitos (Top 10)"
        )
        fig_pie.update_layout(template="plotly_dark")
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.warning("No se pudieron cargar datos de categorías.")
    
    st.subheader("Distribución por Hora del Día")
    df_hora = database.get_distribution_by_hour()
    if not df_hora.empty:
        fig_bar = px.bar(
            df_hora, 
            x='hora_hecho', 
            y='total', 
            title="Distribución de Crímenes por Hora"
        )
        fig_bar.update_layout(template="plotly_dark", xaxis_title="Hora del Día", yaxis_title="Total de Delitos")
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.warning("No se pudieron cargar datos por hora.")
    
with col2:
    st.subheader("Heatmap de Crímenes Violentos vs. No Violentos por Hora")
    df_heatmap = database.get_violence_heatmap_data()
    if not df_heatmap.empty:
        fig_heat = px.density_heatmap(
            df_heatmap, 
            x="hora_hecho", 
            y="violence_type", 
            z="total", 
            histfunc="sum",
            title="Intensidad de Violencia por Hora"
        )
        fig_heat.update_layout(template="plotly_dark", xaxis_title="Hora del Día", yaxis_title="Tipo de Violencia")
        st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.warning("No se pudieron cargar datos del heatmap de violencia.")

    st.subheader("Métricas de Violencia (Crímenes Violentos)")
    df_metrics = database.get_violence_time_metrics()
    if not df_metrics.empty:
        metric_cols = st.columns(2)
        # Asegurarnos de que hay al menos 2 filas para evitar error de índice
        if len(df_metrics) >= 2:
            for index, row in df_metrics.iterrows():
                metric_cols[index % 2].metric(
                    label=f"{row['franja_horaria']}", 
                    value=f"{row['porcentaje']:.1f}%"
                )
        elif len(df_metrics) == 1:
             metric_cols[0].metric(
                    label=f"{df_metrics.iloc[0]['franja_horaria']}", 
                    value=f"{df_metrics.iloc[0]['porcentaje']:.1f}%"
                )
    else:
        st.warning("No se pudieron cargar métricas de violencia.")