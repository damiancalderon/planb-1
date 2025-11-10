# modules/ui_info.py
import streamlit as st
import pandas as pd
import altair as alt

# ✅ usamos helpers propios
from .helpers import MONTH_NAMES
# ✅ usamos el runner DuckDB centralizado
from .data import run_query, get_violence_time_metrics

# =================== THEME ===================
def _enable_transparent_theme():
    theme_conf = {
        "config": {
            "background": "transparent",
            "view": {"fill": "transparent"},
            "axis": {
                "labelFontSize": 13,
                "titleFontSize": 14,
                "labelColor": "#E5E7EB",
                "titleColor": "#E5E7EB",
                "gridColor": "#33415555",
                "domainColor": "#64748B"
            },
            "title": {"fontSize": 16, "fontWeight": "bold", "color": "#E5E7EB"},
            "legend": {"labelFontSize": 13, "titleFontSize": 14,
                       "labelColor": "#E5E7EB", "titleColor": "#E5E7EB"}
        }
    }
    alt.themes.register("transp_dark", lambda: theme_conf)
    alt.themes.enable("transp_dark")
_enable_transparent_theme()

THEME_PALETTE = [
    "#1E3A8A", "#3B82F6", "#60A5FA", "#93C5FD", "#A3B3C2",
    "#64748B", "#0EA5E9", "#14B8A6", "#F59E0B", "#64748B",
]

# =================== HELPERS ===================
def _cfg(width=700, height=350):
    return {"width": width, "height": height}

def _axis(title, orient=None):
    base = {"title": title, "grid": True, "gridColor": "#33415555"}
    if orient: base["orient"] = orient
    return base

# =================== DATA LOAD (DuckDB) ===================
@st.cache_data(ttl=3600, show_spinner=False)
def load_info_df(year_min: int = 2016, limit: int | None = 100_000) -> pd.DataFrame:
    """
    Trae columnas necesarias desde DuckDB y las normaliza a los nombres que
    esperan las funciones de EDA (anio_inicio, mes_inicio, hour_hecho, *_std, etc.).
    """
    limit_clause = "LIMIT ?" if limit is not None else ""
    params = [year_min] + ([int(limit)] if limit is not None else [])

    # Nota: DuckDB soporta EXTRACT() y MONTH()/HOUR(). Cast explícitos por seguridad.
    query = f"""
        SELECT
            anio_hecho                                     AS anio_inicio,
            EXTRACT(MONTH FROM CAST(fecha_hecho AS DATE))  AS mes_inicio,
            EXTRACT(HOUR  FROM hora_hecho)                 AS hour_hecho,
            alcaldia_hecho                                 AS alcaldia_std,
            /* si no existe colonia_hecho en tu esquema, quedará NULL y se maneja abajo */
            TRY_CAST(colonia_hecho AS VARCHAR)             AS colonia_std,
            categoria_delito                               AS delito_std,
            crime_classification,
            violence_type
        FROM crimes
        WHERE anio_hecho >= ?
        {limit_clause}
    """
    df = run_query(query, params=params)
    if df.empty:
        return df

    # Normalizaciones suaves
    for col in ["anio_inicio", "mes_inicio", "hour_hecho"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Asegurar columnas esperadas (aunque vengan vacías)
    for needed in ["alcaldia_std", "colonia_std", "delito_std",
                   "crime_classification", "violence_type"]:
        if needed not in df.columns:
            df[needed] = pd.Series(dtype="object")

    return df

# =================== CACHES ===================
@st.cache_data(ttl=3600, show_spinner=False)
def get_violence_time_metrics_cached() -> pd.DataFrame:
    try:
        return get_violence_time_metrics()
    except Exception:
        return pd.DataFrame()

@st.cache_data(show_spinner=False, max_entries=2)
def compute_eda_aggregates(df: pd.DataFrame):
    YEAR_RANGE, MONTH_RANGE, HOUR_RANGE = range(2015, 2025), range(1, 13), range(0, 24)

    df = df.copy()
    df["anio_inicio"] = pd.to_numeric(df.get("anio_inicio"), errors="coerce")
    df["mes_inicio"]  = pd.to_numeric(df.get("mes_inicio"),  errors="coerce")
    if "hour_hecho" in df.columns:
        df["hour_hecho"] = pd.to_numeric(df["hour_hecho"], errors="coerce")

    year_counts = (
        df.dropna(subset=["anio_inicio"])
          .groupby("anio_inicio").size()
          .reindex(YEAR_RANGE, fill_value=0)
          .reset_index(name="count").rename(columns={"anio_inicio":"year"})
    )
    month_counts = (
        df.dropna(subset=["mes_inicio"])
          .groupby("mes_inicio").size()
          .reindex(MONTH_RANGE, fill_value=0)
          .reset_index(name="count").rename(columns={"mes_inicio":"month"})
    )
    month_counts["month_name"] = month_counts["month"].map(lambda m: MONTH_NAMES.get(int(m), str(m)))

    if "hour_hecho" in df.columns:
        hour_counts = (
            df.dropna(subset=["hour_hecho"]).groupby("hour_hecho").size()
              .reindex(HOUR_RANGE, fill_value=0).reset_index(name="count").rename(columns={"hour_hecho":"hour"})
        )
    else:
        hour_counts = pd.DataFrame({"hour": list(HOUR_RANGE), "count": 0})

    alc_counts = df.get("alcaldia_std", pd.Series([], dtype="object")).value_counts().sort_values(ascending=True).reset_index()
    if not alc_counts.empty:
        alc_counts.columns = ["alcaldia","count"]
    else:
        alc_counts = pd.DataFrame(columns=["alcaldia","count"])

    col_counts = df.get("colonia_std", pd.Series([], dtype="object")).value_counts().nlargest(10).sort_values(ascending=True).reset_index()
    if not col_counts.empty:
        col_counts.columns = ["colonia","count"]
    else:
        col_counts = pd.DataFrame(columns=["colonia","count"])

    del_counts = df.get("delito_std", pd.Series([], dtype="object")).value_counts().nlargest(10).sort_values(ascending=True).reset_index()
    if not del_counts.empty:
        del_counts.columns = ["delito","count"]
    else:
        del_counts = pd.DataFrame(columns=["delito","count"])

    cls_counts = df.get("crime_classification", pd.Series([], dtype="object")).value_counts().reset_index()
    if not cls_counts.empty:
        cls_counts.columns = ["classification","count"]
    else:
        cls_counts = pd.DataFrame(columns=["classification","count"])

    if "violence_type" in df.columns:
        v_counts = df["violence_type"].value_counts().reset_index()
        v_counts.columns = ["violence","count"]
    else:
        v_counts = pd.DataFrame(columns=["violence","count"])

    if {"alcaldia_std","crime_classification"}.issubset(df.columns):
        stacked = df.groupby(['alcaldia_std','crime_classification']).size().reset_index(name="count")
    else:
        stacked = pd.DataFrame(columns=["alcaldia_std","crime_classification","count"])

    if {"anio_inicio","alcaldia_std"}.issubset(df.columns):
        yearly_mun = (
            df.dropna(subset=['anio_inicio'])
              .groupby(['anio_inicio','alcaldia_std']).size().reset_index(name='count')
        )
        totals = yearly_mun.groupby("alcaldia_std")["count"].sum().sort_values(ascending=False)
        top8 = totals.head(8).index.tolist()
        yearly_mun_top = yearly_mun[yearly_mun["alcaldia_std"].isin(top8)].copy()
        yearly_mun_top["anio_inicio"] = pd.Categorical(
            yearly_mun_top["anio_inicio"], categories=range(2015, 2025), ordered=True
        )
    else:
        top8, yearly_mun_top = [], pd.DataFrame(columns=["anio_inicio","alcaldia_std","count"])

    if {"anio_inicio","crime_classification"}.issubset(df.columns):
        year_class = (
            df.dropna(subset=['anio_inicio'])
              .groupby(['anio_inicio','crime_classification']).size().reset_index(name='count')
        )
        year_class["anio_inicio"] = pd.Categorical(year_class["anio_inicio"], categories=range(2015, 2025), ordered=True)
    else:
        year_class = pd.DataFrame(columns=["anio_inicio","crime_classification","count"])

    if {"anio_inicio","violence_type"}.issubset(df.columns):
        yearly_viol = (
            df.dropna(subset=['anio_inicio'])
              .groupby(['anio_inicio','violence_type']).size().reset_index(name='count')
        )
        yearly_viol["anio_inicio"] = pd.Categorical(yearly_viol["anio_inicio"], categories=range(2015, 2025), ordered=True)
    else:
        yearly_viol = pd.DataFrame(columns=["anio_inicio","violence_type","count"])

    return {
        "year_counts": year_counts,
        "month_counts": month_counts,
        "hour_counts": hour_counts,
        "alc_counts": alc_counts,
        "col_counts": col_counts,
        "del_counts": del_counts,
        "cls_counts": cls_counts,
        "v_counts": v_counts,
        "stacked": stacked,
        "yearly_mun_top": yearly_mun_top,
        "year_class": year_class,
        "yearly_viol": yearly_viol,
        "top8": top8
    }

# =================== CHART PRIMS ===================
def bar_chart(df, x, y, *, x_type="O", y_type="Q", sort=None, title="", color=None, width=700, height=350, x_axis=None, y_axis=None):
    return alt.Chart(df, **_cfg(width, height)).mark_bar(color=color).encode(
        x=alt.X(f"{x}:{x_type}", sort=sort, axis=x_axis),
        y=alt.Y(f"{y}:{y_type}", axis=y_axis),
        tooltip=[f"{x}:{x_type}", f"{y}:{y_type}"]
    ).properties(title=title)

def barh_chart(df, y, x, *, y_type="N", x_type="Q", sort=None, title="", color=None, width=700, height=350, x_axis=None, y_axis=None):
    return alt.Chart(df, **_cfg(width, height)).mark_bar(color=color).encode(
        x=alt.X(f"{x}:{x_type}", axis=x_axis),
        y=alt.Y(f"{y}:{y_type}", sort=sort, axis=y_axis),
        tooltip=[f"{y}:{y_type}", f"{x}:{x_type}"]
    ).properties(title=title)

def line_chart(df, x, y, color_field=None, *, x_type="O", y_type="Q", title="", width=700, height=350, x_axis=None, y_axis=None, point=False):
    mk = alt.Chart(df, **_cfg(width, height)).mark_line(point=point, strokeWidth=2).encode(
        x=alt.X(f"{x}:{x_type}", axis=x_axis),
        y=alt.Y(f"{y}:{y_type}", axis=y_axis),
        tooltip=[f"{x}:{x_type}", f"{y}:{y_type}"]
    )
    if color_field:
        mk = mk.encode(color=alt.Color(f"{color_field}:N",
                     scale=alt.Scale(range=THEME_PALETTE[:max(1, df[color_field].nunique())])))
    return mk.properties(title=title)

def donut_chart(df, field, count_col, *, title="", colors=None, width=700, height=360, inner_radius=70):
    colors = colors or THEME_PALETTE[:max(3, len(df))]
    return alt.Chart(df, **_cfg(width, height)).mark_arc(innerRadius=inner_radius).encode(
        theta=alt.Theta(f"{count_col}:Q"),
        color=alt.Color(f"{field}:N", scale=alt.Scale(range=colors)),
        tooltip=[alt.Tooltip(f"{field}:N", title="Category"),
                 alt.Tooltip(f"{count_col}:Q", title="Count")]
    ).properties(title=title)

# =================== RENDER ===================
def render():
    st.title("ℹ️ Exploratory Data Analysis (EDA)")
    st.caption("Datos servidos desde DuckDB (cacheados).")

    with st.sidebar:
        st.markdown("### ⚙️ Filtros EDA")
        year_min = st.number_input("Año mínimo", value=2016, min_value=2000, max_value=2030, step=1)
        limit = st.select_slider("Límite de filas (para cargar más rápido)",
                                 options=[None, 50_000, 100_000, 200_000], value=100_000)
        if st.button("Recargar datos"):
            st.cache_data.clear()

    # ⬇️ Carga desde DuckDB y normaliza columnas
    df = load_info_df(year_min=year_min, limit=None if limit is None else int(limit))
    if df.empty:
        st.warning("No hay datos para mostrar en Info.")
        return

    # Cálculos cacheados
    agg = compute_eda_aggregates(df)
    df_metrics = get_violence_time_metrics_cached()

    # Contexto
    st.header("Contexto del Problema")
    st.markdown("""
    La Ciudad de México enfrenta desafíos significativos en materia de seguridad.
    Este estudio analiza incidentes de robo dividiéndolos en **violentos** y **no violentos**
    para identificar patrones temporales y espaciales.
    """)

    st.header("Hipótesis Principal")
    st.markdown("""
    > "Al menos el 80% de los crímenes violentos ocurren de noche,
    mientras que al menos el 80% de los crímenes no violentos ocurren de día."
    """)

    st.subheader("Resultados del Análisis de Horarios")
    if not df_metrics.empty:
        df_metrics = df_metrics.rename(columns={"franja_horaria": "Periodo", "porcentaje": "Porcentaje"})
        st.dataframe(df_metrics, use_container_width=True)
        try:
            noche_pct = df_metrics[df_metrics['Periodo'] == 'Noche (19-07)']['Porcentaje'].iloc[0]
            dia_pct = df_metrics[df_metrics['Periodo'] == 'Día (07-19)']['Porcentaje'].iloc[0]
            c1, c2 = st.columns(2)
            c1.metric("Crímenes Violentos de Noche (19:00-07:00)", f"{noche_pct:.1f}%")
            c2.metric("Crímenes Violentos de Día (07:00-19:00)", f"{dia_pct:.1f}%")
        except IndexError:
            st.warning("No se pudieron calcular las métricas de día/noche.")
    else:
        st.warning("No se pudieron cargar las métricas de violencia.")

    st.info("Conclusión: la hipótesis no se sostiene totalmente, aunque la noche sigue siendo significativa.")

    # === UNIVARIATE ===
    with st.expander("📈 Univariate: Years, Months, Hours", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            st.altair_chart(
                bar_chart(agg["year_counts"], "year", "count",
                          sort=list(range(2015, 2025)), title="Crimes by Year",
                          color=THEME_PALETTE[0],
                          x_axis=_axis("Year"), y_axis=_axis("Number of Cases")),
                use_container_width=True
            )
        with c2:
            st.altair_chart(
                bar_chart(agg["month_counts"], "month_name", "count",
                          sort=[MONTH_NAMES[m] for m in range(1,13)], title="Crimes by Month",
                          color=THEME_PALETTE[2],
                          x_axis=_axis("Month"), y_axis=_axis("Number of Cases")),
                use_container_width=True
            )
        st.altair_chart(
            bar_chart(agg["hour_counts"], "hour", "count", sort=list(range(24)),
                      title="Distribution of Crimes by Hour of Day",
                      color=THEME_PALETTE[1],
                      x_axis=_axis("Hour of Day"), y_axis=_axis("Number of Cases"), width=980),
            use_container_width=True
        )

    # === SPATIAL & CATEGORIES ===
    with st.expander("🗺️ Spatial & Categories", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.altair_chart(
                barh_chart(agg["alc_counts"], "alcaldia", "count",
                           title="Distribution of Crimes by District",
                           color=THEME_PALETTE[3],
                           x_axis=_axis("Number of Cases"), y_axis=_axis("District", orient="left")),
                use_container_width=True
            )
        with c2:
            st.altair_chart(
                barh_chart(agg["col_counts"], "colonia", "count",
                           title="Top 10 Neighborhoods by Crime Count",
                           color=THEME_PALETTE[4],
                           x_axis=_axis("Number of Cases"), y_axis=_axis("Neighborhood", orient="left")),
                use_container_width=True
            )
        st.altair_chart(
            barh_chart(agg["del_counts"], "delito", "count",
                       title="Top 10 Crimes by Count", color=THEME_PALETTE[5],
                       x_axis=_axis("Number of Cases"), y_axis=_axis("Crime Type", orient="left"), height=320),
            use_container_width=True
        )

    # === CLASSIFICATION & VIOLENCE ===
    with st.expander("🚨 Classification & Violence", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.altair_chart(
                donut_chart(agg["cls_counts"], "classification", "count",
                            title="Distribution by Crime Classification"),
                use_container_width=True
            )
        with c2:
            v_counts = agg["v_counts"]
            if not v_counts.empty:
                colors = ["#1E3A8A", "#A3B3C2"] if set(v_counts["violence"]) >= {"Violent", "Non-Violent"} else THEME_PALETTE[:len(v_counts)]
                st.altair_chart(
                    donut_chart(v_counts, "violence", "count",
                                title="Violent vs Non-Violent", colors=colors),
                    use_container_width=True
                )
            else:
                st.info("No se encontró columna 'violence_type'.")

        stacked = agg["stacked"]
        if not stacked.empty:
            order_mun = (stacked.groupby("alcaldia_std")["count"].sum().sort_values(ascending=True).index.tolist())
            ch = alt.Chart(stacked, **_cfg(width=900, height=420)).mark_bar().encode(
                x=alt.X("count:Q", axis=_axis("Number of Cases")),
                y=alt.Y("alcaldia_std:N", sort=order_mun, axis=_axis("Municipality", orient="left")),
                color=alt.Color("crime_classification:N",
                                scale=alt.Scale(range=THEME_PALETTE[:max(3, stacked['crime_classification'].nunique())])),
                tooltip=["alcaldia_std:N","crime_classification:N","count:Q"]
            ).properties(title="Municipality vs Crime Classification (stacked)")
            st.altair_chart(ch, use_container_width=True)
        else:
            st.info("Sin datos para el stacked por municipio y clasificación.")

    # === TRENDS ===
    with st.expander("📈 Trends over Time", expanded=False):
        if not agg["yearly_mun_top"].empty:
            st.altair_chart(
                line_chart(agg["yearly_mun_top"], "anio_inicio", "count", color_field="alcaldia_std",
                           title="Trend of Crimes per Year by Municipality (Top 8)",
                           width=950, height=420,
                           x_axis=_axis("Year"), y_axis=_axis("Number of Cases")),
                use_container_width=True
            )

        if not agg["year_class"].empty:
            st.altair_chart(
                line_chart(agg["year_class"], "anio_inicio", "count", color_field="crime_classification",
                           title="Trend of Crimes per Year by Classification",
                           width=950, height=420,
                           x_axis=_axis("Year"), y_axis=_axis("Number of Cases")),
                use_container_width=True
            )

        if not agg["yearly_viol"].empty:
            st.altair_chart(
                line_chart(agg["yearly_viol"], "anio_inicio", "count", color_field="violence_type",
                           title="Trend of Violent and Non-Violent Crimes per Year",
                           width=950, height=420,
                           x_axis=_axis("Year"), y_axis=_axis("Number of Cases")),
                use_container_width=True
            )
