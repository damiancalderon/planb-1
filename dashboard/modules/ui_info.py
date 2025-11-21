# modules/ui_info.py
import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px
import plotly.graph_objects as go  # Para los gauges

# ✅ helpers propios
from .helpers import MONTH_NAMES
# ✅ runner DuckDB centralizado
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
MONTH_NUM = {v: k for k, v in MONTH_NAMES.items()}  # "Enero"->1, etc.

def _cfg(width=700, height=350):
    return {"width": width, "height": height}

def _axis(title, orient=None):
    base = {"title": title, "grid": True, "gridColor": "#33415555"}
    if orient:
        base["orient"] = orient
    return base

def _safe_uniques(series: pd.Series) -> list:
    """Regresa valores únicos limpios (sin NaN, '', 'nan', 'delito de bajo impacto')."""
    try:
        vals = []
        for v in series.dropna().unique().tolist():
            s = str(v).strip()
            if not s:
                continue
            low = s.lower()
            if low in {"nan", "none", "delito de bajo impacto", "delitos de bajo impacto", "bajo impacto"}:
                continue
            vals.append(s)
        vals = sorted(vals)
    except Exception:
        vals = []
    return vals

# =================== DATA LOAD (DuckDB) ===================
@st.cache_data(ttl=3600, show_spinner=False)
def load_info_df(year_min: int = 2016, limit: int | None = None) -> pd.DataFrame:
    """
    Trae columnas necesarias desde DuckDB y las normaliza a nombres esperados.
    Por defecto carga TODO (limit=None).
    """
    limit_clause = "LIMIT ?" if limit is not None else ""
    params = [year_min] + ([int(limit)] if limit is not None else [])

    query = f"""
        SELECT
            anio_hecho                                     AS anio_inicio,
            EXTRACT(MONTH FROM CAST(fecha_hecho AS DATE))  AS mes_inicio,
            EXTRACT(HOUR  FROM hora_hecho)                 AS hour_hecho,
            alcaldia_hecho                                 AS alcaldia_std,
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

    # Normalizaciones numéricas
    for col in ["anio_inicio", "mes_inicio", "hour_hecho"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for needed in ["alcaldia_std", "colonia_std", "delito_std", "crime_classification", "violence_type"]:
        if needed not in df.columns:
            df[needed] = pd.Series(dtype="object")

    # Limpieza de strings y mapeo de 'nan' / bajo impacto a NA
    low_impact_labels = {
        "delito de bajo impacto",
        "delitos de bajo impacto",
        "bajo impacto"
    }
    for col in ["delito_std", "crime_classification"]:
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip()
            df[col] = df[col].replace(["", "nan", "NaN", "None"], pd.NA)
            df[col] = df[col].mask(
                df[col].str.lower().isin(low_impact_labels),
                pd.NA
            )

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
    YEAR_RANGE, MONTH_RANGE, HOUR_RANGE = range(2015, 2026), range(1, 13), range(0, 24)

    df = df.copy()
    df["anio_inicio"] = pd.to_numeric(df.get("anio_inicio"), errors="coerce")
    df["mes_inicio"]  = pd.to_numeric(df.get("mes_inicio"),  errors="coerce")
    if "hour_hecho" in df.columns:
        df["hour_hecho"] = pd.to_numeric(df["hour_hecho"], errors="coerce")

    # Drop rows with NaN in 'alcaldia_std' to prevent it from appearing in graphs
    if "alcaldia_std" in df.columns:
        df["alcaldia_std"] = df["alcaldia_std"].replace("", pd.NA)
        df = df.dropna(subset=["alcaldia_std"])

    year_counts = (
        df.dropna(subset=["anio_inicio"])
          .groupby("anio_inicio").size()
          .reindex(YEAR_RANGE, fill_value=0)
          .reset_index(name="count").rename(columns={"anio_inicio": "year"})
    )
    month_counts = (
        df.dropna(subset=["mes_inicio"])
          .groupby("mes_inicio").size()
          .reindex(MONTH_RANGE, fill_value=0)
          .reset_index(name="count").rename(columns={"mes_inicio": "month"})
    )
    month_counts["month_name"] = month_counts["month"].map(
        lambda m: MONTH_NAMES.get(int(m), str(m))
    )

    if "hour_hecho" in df.columns:
        hour_counts = (
            df.dropna(subset=["hour_hecho"]).groupby("hour_hecho").size()
              .reindex(HOUR_RANGE, fill_value=0)
              .reset_index(name="count").rename(columns={"hour_hecho": "hour"})
        )
    else:
        hour_counts = pd.DataFrame({"hour": list(HOUR_RANGE), "count": 0})

    alc_counts = df.get("alcaldia_std", pd.Series([], dtype="object")).value_counts().sort_values(ascending=True).reset_index()
    if not alc_counts.empty:
        alc_counts.columns = ["alcaldia", "count"]
    else:
        alc_counts = pd.DataFrame(columns=["alcaldia", "count"])

    col_counts = df.get("colonia_std", pd.Series([], dtype="object")).value_counts().nlargest(10).sort_values(ascending=True).reset_index()
    if not col_counts.empty:
        col_counts.columns = ["colonia", "count"]
    else:
        col_counts = pd.DataFrame(columns=["colonia", "count"])

    # Conteo de delitos (sin NaN ni bajo impacto)
    del_series = df.get("delito_std", pd.Series([], dtype="object")).astype("string")
    if not del_series.empty:
        del_series = del_series.dropna().apply(lambda x: x.strip())
        del_series = del_series[
            ~del_series.str.lower().isin(
                {"nan", "none", "delito de bajo impacto", "delitos de bajo impacto", "bajo impacto"}
            )
        ]
        del_counts = del_series.value_counts().nlargest(10).sort_values(ascending=True).reset_index()
        if not del_counts.empty:
            del_counts.columns = ["delito", "count"]
        else:
            del_counts = pd.DataFrame(columns=["delito", "count"])
    else:
        del_counts = pd.DataFrame(columns=["delito", "count"])

    # Conteo de clasificación (sin NaN ni bajo impacto)
    cls_series = df.get("crime_classification", pd.Series([], dtype="object")).astype("string")
    if not cls_series.empty:
        cls_series = cls_series.dropna().apply(lambda x: x.strip())
        cls_series = cls_series[
            ~cls_series.str.lower().isin(
                {"nan", "none", "delito de bajo impacto", "delitos de bajo impacto", "bajo impacto"}
            )
        ]
        cls_counts = cls_series.value_counts().reset_index()
        if not cls_counts.empty:
            cls_counts.columns = ["classification", "count"]
        else:
            cls_counts = pd.DataFrame(columns=["classification", "count"])
    else:
        cls_counts = pd.DataFrame(columns=["classification", "count"])

    if "violence_type" in df.columns:
        v_counts_raw = df["violence_type"].astype("string")
        v_counts_raw = v_counts_raw.dropna().apply(lambda x: x.strip())
        v_counts = v_counts_raw.value_counts().reset_index()
        v_counts.columns = ["violence", "count"]
    else:
        v_counts = pd.DataFrame(columns=["violence", "count"])

    if {"alcaldia_std", "crime_classification"}.issubset(df.columns):
        stacked = df.groupby(["alcaldia_std", "crime_classification"]).size().reset_index(name="count")
        stacked["crime_classification"] = stacked["crime_classification"].astype("string")
        stacked = stacked[
            ~stacked["crime_classification"].str.lower().isin(
                {"nan", "none", "delito de bajo impacto", "delitos de bajo impacto", "bajo impacto"}
            )
        ]
    else:
        stacked = pd.DataFrame(columns=["alcaldia_std", "crime_classification", "count"])

    if {"anio_inicio", "alcaldia_std"}.issubset(df.columns):
        yearly_mun = (
            df.dropna(subset=["anio_inicio"])
              .groupby(["anio_inicio", "alcaldia_std"]).size().reset_index(name="count")
        )
        totals = yearly_mun.groupby("alcaldia_std")["count"].sum().sort_values(ascending=False)
        top8 = totals.head(8).index.tolist()
        yearly_mun_top = yearly_mun[yearly_mun["alcaldia_std"].isin(top8)].copy()
        yearly_mun_top["anio_inicio"] = pd.Categorical(
            yearly_mun_top["anio_inicio"], categories=range(2015, 2026), ordered=True
        )
    else:
        top8, yearly_mun_top = [], pd.DataFrame(columns=["anio_inicio", "alcaldia_std", "count"])

    if {"anio_inicio", "crime_classification"}.issubset(df.columns):
        year_class = (
            df.dropna(subset=["anio_inicio"])
              .groupby(["anio_inicio", "crime_classification"]).size().reset_index(name="count")
        )
        year_class["anio_inicio"] = pd.Categorical(
            year_class["anio_inicio"], categories=range(2015, 2026), ordered=True
        )
        year_class["crime_classification"] = year_class["crime_classification"].astype("string")
        year_class = year_class[
            ~year_class["crime_classification"].str.lower().isin(
                {"nan", "none", "delito de bajo impacto", "delitos de bajo impacto", "bajo impacto"}
            )
        ]
    else:
        year_class = pd.DataFrame(columns=["anio_inicio", "crime_classification", "count"])

    if {"anio_inicio", "violence_type"}.issubset(df.columns):
        yearly_viol = (
            df.dropna(subset=["anio_inicio"])
              .groupby(["anio_inicio", "violence_type"]).size().reset_index(name="count")
        )
        yearly_viol["anio_inicio"] = pd.Categorical(
            yearly_viol["anio_inicio"], categories=range(2015, 2026), ordered=True
        )
    else:
        yearly_viol = pd.DataFrame(columns=["anio_inicio", "violence_type", "count"])

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
        "top8": top8,
    }

# =================== CHART PRIMS ===================
def bar_chart(df, x, y, *, x_type="O", y_type="Q", sort=None, title="", color=None, width=700, height=350, x_axis=None, y_axis=None):
    return alt.Chart(df, **_cfg(width, height)).mark_bar(color=color).encode(
        x=alt.X(f"{x}:{x_type}", sort=sort, axis=x_axis),
        y=alt.Y(f"{y}:{y_type}", axis=y_axis),
        tooltip=[f"{x}:{x_type}", f"{y}:{y_type}"],
    ).properties(title=title)

def barh_chart(df, y, x, *, y_type="N", x_type="Q", sort=None, title="", color=None, width=700, height=350, x_axis=None, y_axis=None):
    return alt.Chart(df, **_cfg(width, height)).mark_bar(color=color).encode(
        x=alt.X(f"{x}:{x_type}", axis=x_axis),
        y=alt.Y(f"{y}:{y_type}", sort=sort, axis=y_axis),
        tooltip=[f"{y}:{y_type}", f"{x}:{x_type}"],
    ).properties(title=title)

def line_chart(df, x, y, color_field=None, *, x_type="O", y_type="Q", title="", width=700, height=350, x_axis=None, y_axis=None, point=False):
    mk = alt.Chart(df, **_cfg(width, height)).mark_line(point=point, strokeWidth=2).encode(
        x=alt.X(f"{x}:{x_type}", axis=x_axis),
        y=alt.Y(f"{y}:{y_type}", axis=y_axis),
        tooltip=[f"{x}:{x_type}", f"{y}:{y_type}"],
    )
    if color_field and color_field in df.columns and not df.empty:
        uniq = max(1, df[color_field].nunique())
        mk = mk.encode(
            color=alt.Color(
                f"{color_field}:N",
                scale=alt.Scale(range=THEME_PALETTE[:uniq]),
            )
        )
    return mk.properties(title=title)

def donut_chart(df, field, count_col, *, title="", colors=None, width=700, height=360, inner_radius=70):
    n = max(3, len(df)) if isinstance(df, pd.DataFrame) else 3
    colors = colors or THEME_PALETTE[:n]
    return alt.Chart(df, **_cfg(width, height)).mark_arc(innerRadius=inner_radius).encode(
        theta=alt.Theta(f"{count_col}:Q"),
        color=alt.Color(f"{field}:N", scale=alt.Scale(range=colors)),
        tooltip=[
            alt.Tooltip(f"{field}:N", title="Categoría"),
            alt.Tooltip(f"{count_col}:Q", title="Conteo"),
        ],
    ).properties(title=title)

def heatmap_chart(df, x, y, z, *, title="", width=700, height=350, x_axis=None, y_axis=None):
    return alt.Chart(df, **_cfg(width, height)).mark_rect().encode(
        x=alt.X(f"{x}:N", axis=x_axis),
        y=alt.Y(f"{y}:N", axis=y_axis),
        color=alt.Color(f"{z}:Q", scale=alt.Scale(scheme="blues")),
        tooltip=[f"{x}:N", f"{y}:N", f"{z}:Q"],
    ).properties(title=title)

# =================== RENDER ===================
def render():
    st.title("Análisis Exploratorio de Datos (AED)")

    # Carga TODO (sin límite por defecto)
    base_df = load_info_df(year_min=2015, limit=None)
    if base_df.empty:
        st.warning("No hay datos para mostrar en Info.")
        return

    # =================== SIDEBAR FILTERS (globales) ===================
    with st.sidebar:
        st.markdown("### Filtros Globales")

        # Rango de años dinámico
        y_min, y_max = int(base_df["anio_inicio"].min()), int(base_df["anio_inicio"].max())
        year_range = st.slider(
            "Años",
            min_value=y_min,
            max_value=y_max,
            value=(max(y_min, 2016), y_max),
            step=1,
        )

        # Meses (multi)
        month_options = [MONTH_NAMES[m] for m in range(1, 13)]
        months_sel = st.multiselect("Meses", options=month_options, default=month_options)

        # Rango de horas
        hour_has_col = "hour_hecho" in base_df.columns
        if hour_has_col:
            hour_range = st.slider("Hora del día", 0, 23, (0, 23), step=1)
        else:
            hour_range = (0, 23)

        # Alcaldías / Clasificación / Violencia / Delitos
        alcs = _safe_uniques(base_df["alcaldia_std"]) if "alcaldia_std" in base_df else []
        cls  = _safe_uniques(base_df["crime_classification"]) if "crime_classification" in base_df else []
        viol = _safe_uniques(base_df["violence_type"]) if "violence_type" in base_df else []
        dels = _safe_uniques(base_df["delito_std"]) if "delito_std" in base_df else []

        alc_sel = st.multiselect("Alcaldías", options=alcs, default=alcs[:])
        cls_sel = st.multiselect("Clasificación", options=cls, default=cls[:])
        viol_sel = st.multiselect("Violencia", options=viol, default=viol[:])

        # Para no saturar, top N delitos por frecuencia como opciones por defecto
        if dels:
            top_del = (
                pd.Series(base_df["delito_std"], dtype="object")
                .value_counts()
                .head(30)
                .index
                .tolist()
            )
            top_del = [
                d
                for d in top_del
                if str(d).strip().lower()
                not in {
                    "nan",
                    "none",
                    "delito de bajo impacto",
                    "delitos de bajo impacto",
                    "bajo impacto",
                }
            ]
            delito_default = top_del
        else:
            delito_default = []
        delito_sel = st.multiselect(
            "Delitos (30 principales por defecto)", options=dels, default=delito_default
        )

        if st.button("🔄 Reset filtros"):
            st.experimental_rerun()

    # =================== APPLY FILTERS ===================
    df = base_df.copy()

    # Filtro robusto de delitos de bajo impacto
    if "delito_std" in df.columns:
        df["delito_std"] = df["delito_std"].astype("string")
        mask_bajo = df["delito_std"].str.lower().str.contains("bajo impacto", na=False)
        df = df[~mask_bajo]

    df = df[(df["anio_inicio"] >= year_range[0]) & (df["anio_inicio"] <= year_range[1])]

    months_num = [MONTH_NUM.get(mn, None) for mn in months_sel]
    months_num = [m for m in months_num if m is not None]
    if months_num:
        df = df[df["mes_inicio"].isin(months_num)]

    if hour_has_col and hour_range != (0, 23):
        h0, h1 = hour_range
        df = df[(df["hour_hecho"] >= h0) & (df["hour_hecho"] <= h1)]

    if "alcaldia_std" in df.columns and alc_sel:
        df = df[df["alcaldia_std"].isin(alc_sel)]

    if "crime_classification" in df.columns and cls_sel:
        df = df[df["crime_classification"].isin(cls_sel)]

    if "violence_type" in df.columns and viol_sel:
        df = df[df["violence_type"].isin(viol_sel)]

    if "delito_std" in df.columns and delito_sel:
        df = df[df["delito_std"].isin(delito_sel)]

    # =================== AGG & METRICS ===================
    agg = compute_eda_aggregates(df)
    df_metrics = get_violence_time_metrics_cached()

    # ====== CONTEXTO + KPIs ======
    st.header("Contexto del Problema")
    st.markdown(
        """
    La Ciudad de México enfrenta desafíos significativos en materia de seguridad.
    Este estudio analiza incidentes de robo dividiéndolos en **violentos** y **no violentos**
    para identificar patrones temporales y espaciales.
    """
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Incidentes filtrados", f"{len(df):,}")
    c2.metric("Años seleccionados", f"{year_range[0]}–{year_range[1]}")
    c3.metric("Alcaldías activas", str(len(alc_sel) if alc_sel else 0))

    st.subheader("Resultados del Análisis de Horarios (Violencia)")
    if not df_metrics.empty:
        df_metrics = df_metrics.rename(
            columns={"franja_horaria": "Periodo", "porcentaje": "Porcentaje"}
        )
        st.dataframe(df_metrics, use_container_width=True)

        try:
            noche_pct = float(
                df_metrics[df_metrics["Periodo"] == "Noche (19-07)"]["Porcentaje"].iloc[0]
            )
            dia_pct = float(
                df_metrics[df_metrics["Periodo"] == "Día (07-19)"]["Porcentaje"].iloc[0]
            )

            st.markdown("#### Distribución de crímenes violentos por horario")

            col1, col2 = st.columns(2)

            # Gauge Noche
            with col1:
                fig_noche = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=noche_pct,
                        number={"suffix": "%"},
                        title={
                            "text": "Crímenes Violentos de Noche<br><span style='font-size:0.8em'>(19:00–07:00)</span>"
                        },
                        gauge={
                            "axis": {"range": [0, 100]},
                            "bar": {"color": "#1E3A8A"},
                            "steps": [
                                {"range": [0, 33], "color": "#E5E7EB"},
                                {"range": [33, 66], "color": "#93C5FD"},
                                {"range": [66, 100], "color": "#1D4ED8"},
                            ],
                        },
                    )
                )
                fig_noche.update_layout(
                    margin=dict(l=20, r=20, t=60, b=10),
                    height=280,
                )
                st.plotly_chart(fig_noche, use_container_width=True)

            # Gauge Día
            with col2:
                fig_dia = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=dia_pct,
                        number={"suffix": "%"},
                        title={
                            "text": "Crímenes Violentos de Día<br><span style='font-size:0.8em'>(07:00–19:00)</span>"
                        },
                        gauge={
                            "axis": {"range": [0, 100]},
                            "bar": {"color": "#F59E0B"},
                            "steps": [
                                {"range": [0, 33], "color": "#FEF3C7"},
                                {"range": [33, 66], "color": "#FCD34D"},
                                {"range": [66, 100], "color": "#D97706"},
                            ],
                        },
                    )
                )
                fig_dia.update_layout(
                    margin=dict(l=20, r=20, t=60, b=10),
                    height=280,
                )
                st.plotly_chart(fig_dia, use_container_width=True)

        except IndexError:
            st.warning("No se pudieron calcular las métricas de día/noche.")
    else:
        st.warning("No se pudieron cargar las métricas de violencia.")

    st.info(
        "Conclusión: la hipótesis no se sostiene totalmente, aunque la noche sigue siendo significativa."
    )

    # === UNIVARIATE ===
    st.header("Distribuciones Univariadas")
    cU1, cU2 = st.columns(2)
    with cU1:
        st.altair_chart(
            bar_chart(
                agg["year_counts"],
                "year",
                "count",
                sort=list(range(2015, 2026)),
                title="Crímenes por Año",
                color=THEME_PALETTE[0],
                x_axis=_axis("Año"),
                y_axis=_axis("Número de casos"),
            ),
            use_container_width=True,
        )
    with cU2:
        st.altair_chart(
            bar_chart(
                agg["month_counts"],
                "month_name",
                "count",
                sort=[MONTH_NAMES[m] for m in range(1, 13)],
                title="Crímenes por Mes",
                color=THEME_PALETTE[2],
                x_axis=_axis("Mes"),
                y_axis=_axis("Número de casos"),
            ),
            use_container_width=True,
        )
    st.altair_chart(
        bar_chart(
            agg["hour_counts"],
            "hour",
            "count",
            sort=list(range(24)),
            title="Distribución por Hora del Día",
            color=THEME_PALETTE[1],
            x_axis=_axis("Hora del día"),
            y_axis=_axis("Número de casos"),
            width=980,
        ),
        use_container_width=True,
    )

    # === SPATIAL & CATEGORIES ===
    st.header("Espacial & Categorías")
    cS1, cS2 = st.columns(2)
    with cS1:
        st.altair_chart(
            barh_chart(
                agg["alc_counts"],
                "alcaldia",
                "count",
                title="Distribución por Alcaldía",
                color=THEME_PALETTE[3],
                x_axis=_axis("Número de casos"),
                y_axis=_axis("Alcaldía", orient="left"),
            ),
            use_container_width=True,
        )
    with cS2:
        st.altair_chart(
            barh_chart(
                agg["col_counts"],
                "colonia",
                "count",
                title="10 colonias con más incidentes",
                color=THEME_PALETTE[4],
                x_axis=_axis("Número de casos"),
                y_axis=_axis("Colonia", orient="left"),
            ),
            use_container_width=True,
        )
    st.altair_chart(
        barh_chart(
            agg["del_counts"],
            "delito",
            "count",
            title="10 delitos con mayor frecuencia",
            color=THEME_PALETTE[5],
            x_axis=_axis("Número de casos"),
            y_axis=_axis("Tipo de delito", orient="left"),
            height=320,
        ),
        use_container_width=True,
    )

    # === CLASSIFICATION & VIOLENCE ===
    st.header("Clasificación & Violencia")
    cC1, cC2 = st.columns(2)

    # Donut Chart de Clasificación
    with cC1:
        st.altair_chart(
            donut_chart(
                agg["cls_counts"],
                "classification",
                "count",
                title="Distribución por Clasificación",
            ),
            use_container_width=True,
        )

    # Donut Chart de Violencia
    with cC2:
        v_counts = agg["v_counts"]
        if not v_counts.empty:
            colors = (
                ["#1E3A8A", "#A3B3C2"]
                if set(v_counts["violence"]) >= {"Violent", "Non-Violent"}
                else THEME_PALETTE[: len(v_counts)]
            )
            st.altair_chart(
                donut_chart(
                    v_counts,
                    "violence",
                    "count",
                    title="Violento vs No Violento",
                    colors=colors,
                ),
                use_container_width=True,
            )
        else:
            st.info("No se encontró columna 'violence_type'.")

    # === SPATIAL & CATEGORIES STACKED ===
    stacked = agg["stacked"]
    if not stacked.empty:
        order_mun = (
            stacked.groupby("alcaldia_std")["count"]
            .sum()
            .sort_values(ascending=True)
            .index
            .tolist()
        )
        ch = (
            alt.Chart(stacked, **_cfg(width=900, height=420))
            .mark_bar()
            .encode(
                x=alt.X("count:Q", axis=_axis("Número de casos")),
                y=alt.Y(
                    "alcaldia_std:N",
                    sort=order_mun,
                    axis=_axis("Alcaldía", orient="left"),
                ),
                color=alt.Color(
                    "crime_classification:N",
                    scale=alt.Scale(
                        range=THEME_PALETTE[
                            : max(3, stacked["crime_classification"].nunique())
                        ]
                    ),
                ),
                tooltip=["alcaldia_std:N", "crime_classification:N", "count:Q"],
            )
            .properties(title="Alcaldía vs Clasificación de Delito (apilado)")
        )
        st.altair_chart(ch, use_container_width=True)
    else:
        st.info("Sin datos para el stacked por alcaldía y clasificación.")

  
