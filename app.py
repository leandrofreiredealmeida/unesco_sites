from __future__ import annotations

import streamlit as st
import plotly.express as px

from unesco_sites.models.recommender import get_recommendations
from unesco_sites.models.similarity import CAT_COLOR_MAP, CATEGORIES, SHORT_CAT
from unesco_sites.pipeline import PipelineState, build

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    layout="wide",
    page_title="UNESCO World Heritage Sites",
    page_icon="🏛",
    initial_sidebar_state="collapsed",
)

# Remove default Streamlit padding so the map fills the viewport
st.markdown(
    """
    <style>
      section.main > div { padding-top: 0.5rem; padding-bottom: 0; }
      [data-testid="stAppViewContainer"] { background-color: #2E3440; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Data & model loading (cached) ─────────────────────────────────────────────


@st.cache_resource(show_spinner="Carregando dados e modelos…")
def _load_pipeline() -> PipelineState:
    return build()


state = _load_pipeline()
df = state.df
tfidf_matrix = state.tfidf_matrix

# ── Session state for selected site ──────────────────────────────────────────
# Persists the selection across reruns triggered by the slider.

if "site_cd" not in st.session_state:
    st.session_state["site_cd"] = None
# Incrementing this key remounts the chart with no selection, used by the clear button.
if "map_key" not in st.session_state:
    st.session_state["map_key"] = 0

# ── Layout: map (left) | panel (right) ───────────────────────────────────────

map_col, panel_col = st.columns([7, 3], gap="small")

# ── Map ───────────────────────────────────────────────────────────────────────

with map_col:
    df_plot = df.assign(df_idx=df.index)

    fig = px.scatter_mapbox(
        df_plot,
        lat="latitude",
        lon="longitude",
        color="category_thematic",
        color_discrete_map=CAT_COLOR_MAP,
        category_orders={"category_thematic": CATEGORIES},
        hover_name="name_en",
        hover_data={
            "category_thematic": True,
            "region": True,
            "year_inscribed": True,
            "latitude": False,
            "longitude": False,
            "df_idx": False,
        },
        custom_data=[
            "df_idx",
            "name_en",
            "category_thematic",
            "category",
            "region",
            "year_inscribed",
        ],
        zoom=1.3,
        center={"lat": 20, "lon": 10},
        mapbox_style="open-street-map",
        height=820,
    )
    fig.update_traces(marker={"size": 7, "opacity": 0.85})
    fig.update_layout(
        paper_bgcolor="#2E3440",
        font={"color": "#ECEFF4", "family": "sans-serif"},
        legend={
            "title": {"text": "Categoria Temática", "font": {"color": "#D8DEE9", "size": 12}},
            "bgcolor": "#3B4252",
            "bordercolor": "#4C566A",
            "borderwidth": 1,
            "font": {"color": "#D8DEE9", "size": 10},
        },
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        uirevision="map",  # keeps zoom/pan state across reruns
    )

    selected = st.plotly_chart(
        fig,
        on_select="rerun",
        use_container_width=True,
        key=f"map_{st.session_state['map_key']}",
    )

    # Persist selection so slider reruns don't lose the chosen site
    if selected and selected.selection.points:
        st.session_state["site_cd"] = selected.selection.points[0]["customdata"]

# ── Panel ─────────────────────────────────────────────────────────────────────

with panel_col:
    site_cd: list | None = st.session_state.get("site_cd")

    if site_cd is None:
        # ── Empty state ───────────────────────────────────────────────────────
        st.markdown("## 🌍 Explorar sítios")
        st.markdown(
            "Clique em qualquer marcador no mapa para descobrir "
            "sítios geograficamente próximos ou tematicamente similares."
        )
        st.divider()
        st.markdown(f"**{len(df):,} sítios** em {df['region'].nunique()} regiões")
        st.markdown(f"**{df['category_thematic'].nunique()} categorias** temáticas")
        st.markdown(f"Inscrições de **{int(df['year_inscribed'].min())}** a **{int(df['year_inscribed'].max())}**")

        # Mini distribution
        counts = (
            df["category_thematic"]
            .map(SHORT_CAT)
            .value_counts()
            .sort_values(ascending=False)
        )
        for cat_short, n in counts.items():
            color = next(
                (CAT_COLOR_MAP[k] for k, v in SHORT_CAT.items() if v == cat_short),
                "#88C0D0",
            )
            pct = n / len(df) * 100
            st.markdown(
                f"""
                <div style="display:flex; align-items:center; gap:8px; margin:3px 0;">
                  <span style="background:{color}; width:10px; height:10px;
                               border-radius:50%; flex-shrink:0;"></span>
                  <span style="color:#D8DEE9; font-size:0.82em; flex:1;">{cat_short}</span>
                  <span style="color:#81A1C1; font-size:0.82em; width:32px; text-align:right;">{n}</span>
                  <div style="background:#4C566A; border-radius:3px; height:6px; width:80px; flex-shrink:0;">
                    <div style="background:{color}; width:{pct:.1f}%; height:6px; border-radius:3px;"></div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    else:
        # ── Site selected ─────────────────────────────────────────────────────
        site_idx = int(site_cd[0])
        site_name = str(site_cd[1])
        site_thematic = str(site_cd[2])
        site_unesco = str(site_cd[3])
        site_region = str(site_cd[4])
        raw_year = site_cd[5]
        site_year = int(raw_year) if raw_year and str(raw_year) not in ("nan", "None") else "—"

        color = CAT_COLOR_MAP.get(site_thematic, "#88C0D0")
        short = SHORT_CAT.get(site_thematic, site_thematic)

        # ── Site card ─────────────────────────────────────────────────────────
        st.markdown(
            f"""
            <div style="background:#3B4252; border-left:4px solid {color};
                        padding:12px 16px; border-radius:6px; margin-bottom:10px;">
              <div style="font-size:1.05em; font-weight:700; color:#ECEFF4;
                          line-height:1.35;">{site_name}</div>
              <div style="margin-top:6px; display:flex; flex-wrap:wrap; gap:6px; align-items:center;">
                <span style="background:{color}; color:#2E3440; padding:2px 9px;
                             border-radius:12px; font-size:0.78em; font-weight:600;">
                  {short}
                </span>
                <span style="color:#A0B0C0; font-size:0.8em;">{site_unesco}</span>
                <span style="color:#A0B0C0; font-size:0.8em;">·</span>
                <span style="color:#A0B0C0; font-size:0.8em;">{site_year}</span>
              </div>
              <div style="color:#81A1C1; font-size:0.78em; margin-top:4px;">{site_region}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Slider ────────────────────────────────────────────────────────────
        geo_weight = st.slider(
            "Critério de recomendação",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            key="geo_weight",
        )
        lbl_col1, lbl_col2 = st.columns(2)
        lbl_col1.caption("← Similaridade temática")
        lbl_col2.caption("Proximidade geo →")

        # ── Recommendations ───────────────────────────────────────────────────
        st.markdown(
            f"### Gostou de _{site_name}_?\n**Talvez você possa gostar de…**"
        )

        recs = get_recommendations(
            site_idx,
            df,
            tfidf_matrix,
            top_n=10,
            geo_weight=geo_weight,
        )

        max_score = recs["score"].max() or 1.0

        for _, row in recs.iterrows():
            rec_color = CAT_COLOR_MAP.get(row["category_thematic"], "#88C0D0")
            rec_short = SHORT_CAT.get(row["category_thematic"], row["category_thematic"])
            bar_pct = row["score"] / max_score * 100
            dist_str = f"{row['geo_dist_km']:,} km"

            st.markdown(
                f"""
                <div style="background:#3B4252; padding:9px 12px; border-radius:5px;
                            margin-bottom:6px; border-left:3px solid {rec_color};">
                  <div style="color:#ECEFF4; font-size:0.9em; font-weight:600;
                              line-height:1.3;">{row["name_en"]}</div>
                  <div style="display:flex; gap:6px; align-items:center;
                              margin-top:5px; flex-wrap:wrap;">
                    <span style="background:{rec_color}; color:#2E3440; padding:1px 7px;
                                 border-radius:10px; font-size:0.73em; font-weight:600;">
                      {rec_short}
                    </span>
                    <span style="color:#81A1C1; font-size:0.76em;">📍 {dist_str}</span>
                    <span style="color:#81A1C1; font-size:0.76em;">
                      🔤 {row["nlp_sim"]:.3f}
                    </span>
                  </div>
                  <div style="background:#434C5E; border-radius:3px; height:4px; margin-top:6px;">
                    <div style="background:{rec_color}; width:{bar_pct:.1f}%;
                                height:4px; border-radius:3px;"></div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if st.button("Limpar seleção", use_container_width=True):
            st.session_state["site_cd"] = None
            st.session_state["map_key"] += 1
            st.rerun()
