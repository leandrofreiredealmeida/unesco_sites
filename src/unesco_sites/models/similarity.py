from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as pairwise_cosine

from unesco_sites import logger

# ── Thematic categories ──────────────────────────────────────────────────────

CATEGORY_PROTOTYPES: dict[str, str] = {
    "Ruínas e sítios arqueológicos": (
        "Ancient ruined city, temple ruins or abandoned archaeological settlement. "
        "Excavated remains of walls, columns, inscriptions and artifacts uncovering "
        "a prehistoric or historic civilization. Monumental structures in ruin, "
        "ceremonial complexes and unearthed dwellings of ancient inhabitants."
    ),
    "Centros urbanos vivos": (
        "Historic centre or old town of a city still inhabited today. The preserved "
        "historic district features traditional architecture, historic streets and an "
        "urban fabric of residential and commercial buildings. Historic town with "
        "continuous occupation and a living urban community."
    ),
    "Conjuntos religiosos e sagrados": (
        "Church, monastery, cathedral, mosque, temple or shrine of outstanding "
        "religious significance. Sacred site for pilgrimage, worship and spiritual "
        "devotion. Buddhist, Hindu, Islamic or Christian religious architecture and "
        "monastic community of monks, nuns or clergy."
    ),
    "Fortificações e sistemas defensivos": (
        "Fortress, castle, fortified city walls, ramparts, citadel or military "
        "bastion. Medieval or historic defensive system with towers, battlements "
        "and garrison. Fortified stronghold and military architecture built for "
        "defense and protection against attack."
    ),
    "Paisagens culturais e parques humanizados": (
        "Cultural landscape integrating human activity and natural environment. "
        "Terraced agricultural fields, vineyards, rice paddies or pastoral farmland "
        "shaped over centuries by traditional communities. Managed rural landscape "
        "reflecting a living heritage of farming and land use practices."
    ),
    "Palácios e sedes de poder político": (
        "Royal palace, imperial residence or seat of political power. Ceremonial "
        "governmental complex with throne room, state apartments, royal gardens and "
        "official halls. Dynastic capital or administrative headquarters representing "
        "monarchical authority and political power."
    ),
    "Necrópoles e espaços funerários": (
        "Necropolis, pyramid, mausoleum, burial ground or funerary monument dedicated "
        "to the deceased. Ancient tombs, burial chambers and mortuary structures "
        "reflecting funerary rituals and beliefs about death. Graves and memorial "
        "monuments marking the resting place of rulers or communities."
    ),
    "Engenharia e infraestrutura histórica": (
        "Historic engineering achievement such as aqueduct, bridge, canal, viaduct "
        "or railway. Technical infrastructure for water supply, transport or navigation. "
        "Outstanding hydraulic works, industrial canal, lighthouse or dam of remarkable "
        "engineering significance."
    ),
    "Conjuntos rurais e aldeias": (
        "Traditional rural village or historic agricultural settlement with vernacular "
        "architecture. Preserved farmhouses, barns and rural buildings constructed "
        "with local materials. Small community in a rural landscape representing "
        "an authentic and intact historic way of life."
    ),
    "Sítios industriais e tecnológicos": (
        "Industrial heritage site including mine, factory, ironworks, colliery or "
        "blast furnace. Historic manufacturing complex representing the industrial "
        "revolution and technological innovation. Coal mine, steel mill, workers "
        "housing and machinery of outstanding industrial heritage significance."
    ),
    "Arte rupestre e inscrições primitivas": (
        "Prehistoric cave paintings, rock art, petroglyphs or pictographs carved or "
        "painted on rock surfaces. Paleolithic or Neolithic images of animals and "
        "symbols created by hunter-gatherers. Painted cave shelter with ancient "
        "pigment drawings documenting early human artistic expression."
    ),
    "Monumentos comemorativos e memoriais": (
        "Monument, memorial or commemorative structure erected to honor a historic "
        "event, national hero or collective memory. Peace memorial, war monument or "
        "national landmark symbolizing freedom, independence or cultural identity. "
        "Statue or dedicated structure built to remember and celebrate."
    ),
}

CATEGORIES: list[str] = list(CATEGORY_PROTOTYPES.keys())

THEMATIC_COLORS: list[str] = [
    "#88C0D0",  # Ruínas
    "#A3BE8C",  # Centros urbanos
    "#EBCB8B",  # Religiosos
    "#BF616A",  # Fortificações
    "#B48EAD",  # Paisagens
    "#D08770",  # Palácios
    "#81A1C1",  # Necrópoles
    "#5E81AC",  # Engenharia
    "#8FBCBB",  # Aldeias rurais
    "#6B8E9F",  # Industriais
    "#C5956B",  # Arte rupestre
    "#9E7BB5",  # Memoriais
]

CAT_COLOR_MAP: dict[str, str] = dict(zip(CATEGORIES, THEMATIC_COLORS))

CATEGORY_EN: dict[str, str] = {
    "Ruínas e sítios arqueológicos": "Ruins & Archaeological Sites",
    "Centros urbanos vivos": "Living Urban Centres",
    "Conjuntos religiosos e sagrados": "Religious & Sacred Ensembles",
    "Fortificações e sistemas defensivos": "Fortifications & Defensive Systems",
    "Paisagens culturais e parques humanizados": "Cultural Landscapes",
    "Palácios e sedes de poder político": "Palaces & Seats of Power",
    "Necrópoles e espaços funerários": "Necropolises & Funerary Sites",
    "Engenharia e infraestrutura histórica": "Historic Engineering & Infrastructure",
    "Conjuntos rurais e aldeias": "Rural Ensembles & Villages",
    "Sítios industriais e tecnológicos": "Industrial & Technological Sites",
    "Arte rupestre e inscrições primitivas": "Rock Art & Early Inscriptions",
    "Monumentos comemorativos e memoriais": "Commemorative Monuments & Memorials",
}

SHORT_CAT: dict[str, str] = {
    "Ruínas e sítios arqueológicos": "Ruins",
    "Centros urbanos vivos": "Urban Centres",
    "Conjuntos religiosos e sagrados": "Religious",
    "Fortificações e sistemas defensivos": "Fortifications",
    "Paisagens culturais e parques humanizados": "Cultural Landscapes",
    "Palácios e sedes de poder político": "Palaces",
    "Necrópoles e espaços funerários": "Necropolises",
    "Engenharia e infraestrutura histórica": "Engineering",
    "Conjuntos rurais e aldeias": "Rural Villages",
    "Sítios industriais e tecnológicos": "Industrial",
    "Arte rupestre e inscrições primitivas": "Rock Art",
    "Monumentos comemorativos e memoriais": "Memorials",
}

MULTILABEL_THRESHOLD = 0.05

# ── Model functions ──────────────────────────────────────────────────────────


def build_tfidf(df: pd.DataFrame) -> tuple[TfidfVectorizer, csr_matrix]:
    """Fits a TF-IDF vectorizer on the full_text column.

    Args:
        df: DataFrame that already has no NaN values in full_text.

    Returns:
        Tuple of (fitted vectorizer, sparse TF-IDF matrix of shape n_sites x 8000).
    """
    logger.info("Fitting TF-IDF vectorizer on %d documents...", len(df))
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        max_features=8_000,
        sublinear_tf=True,
        ngram_range=(1, 2),
        min_df=2,
    )
    tfidf_matrix: csr_matrix = vectorizer.fit_transform(df["full_text"])
    sparsity = (1 - tfidf_matrix.nnz / (tfidf_matrix.shape[0] * tfidf_matrix.shape[1])) * 100
    logger.info(
        "TF-IDF fitted: %d docs x %d terms (sparsity %.1f%%)",
        tfidf_matrix.shape[0],
        tfidf_matrix.shape[1],
        sparsity,
    )
    return vectorizer, tfidf_matrix


def classify_thematic(
    df: pd.DataFrame,
    tfidf_matrix: csr_matrix,
    vectorizer: TfidfVectorizer,
) -> pd.DataFrame:
    """Assigns each site a primary thematic category via prototype matching.

    Projects each of the 12 prototype texts into the fitted TF-IDF space,
    then assigns to each site the category with the highest cosine similarity.

    Args:
        df: DataFrame aligned row-by-row with tfidf_matrix.
        tfidf_matrix: Sparse matrix (n_sites x n_terms).
        vectorizer: Already-fitted TfidfVectorizer.

    Returns:
        Copy of df with added columns category_thematic and categories_multi.
    """
    logger.info("Classifying %d sites into %d thematic categories...", len(df), len(CATEGORIES))
    proto_matrix: csr_matrix = vectorizer.transform(CATEGORY_PROTOTYPES.values())
    sim_matrix: np.ndarray = pairwise_cosine(tfidf_matrix, proto_matrix)
    sim_df = pd.DataFrame(sim_matrix, columns=CATEGORIES, index=df.index)

    result = df.copy()
    result["category_thematic"] = sim_df.idxmax(axis=1)
    result["categories_multi"] = [
        [cat for cat, score in row.items() if score >= MULTILABEL_THRESHOLD]
        for _, row in sim_df.iterrows()
    ]

    logger.info(
        "Classification done: %d unique primary categories assigned",
        result["category_thematic"].nunique(),
    )
    return result
