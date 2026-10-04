import json

from citations import CitationRegistry, articles_from_turns, render_citations


class _Request:
    def __init__(self, name):
        self.name = name


class _ToolResult:
    def __init__(self, value, name="search_news"):
        self.value = value
        self.request = _Request(name)


class _Turn:
    def __init__(self, *contents):
        self.contents = list(contents)


def _article(url, **extra):
    return {"url": url, "titre": f"Titre {url}", "source": "RFI", **extra}


def test_number_est_cumulatif_dans_un_meme_tour():
    registry = CitationRegistry(lambda: [])
    first = registry.number([_article("a"), _article("b")])
    second = registry.number([_article("b"), _article("c")])
    assert [x["numero"] for x in first] == [1, 2]
    # "b" garde son numéro, "c" prend le suivant.
    assert [x["numero"] for x in second] == [2, 3]


def test_number_reprend_apres_l_historique():
    turns = [_Turn(_ToolResult([{"numero": 1, **_article("a")}]))]
    registry = CitationRegistry(lambda: turns)
    numbered = registry.number([_article("z"), _article("a")])
    assert [x["numero"] for x in numbered] == [2, 1]


def test_articles_from_turns_lit_les_valeurs_json_et_ignore_le_reste():
    turns = [
        _Turn(_ToolResult(json.dumps([{"numero": 1, **_article("a")}]))),
        _Turn(_ToolResult([{"numero": 2, **_article("b")}], "latest_news")),
        _Turn(_ToolResult([{"numero": 9, **_article("x")}], "autre_outil")),
        _Turn(_ToolResult("pas du json")),
        _Turn(object()),
    ]
    assert sorted(articles_from_turns(turns)) == [1, 2]


def test_changement_de_conversation_sans_collision():
    """Un numéro en attente d'une autre conversation ne masque pas
    l'article qui porte ce numéro dans la conversation affichée."""
    turns: list = []
    registry = CitationRegistry(lambda: turns)
    registry.number([_article("ancienne-conv")])  # numero 1, en attente
    turns.append(_Turn(_ToolResult([{"numero": 1, **_article("a")}])))
    assert registry.known()[1]["url"] == "a"
    assert [x["numero"] for x in registry.number([_article("n")])] == [2]


def test_render_renumerote_et_construit_les_sources():
    articles = {
        4: _article("u4", date="2026-08-01"),
        7: _article("u7"),
        9: _article("u9"),
    }
    out = render_citations("Fait A `[7]`. Fait B `[4, 7]`.", articles)
    assert "Fait A [`[1]`](u7). Fait B [`[2]`](u4) [`[1]`](u7)." in out
    sources = out.split("> **Sources**\n")[1]
    assert sources.splitlines()[0] == "> **[1]** RFI : [Titre u7](u7)"
    assert "> **[2]** RFI · 2026-08-01 : [Titre u4](u4)" in sources
    assert "u9" not in out  # non cité, donc absent des sources


def test_render_supprime_les_numeros_inconnus():
    out = render_citations("Fait `[1]`. Inventé `[42]`.", {1: _article("u1")})
    assert "Inventé." in out
    assert "42" not in out


def test_render_ne_touche_pas_aux_crochets_ordinaires():
    out = render_citations("En [2025], voir `[1]`.", {1: _article("u1")})
    assert "En [2025]," in out


def test_render_accepte_une_citation_sans_accents_graves():
    out = render_citations("Fait [1].", {1: _article("u1")})
    assert "Fait [`[1]`](u1)." in out


def test_render_retire_les_sources_ecrites_par_le_llm():
    text = "Fait `[1]`.\n\n> Sources\n> RFI - truc"
    out = render_citations(text, {1: _article("u1")})
    assert "truc" not in out
    assert out.count("Sources") == 1


def test_render_sans_article_ni_citation():
    assert render_citations("Bonjour.", {}) == "Bonjour."
    assert render_citations("Bonjour.", {1: _article("u1")}) == "Bonjour."
