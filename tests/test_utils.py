import pytest

import utils


def test_normalize_pays_ignore_accents_et_casse():
    assert utils.normalize_pays(" Sénégal ") == "senegal"
    assert utils.normalize_pays("Côte d'Ivoire") == "cote d'ivoire"


def test_clean_text_decode_les_entites_html():
    assert utils.clean_text("Revue du 05 ao&#xfb;t  2026\n") == (
        "Revue du 05 août 2026"
    )


def test_source_name():
    assert utils.source_name("https://fr.allafrica.com/stories/1.html") == (
        "AllAfrica"
    )
    assert utils.source_name("https://www.lefaso.net/spip.php?article1") == (
        "leFaso.net"
    )
    # Domaine inconnu : titre du flux, sinon le domaine lui-même.
    assert utils.source_name("https://x.org/a", "Flux &amp; Co") == "Flux & Co"
    assert utils.source_name("https://www.x.org/a") == "x.org"


def test_is_press_review():
    assert utils.is_press_review("Revue de presse de l'Afrique francophone")
    assert not utils.is_press_review("Mali : la presse en revue")


@pytest.mark.parametrize(
    "raw, expected_iso",
    [
        ("Thu, 06 Aug 2026 20:48:21 +0000", "2026-08-06T20:48:21+00:00"),
        ("2026-08-06T22:44:00Z", "2026-08-06T22:44:00+00:00"),
        ("Thu, 06 Aug 2026 22:00:00 +0200", "2026-08-06T20:00:00+00:00"),
    ],
)
def test_parse_date_formats(raw, expected_iso):
    iso, epoch = utils.parse_date(raw)
    assert iso == expected_iso
    assert epoch > 0


def test_parse_date_illisible():
    assert utils.parse_date("") == ("", 0)
    assert utils.parse_date("pas une date") == ("", 0)


def test_date_to_epoch():
    start = utils.date_to_epoch("2026-08-06")
    end = utils.date_to_epoch("2026-08-06", end_of_day=True)
    assert end - start == 24 * 3600 - 1
    with pytest.raises(ValueError, match="AAAA-MM-JJ"):
        utils.date_to_epoch("hier")


# --- Classement d'un article par sujet ---

LEFASO = "https://lefaso.net/spip.php?article1"
ALLAFRICA = "https://fr.allafrica.com/stories/1.html"


def test_mentioned_pays():
    text = "Les Burkinabè et les Maliennes se retrouvent à Abidjan."
    assert utils.mentioned_pays(text) == {
        "burkina faso",
        "mali",
        "cote d'ivoire",
    }
    # Apostrophe typographique, et pas de faux positif sur "Somalie".
    assert utils.mentioned_pays("La Côte d’Ivoire") == {"cote d'ivoire"}
    assert utils.mentioned_pays("La Somalie et le site Burkina24") == set()


def test_classify_article_local_reste_dans_son_pays():
    assert (
        utils.classify_pays(
            "Burkina Faso",
            "Rentrée scolaire : les élèves reprennent",
            "À Ouagadougou, la rentrée s'est bien passée.",
            LEFASO,
        )
        == "burkina faso"
    )


def test_classify_sujet_etranger_dans_un_journal_national():
    assert (
        utils.classify_pays(
            "Burkina Faso",
            "FIFA : Infantino sous pression",
            "Réunion de crise à Rabat autour du président de la FIFA.",
            LEFASO,
        )
        == "monde"
    )


def test_classify_titre_nommant_un_autre_pays():
    assert (
        utils.classify_pays(
            "Burkina Faso",
            "Mali : les FAMa repoussent une attaque",
            "Selon l'état-major, relayé à Ouagadougou...",
            LEFASO,
        )
        == "mali"
    )
    assert (
        utils.classify_pays(
            "Monde", "Burkina : quel bilan ?", "Analyse.", "https://f24.com/a"
        )
        == "burkina faso"
    )


def test_classify_titre_nommant_deux_pays_garde_le_flux():
    assert (
        utils.classify_pays(
            "Burkina Faso",
            "Le Sénégal bat le Burkina en amical",
            "Match disputé à Dakar.",
            LEFASO,
        )
        == "burkina faso"
    )


def test_classify_fait_confiance_aux_flux_par_sujet():
    # Article local d'AllAfrica Sénégal qui ne nomme pas le pays.
    assert (
        utils.classify_pays(
            "Senegal",
            "Kolda - 44 villages réclament l'électrification",
            "Les populations ont marché ce lundi.",
            ALLAFRICA,
        )
        == "senegal"
    )


def test_classify_flux_monde_sans_pays_reste_monde():
    assert (
        utils.classify_pays(
            "Monde", "DeepSeek lance un modèle", "Texte.", "https://clubic.com/a"
        )
        == "monde"
    )
