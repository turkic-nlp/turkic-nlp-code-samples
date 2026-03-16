"""
Generate per-language TurkicNLP Jupyter notebooks.
Run: python generate_notebooks.py
"""

import json, os, textwrap

NOTEBOOKS_DIR = "notebooks"
os.makedirs(NOTEBOOKS_DIR, exist_ok=True)


def nb(cells):
    """Return a minimal valid notebook dict."""
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.9.0"},
        },
        "cells": cells,
    }


def md(text):
    return {
        "cell_type": "markdown",
        "id": f"md-{abs(hash(text)) % 10**8:08d}",
        "metadata": {},
        "source": text.strip(),
    }


def code(src):
    return {
        "cell_type": "code",
        "id": f"code-{abs(hash(src)) % 10**8:08d}",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src.strip(),
    }


# ---------------------------------------------------------------------------
# Install cell (common to all notebooks)
# ---------------------------------------------------------------------------
INSTALL = code("""\
# Install TurkicNLP
# pip install turkicnlp          # core (tokenization, transliteration)
# pip install "turkicnlp[stanza]"  # adds POS, lemma, depparse, NER
# pip install "turkicnlp[nllb]"    # adds cross-lingual embeddings + translation
# pip install "turkicnlp[all]"     # all optional dependencies
""")

IMPORT = code("""\
import turkicnlp
from turkicnlp import Pipeline
""")

# ---------------------------------------------------------------------------
# Per-language notebook definitions
# ---------------------------------------------------------------------------
# Structure: (filename_stem, display_name, iso, notebook_cells_fn)
# ---------------------------------------------------------------------------


def turkish():
    cells = [
        md("# Turkish (tur) — Full NLP Pipeline\n\n"
           "Turkish is the most resource-rich Turkic language in TurkicNLP, supporting "
           "the full processing pipeline: tokenization, multi-word token expansion, "
           "morphological analysis (Apertium FST, Production quality), POS tagging, "
           "lemmatization, dependency parsing (7 UD treebank variants), named entity "
           "recognition, sentence embeddings, and machine translation via NLLB-200."),
        INSTALL,
        IMPORT,
        md("## 1. Download Models"),
        code("""\
# Download all Turkish models (Stanza + Apertium FST)
turkicnlp.download("tur")
"""),
        md("## 2. Tokenisation and Multi-Word Token Expansion"),
        code("""\
# Rule-based tokeniser (default) with MWT expansion
nlp_tok = Pipeline("tur", processors=["tokenize", "mwt"])

doc = nlp_tok("İstanbul'a gitmek istiyorum.")
for sent in doc.sentences:
    print("Tokens:", [tok.text for tok in sent.tokens])
    print("Words: ", [w.text for w in sent.words])
"""),
        md("## 3. Morphological Analysis (Apertium FST)"),
        code("""\
turkicnlp.download("tur", processors=["tokenize", "morph"])

nlp_morph = Pipeline(
    "tur",
    processors=["tokenize", "morph"],
    morph_backend="apertium",
)

doc = nlp_morph("Türkiye'nin başkenti Ankara'dır.")
for word in doc.words:
    print(f"{word.text:<20} lemma={word.lemma:<15} feats={word.feats}")
"""),
        md("## 4. POS Tagging, Lemmatisation, and Dependency Parsing\n\n"
           "Seven UD treebank variants are available for Turkish. "
           "The default is `IMST` (general domain). Others: `BOUN`, `FrameNet`, "
           "`KeNet`, `ATIS` (aviation), `Penn`, `Tourism`."),
        code("""\
# Default IMST treebank
nlp_parse = Pipeline("tur", processors=["tokenize", "pos", "lemma", "depparse"])

doc = nlp_parse("Ahmet bugün İstanbul'a gitti.")
print(f"{'Word':<15} {'UPOS':<8} {'Lemma':<15} {'Head':<5} {'Deprel'}")
print("-" * 55)
for w in doc.words:
    print(f"{w.text:<15} {w.upos:<8} {w.lemma:<15} {w.head!s:<5} {w.deprel}")
"""),
        code("""\
# Tourism domain treebank
nlp_tour = Pipeline(
    "tur",
    processors=["tokenize", "pos", "lemma", "depparse"],
    pos_treebank="Tourism",
)
doc = nlp_tour("Otelin konumu mükemmeldi.")
for w in doc.words:
    print(f"{w.text:<20} {w.upos:<8} {w.lemma}")
"""),
        md("## 5. Named Entity Recognition (NER)\n\n"
           "Turkish NER trained on the Starlang corpus, recognising "
           "`PER`, `ORG`, `LOC`, `MISC`."),
        code("""\
nlp_ner = Pipeline("tur", processors=["tokenize", "pos", "lemma", "ner"])

doc = nlp_ner("Ahmet Çelik, Türk Hava Yolları'nda çalışıyor.")
for ent in doc.entities:
    print(f"  {ent.text!r:<30} type={ent.type}")
"""),
        md("## 6. Full Pipeline with CoNLL-U Export"),
        code("""\
nlp_full = Pipeline(
    "tur",
    processors=["tokenize", "mwt", "pos", "lemma", "depparse", "ner"],
)

doc = nlp_full("Mehmet Yılmaz, Ankara'daki bir şirkette müdür olarak çalışmaktadır.")

# Word-level annotations
for w in doc.words:
    print(
        f"{w.text:<20} upos={w.upos:<8} "
        f"lemma={w.lemma:<15} ner={w.ner:<8} dep={w.deprel}"
    )

# Named entity spans
print("\\nEntities:")
for ent in doc.entities:
    print(f"  {ent.text!r} -> {ent.type}")

# CoNLL-U export
print("\\nCoNLL-U:")
print(doc.to_conllu())
"""),
        md("## 7. Sentence Embeddings and Semantic Similarity (NLLB-200)"),
        code("""\
import math

turkicnlp.download("tur", processors=["embeddings"])

embed = Pipeline("tur", processors=["embeddings"])

s1 = "Bugün hava çok güzel ve parkta yürüyüş yaptım."
s2 = "Parkta yürüyüş yapmak bugün çok keyifliydi."
s3 = "Matematik çok zor bir derstir."

d1, d2, d3 = embed(s1), embed(s2), embed(s3)


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x**2 for x in a))
    norm_b = math.sqrt(sum(y**2 for y in b))
    return dot / (norm_a * norm_b)


print(f"sim(s1, s2) = {cosine(d1.embedding, d2.embedding):.4f}  (same topic)")
print(f"sim(s1, s3) = {cosine(d1.embedding, d3.embedding):.4f}  (different topic)")
"""),
        md("## 8. Machine Translation (Turkish → English / Kazakh)"),
        code("""\
turkicnlp.download("tur", processors=["translate"])

# Turkish -> English
nlp_en = Pipeline(
    "tur", processors=["translate"], translate_tgt_lang="eng_Latn"
)
doc = nlp_en("Türkiye, zengin bir tarihe ve kültüre sahip bir ülkedir.")
print("EN:", doc.translation)

# Turkish -> Kazakh
nlp_kk = Pipeline(
    "tur", processors=["translate"], translate_tgt_lang="kaz_Cyrl"
)
doc = nlp_kk("Merhaba, nasılsınız?")
print("KK:", doc.translation)
"""),
        md("## 9. Discover Available Processors"),
        code("""\
print("Languages:", turkicnlp.list_languages()[:6], "...")
print("Turkish processors:", turkicnlp.list_processors("tur"))
"""),
    ]
    return cells


def kazakh():
    cells = [
        md("# Kazakh (kaz) — Full NLP Pipeline with Cyrillic/Latin Support\n\n"
           "Kazakh is the second most resource-rich Turkic language in TurkicNLP. "
           "It supports the full pipeline in both Cyrillic and Latin scripts, "
           "with Production-quality Apertium FST morphology, Stanza neural parsing "
           "(KTB treebank), 25-class NER via KazNERD, and NLLB-200 embeddings/translation."),
        INSTALL,
        IMPORT,
        md("## 1. Download Models"),
        code("""\
turkicnlp.download("kaz")
"""),
        md("## 2. Script Detection and Cyrillic ↔ Latin Transliteration"),
        code("""\
from turkicnlp.scripts import Script
from turkicnlp.scripts.detector import detect_script
from turkicnlp.scripts.transliterator import Transliterator

# Kazakh Cyrillic
cyrl = "Мен Алматыда тұрамын."
print("Detected:", detect_script(cyrl))  # Script.CYRILLIC

# Cyrillic -> Latin (2021 official Kazakh Latin alphabet)
t_fwd = Transliterator("kaz", source=Script.CYRILLIC, target=Script.LATIN)
latin = t_fwd.transliterate(cyrl)
print("Latin:", latin)

# Latin -> Cyrillic
t_rev = Transliterator("kaz", source=Script.LATIN, target=Script.CYRILLIC)
print("Back:", t_rev.transliterate(latin))
"""),
        md("## 3. Morphological Analysis (Apertium FST — Production quality)"),
        code("""\
nlp_morph = Pipeline(
    "kaz",
    processors=["tokenize", "morph"],
    morph_backend="apertium",
)

# Input in Latin
doc = nlp_morph("Men mektepke baramin.")
print("=== Latin input ===")
for w in doc.words:
    print(f"{w.text:<18} lemma={w.lemma:<12} feats={w.feats}")
"""),
        code("""\
# Input in Cyrillic — pipeline auto-detects and processes transparently
nlp_cyrl = Pipeline(
    "kaz",
    processors=["tokenize", "morph"],
    morph_backend="apertium",
    script="Cyrl",
)
doc = nlp_cyrl("Мен мектепке барамын.")
print("=== Cyrillic input ===")
for w in doc.words:
    print(f"{w.text:<18} lemma={w.lemma:<12} feats={w.feats}")
"""),
        md("## 4. POS Tagging, Lemmatisation, and Dependency Parsing (Stanza / KTB)"),
        code("""\
nlp_parse = Pipeline(
    "kaz",
    processors=["tokenize", "pos", "lemma", "depparse"],
    script="Cyrl",
)

doc = nlp_parse("Нұр-Сұлтан Қазақстанның астанасы болып табылады.")
print(f"{'Word':<20} {'UPOS':<8} {'Lemma':<20} {'Head':<5} {'Deprel'}")
print("-" * 65)
for w in doc.words:
    print(f"{w.text:<20} {w.upos:<8} {w.lemma:<20} {w.head!s:<5} {w.deprel}")
"""),
        md("## 5. Named Entity Recognition — 25 Classes (KazNERD)"),
        code("""\
nlp_ner = Pipeline(
    "kaz",
    processors=["tokenize", "pos", "lemma", "ner"],
    script="Cyrl",
)

# "Erzhan went to Astana today."
doc = nlp_ner("Ержан бүгін Астанаға барды.")
for ent in doc.entities:
    print(f"  {ent.text!r:<25} type={ent.type}")
"""),
        code("""\
# Richer sentence with multiple entity types
doc2 = nlp_ner(
    "Қазақстан Республикасының Президенті Қасым-Жомарт Тоқаев "
    "Ақорда сарайында кездесу өткізді."
)
for ent in doc2.entities:
    print(f"  {ent.text!r:<35} type={ent.type}")
"""),
        md("## 6. Full Pipeline with CoNLL-U Export"),
        code("""\
nlp_full = Pipeline(
    "kaz",
    processors=["tokenize", "pos", "lemma", "depparse", "ner"],
    script="Cyrl",
)

doc = nlp_full("Алматы — Қазақстандағы ең үлкен қала.")

for w in doc.words:
    print(
        f"{w.text:<20} upos={w.upos:<8} "
        f"lemma={w.lemma:<20} ner={w.ner:<8} dep={w.deprel}"
    )
print("\\nCoNLL-U:\\n", doc.to_conllu())
"""),
        md("## 7. Sentence Embeddings and Translation"),
        code("""\
import math

turkicnlp.download("kaz", processors=["embeddings", "translate"])

embed = Pipeline("kaz", processors=["embeddings"])
trans = Pipeline("kaz", processors=["translate"], translate_tgt_lang="eng_Latn")

s1 = "Бүгін ауа райы өте жақсы."
s2 = "Ауа-райы бүгін керемет."
d1 = embed(s1)
d2 = embed(s2)

dot = sum(x * y for x, y in zip(d1.embedding, d2.embedding))
norm = (
    math.sqrt(sum(x**2 for x in d1.embedding))
    * math.sqrt(sum(y**2 for y in d2.embedding))
)
print(f"Cosine similarity: {dot/norm:.4f}")
print("Translation:", trans(s1).translation)
"""),
    ]
    return cells


def kyrgyz():
    cells = [
        md("# Kyrgyz (kir) — Full Neural Pipeline with Cyrillic/Latin Support\n\n"
           "Kyrgyz supports the full Stanza neural pipeline via the KTMU UD treebank "
           "(POS, lemma, depparse), Apertium FST morphology (Stable quality), "
           "and bidirectional Cyrillic↔Latin transliteration. "
           "NLLB-200 provides embeddings and translation."),
        INSTALL,
        IMPORT,
        md("## 1. Download Models"),
        code("turkicnlp.download('kir')"),
        md("## 2. Script Detection and Cyrillic ↔ Latin Transliteration"),
        code("""\
from turkicnlp.scripts import Script
from turkicnlp.scripts.detector import detect_script
from turkicnlp.scripts.transliterator import Transliterator

cyrl = "Бишкек Кыргызстандын башкаласы."
print("Detected:", detect_script(cyrl))

t = Transliterator("kir", source=Script.CYRILLIC, target=Script.LATIN)
latin = t.transliterate(cyrl)
print("Latin:", latin)
"""),
        md("## 3. Morphological Analysis (Apertium FST)"),
        code("""\
nlp_morph = Pipeline(
    "kir",
    processors=["tokenize", "morph"],
    morph_backend="apertium",
    script="Cyrl",
)
doc = nlp_morph("Мен мектепке барам.")
for w in doc.words:
    print(f"{w.text:<18} lemma={w.lemma:<12} feats={w.feats}")
"""),
        md("## 4. POS Tagging, Lemmatisation, and Dependency Parsing (KTMU treebank)"),
        code("""\
nlp_parse = Pipeline(
    "kir",
    processors=["tokenize", "pos", "lemma", "depparse"],
    script="Cyrl",
)

doc = nlp_parse("Ала-Тоо тоолор Кыргызстанда жайгашкан.")
print(f"{'Word':<20} {'UPOS':<8} {'Lemma':<20} {'Deprel'}")
print("-" * 60)
for w in doc.words:
    print(f"{w.text:<20} {w.upos:<8} {w.lemma:<20} {w.deprel}")
"""),
        md("## 5. Full Pipeline with CoNLL-U Export"),
        code("""\
nlp_full = Pipeline(
    "kir",
    processors=["tokenize", "morph", "pos", "lemma", "depparse"],
    morph_backend="apertium",
    script="Cyrl",
)
doc = nlp_full("Кыргыз тили Кыргызстандын расмий тили болуп эсептелет.")
print(doc.to_conllu())
"""),
        md("## 6. Embeddings and Translation"),
        code("""\
turkicnlp.download("kir", processors=["translate"])
trans = Pipeline("kir", processors=["translate"], translate_tgt_lang="eng_Latn")
doc = trans("Кыргызстан — Борбордук Азиядагы мамлекет.")
print("EN:", doc.translation)
"""),
    ]
    return cells


def uyghur():
    cells = [
        md("# Uyghur (uig) — Arabic Script NLP Pipeline\n\n"
           "Uyghur is written in Perso-Arabic script (right-to-left). "
           "TurkicNLP provides Arabic-script tokenisation, Apertium FST morphology "
           "(Beta quality), full Stanza neural pipeline via the UDT treebank, "
           "and bidirectional Arabic↔Latin (ULY) transliteration."),
        INSTALL,
        IMPORT,
        md("## 1. Download Models"),
        code("turkicnlp.download('uig')"),
        md("## 2. Script Detection — Arabic Script"),
        code("""\
from turkicnlp.scripts.detector import detect_script

# Uyghur in Perso-Arabic script
arab_text = "مەن مەكتەپكە بارىمەن."
print("Detected:", detect_script(arab_text))
"""),
        md("## 3. Arabic ↔ Latin (ULY) Transliteration"),
        code("""\
from turkicnlp.scripts import Script
from turkicnlp.scripts.transliterator import Transliterator

arab = "مەن مەكتەپكە بارىمەن."

# Arabic -> Latin (ULY — Uyghur Latin Yéziqi standard)
t = Transliterator("uig", source=Script.ARABIC, target=Script.LATIN)
uly = t.transliterate(arab)
print("ULY:", uly)

# Latin -> Arabic
t_back = Transliterator("uig", source=Script.LATIN, target=Script.ARABIC)
print("Back:", t_back.transliterate(uly))
"""),
        md("## 4. Morphological Analysis (Apertium FST — Beta)"),
        code("""\
nlp_morph = Pipeline(
    "uig",
    processors=["tokenize", "morph"],
    morph_backend="apertium",
    script="Arab",
)
doc = nlp_morph("مەن مەكتەپكە بارىمەن.")
for w in doc.words:
    print(f"{w.text:<20} lemma={w.lemma:<15} feats={w.feats}")
"""),
        md("## 5. POS Tagging, Lemmatisation, and Dependency Parsing (UDT treebank)"),
        code("""\
nlp_parse = Pipeline(
    "uig",
    processors=["tokenize", "pos", "lemma", "depparse"],
    script="Arab",
)
doc = nlp_parse("ئۈرۈمچى شىنجاڭنىڭ پايتەختى.")
print(f"{'Word':<20} {'UPOS':<8} {'Lemma':<15} {'Deprel'}")
for w in doc.words:
    print(f"{w.text:<20} {w.upos:<8} {w.lemma:<15} {w.deprel}")
print("\\nCoNLL-U:\\n", doc.to_conllu())
"""),
        md("## 6. Translation"),
        code("""\
turkicnlp.download("uig", processors=["translate"])
trans = Pipeline("uig", processors=["translate"], translate_tgt_lang="eng_Latn")
doc = trans("شىنجاڭ ئۇيغۇر ئاپتونوم رايونى.")
print("EN:", doc.translation)
"""),
    ]
    return cells


def uzbek():
    cells = [
        md("# Uzbek (uzb) — Full NLP Pipeline with Custom Stanza Models\n\n"
           "Uzbek uses the Latin script since 1995 (official). The Cyrillic variant "
           "remains in wide use. TurkicNLP provides Apertium FST morphology (Stable), "
           "custom-trained Stanza neural models for POS tagging, lemmatisation, and "
           "dependency parsing, bidirectional Cyrillic↔Latin transliteration, "
           "and NLLB-200 embeddings/translation."),
        INSTALL,
        IMPORT,
        md("## 1. Download Models"),
        code("turkicnlp.download('uzb')"),
        md("## 2. Cyrillic ↔ Latin Transliteration (1995 official standard)"),
        code("""\
from turkicnlp.scripts import Script
from turkicnlp.scripts.detector import detect_script
from turkicnlp.scripts.transliterator import Transliterator

cyrl = "Мен мактабга бораман."
print("Detected:", detect_script(cyrl))

t = Transliterator("uzb", source=Script.CYRILLIC, target=Script.LATIN)
latin = t.transliterate(cyrl)
print("Latin:", latin)

t_back = Transliterator("uzb", source=Script.LATIN, target=Script.CYRILLIC)
print("Back:", t_back.transliterate(latin))
"""),
        md("## 3. Morphological Analysis (Apertium FST — Stable)"),
        code("""\
nlp = Pipeline(
    "uzb",
    processors=["tokenize", "morph"],
    morph_backend="apertium",
)

# Latin input
doc = nlp("Men maktabga boraman.")
for w in doc.words:
    print(f"{w.text:<18} lemma={w.lemma:<12} feats={w.feats}")
"""),
        code("""\
# Cyrillic input with auto-detection
nlp_cyrl = Pipeline(
    "uzb",
    processors=["tokenize", "morph"],
    morph_backend="apertium",
    script="Cyrl",
)
doc = nlp_cyrl("Мен мактабга бораман.")
for w in doc.words:
    print(f"{w.text:<18} lemma={w.lemma:<12} feats={w.feats}")
"""),
        md("## 4. POS Tagging, Lemmatisation, and Dependency Parsing (Custom Stanza)\n\n"
           "TurkicNLP includes custom-trained Stanza models for Uzbek, providing "
           "POS tagging, lemmatisation, and dependency parsing."),
        code("""\
nlp_parse = Pipeline(
    "uzb",
    processors=["tokenize", "pos", "lemma", "depparse"],
)

doc = nlp_parse("Men maktabga ketdim.")
print(f"{'Word':<20} {'UPOS':<8} {'Lemma':<20} {'Head':<5} {'Deprel'}")
print("-" * 60)
for w in doc.words:
    print(f"{w.text:<20} {w.upos:<8} {w.lemma:<20} {w.head!s:<5} {w.deprel}")
"""),
        md("## 5. Full Pipeline with CoNLL-U Export"),
        code("""\
nlp_full = Pipeline(
    "uzb",
    processors=["tokenize", "morph", "pos", "lemma", "depparse"],
    morph_backend="apertium",
)
doc = nlp_full("O'zbekiston Markaziy Osiyodagi eng yirik davlatlardan biri.")
print(doc.to_conllu())
"""),
        md("## 6. Translation via NLLB-200"),
        code("""\
turkicnlp.download("uzb", processors=["translate"])
trans = Pipeline("uzb", processors=["translate"], translate_tgt_lang="eng_Latn")
doc = trans("O'zbekiston Markaziy Osiyodagi davlat.")
print("EN:", doc.translation)
"""),
    ]
    return cells


def azerbaijani():
    cells = [
        md("# Azerbaijani (aze) — Full NLP Pipeline with Custom Stanza Models\n\n"
           "Azerbaijani uses the Latin script in Azerbaijan (official since 1991) "
           "and the Cyrillic script in Russia. TurkicNLP provides Apertium FST "
           "morphology (Stable), custom-trained Stanza neural models for POS tagging, "
           "lemmatisation, and dependency parsing, bidirectional Cyrillic↔Latin "
           "transliteration, and NLLB-200 embeddings/translation."),
        INSTALL,
        IMPORT,
        md("## 1. Download Models"),
        code("turkicnlp.download('aze')"),
        md("## 2. Cyrillic ↔ Latin Transliteration (1991 standard)"),
        code("""\
from turkicnlp.scripts import Script
from turkicnlp.scripts.transliterator import Transliterator

cyrl = "Мən məktəbə gedirəm."
t = Transliterator("aze", source=Script.CYRILLIC, target=Script.LATIN)
print("Latin:", t.transliterate(cyrl))
"""),
        md("## 3. Morphological Analysis (Apertium FST — Stable)"),
        code("""\
nlp = Pipeline(
    "aze",
    processors=["tokenize", "morph"],
    morph_backend="apertium",
)
doc = nlp("Mən məktəbə gedirəm.")
for w in doc.words:
    print(f"{w.text:<20} lemma={w.lemma:<15} feats={w.feats}")
"""),
        md("## 4. POS Tagging, Lemmatisation, and Dependency Parsing (Custom Stanza)\n\n"
           "TurkicNLP includes custom-trained Stanza models for Azerbaijani, providing "
           "POS tagging, lemmatisation, and dependency parsing."),
        code("""\
nlp_parse = Pipeline(
    "aze",
    processors=["tokenize", "pos", "lemma", "depparse"],
)

doc = nlp_parse("Bakı Azərbaycanın paytaxtıdır.")
print(f"{'Word':<20} {'UPOS':<8} {'Lemma':<20} {'Head':<5} {'Deprel'}")
print("-" * 60)
for w in doc.words:
    print(f"{w.text:<20} {w.upos:<8} {w.lemma:<20} {w.head!s:<5} {w.deprel}")
"""),
        md("## 5. Full Pipeline with CoNLL-U Export"),
        code("""\
nlp_full = Pipeline(
    "aze",
    processors=["tokenize", "morph", "pos", "lemma", "depparse"],
    morph_backend="apertium",
)
doc = nlp_full("Azərbaycan Cənubi Qafqazda yerləşən bir dövlətdir.")
print(doc.to_conllu())
"""),
        md("## 6. Translation"),
        code("""\
turkicnlp.download("aze", processors=["translate"])
trans = Pipeline("aze", processors=["translate"], translate_tgt_lang="eng_Latn")
doc = trans("Azərbaycan Cənubi Qafqazda yerləşən bir dövlətdir.")
print("EN:", doc.translation)
"""),
    ]
    return cells


def tatar():
    cells = [
        md("# Tatar (tat) — Full NLP Pipeline with Custom Stanza Models\n\n"
           "Tatar is written in Cyrillic (primary). The Zamanälif Latin alphabet "
           "provides an alternative script. TurkicNLP offers Production-quality "
           "Apertium FST morphology, custom-trained Stanza neural models for POS "
           "tagging, lemmatisation, and dependency parsing, and bidirectional "
           "Cyrillic↔Latin (Zamanälif) transliteration."),
        INSTALL,
        IMPORT,
        md("## 1. Download Models"),
        code("turkicnlp.download('tat')"),
        md("## 2. Cyrillic ↔ Latin Transliteration (Zamanälif)"),
        code("""\
from turkicnlp.scripts import Script
from turkicnlp.scripts.transliterator import Transliterator

cyrl = "Мин мәктәпкә барам."
t = Transliterator("tat", source=Script.CYRILLIC, target=Script.LATIN)
zamanelif = t.transliterate(cyrl)
print("Zamanälif:", zamanelif)

t_back = Transliterator("tat", source=Script.LATIN, target=Script.CYRILLIC)
print("Back:", t_back.transliterate(zamanelif))
"""),
        md("## 3. Morphological Analysis (Apertium FST — Production quality)"),
        code("""\
nlp = Pipeline(
    "tat",
    processors=["tokenize", "morph"],
    morph_backend="apertium",
    script="Cyrl",
)
doc = nlp("Мин мәктәпкә барам.")
for w in doc.words:
    print(f"{w.text:<18} lemma={w.lemma:<12} feats={w.feats}")
"""),
        md("## 4. POS Tagging, Lemmatisation, and Dependency Parsing (Custom Stanza)\n\n"
           "TurkicNLP includes custom-trained Stanza models for Tatar, providing "
           "POS tagging, lemmatisation, and dependency parsing."),
        code("""\
nlp_parse = Pipeline(
    "tat",
    processors=["tokenize", "pos", "lemma", "depparse"],
    script="Cyrl",
)

doc = nlp_parse("Бер журнал бу ай санында аның тормышын микроскоп астына ала.")
print(f"{'Word':<20} {'UPOS':<8} {'Lemma':<20} {'Head':<5} {'Deprel'}")
print("-" * 60)
for w in doc.words:
    print(f"{w.text:<20} {w.upos:<8} {w.lemma:<20} {w.head!s:<5} {w.deprel}")
"""),
        md("## 5. Full Pipeline with CoNLL-U Export"),
        code("""\
nlp_full = Pipeline(
    "tat",
    processors=["tokenize", "morph", "pos", "lemma", "depparse"],
    morph_backend="apertium",
    script="Cyrl",
)
doc = nlp_full("Татарстан Россия Федерациясе составындагы республика.")
print(doc.to_conllu())
"""),
        md("## 6. Translation"),
        code("""\
turkicnlp.download("tat", processors=["translate"])
trans = Pipeline("tat", processors=["translate"], translate_tgt_lang="rus_Cyrl")
doc = trans("Татарстан Россия Федерациясе составындагы республика.")
print("RU:", doc.translation)
"""),
    ]
    return cells


def ottoman():
    cells = [
        md("# Ottoman Turkish (ota) — Historical Language Pipeline\n\n"
           "Ottoman Turkish (historical, written in Perso-Arabic script) is supported "
           "via a dedicated Stanza model trained on the BOUN UD treebank. "
           "TurkicNLP provides Arabic-script tokenisation, neural POS tagging, "
           "lemmatisation, dependency parsing, and academic Latin transcription."),
        INSTALL,
        IMPORT,
        md("## 1. Download Models"),
        code("turkicnlp.download('ota')"),
        md("## 2. Tokenisation (Arabic Script)"),
        code("""\
nlp_tok = Pipeline("ota", processors=["tokenize"], script="Arab")
doc = nlp_tok("پاشا سرایا گلدی.")  # "The Pasha came to the palace."
for sent in doc.sentences:
    print("Tokens:", [t.text for t in sent.tokens])
"""),
        md("## 3. POS Tagging, Lemmatisation, and Dependency Parsing (BOUN treebank)"),
        code("""\
nlp_parse = Pipeline(
    "ota",
    processors=["tokenize", "pos", "lemma", "depparse"],
    script="Arab",
)
doc = nlp_parse("پاشا سرایا گلدی.")
print(f"{'Word':<15} {'UPOS':<8} {'Lemma':<15} {'Deprel'}")
print("-" * 50)
for w in doc.words:
    print(f"{w.text:<15} {w.upos:<8} {w.lemma:<15} {w.deprel}")
print("\\nCoNLL-U:\\n", doc.to_conllu())
"""),
        md("## 4. Latin Transcription → Arabic Script"),
        code("""\
from turkicnlp.scripts import Script
from turkicnlp.scripts.transliterator import Transliterator

# Academic Latin transcription to Perso-Arabic
t = Transliterator("ota", source=Script.LATIN, target=Script.ARABIC)
arabic = t.transliterate("pasha saraya geldi")
print("Arabic:", arabic)
"""),
        md("## 5. Translation"),
        code("""\
turkicnlp.download("ota", processors=["translate"])
trans = Pipeline("ota", processors=["translate"], translate_tgt_lang="eng_Latn")
doc = trans("پاشا سرایا گلدی.")
print("EN:", doc.translation)
"""),
    ]
    return cells


def bashkir():
    cells = [
        md("# Bashkir (bak) — Full NLP Pipeline with Custom Stanza Models\n\n"
           "Bashkir (Kipchak branch) is written in Cyrillic. "
           "TurkicNLP provides Beta-quality Apertium FST morphological analysis, "
           "custom-trained Stanza neural models for POS tagging, lemmatisation, "
           "and dependency parsing, and NLLB-200 embeddings and translation."),
        INSTALL,
        IMPORT,
        md("## 1. Download Models"),
        code("turkicnlp.download('bak')"),
        md("## 2. Tokenisation"),
        code("""\
nlp_tok = Pipeline("bak", processors=["tokenize"])
doc = nlp_tok("Мин мәктәпкә барам.")
for sent in doc.sentences:
    print([w.text for w in sent.words])
"""),
        md("## 3. Morphological Analysis (Apertium FST — Beta)"),
        code("""\
nlp = Pipeline(
    "bak",
    processors=["tokenize", "morph"],
    morph_backend="apertium",
)
doc = nlp("Мин мәктәпкә барам.")
for w in doc.words:
    print(f"{w.text:<18} lemma={w.lemma:<12} feats={w.feats}")
"""),
        md("## 4. POS Tagging, Lemmatisation, and Dependency Parsing (Custom Stanza)\n\n"
           "TurkicNLP includes custom-trained Stanza models for Bashkir, providing "
           "POS tagging, lemmatisation, and dependency parsing."),
        code("""\
nlp_parse = Pipeline(
    "bak",
    processors=["tokenize", "pos", "lemma", "depparse"],
)

doc = nlp_parse("Бер журнал был айҙың һанында уның тормошон микроскоп аҫтына ала.")
print(f"{'Word':<20} {'UPOS':<8} {'Lemma':<20} {'Head':<5} {'Deprel'}")
print("-" * 60)
for w in doc.words:
    print(f"{w.text:<20} {w.upos:<8} {w.lemma:<20} {w.head!s:<5} {w.deprel}")
"""),
        md("## 5. Full Pipeline with CoNLL-U Export"),
        code("""\
nlp_full = Pipeline(
    "bak",
    processors=["tokenize", "morph", "pos", "lemma", "depparse"],
    morph_backend="apertium",
)
doc = nlp_full("Башҡортостан — Рәсәй Федерацияһы субъекты.")
print(doc.to_conllu())
"""),
        md("## 6. Translation"),
        code("""\
turkicnlp.download("bak", processors=["translate"])
trans = Pipeline("bak", processors=["translate"], translate_tgt_lang="rus_Cyrl")
doc = trans("Башҡортостан — Рәсәй Федерацияһы субъекты.")
print("RU:", doc.translation)
"""),
    ]
    return cells


def turkmen():
    cells = [
        md("# Turkmen (tuk) — Full NLP Pipeline with Custom Stanza Models\n\n"
           "Turkmen uses the Latin script (official since 1993). "
           "TurkicNLP provides Beta-quality Apertium FST morphological analysis, "
           "custom-trained Stanza neural models for POS tagging, lemmatisation, "
           "and dependency parsing, and Cyrillic↔Latin transliteration."),
        INSTALL,
        IMPORT,
        md("## 1. Download Models"),
        code('turkicnlp.download("tuk")'),
        md("## 2. Tokenisation"),
        code("""\
nlp_tok = Pipeline("tuk", processors=["tokenize"])
doc = nlp_tok("Men mekdebe barýaryn.")
print([w.text for w in doc.words])
"""),
        md("## 3. Cyrillic ↔ Latin Transliteration (1993 standard)"),
        code("""\
from turkicnlp.scripts import Script
from turkicnlp.scripts.transliterator import Transliterator

cyrl = "Мен мекдебе барӹарын."
t = Transliterator("tuk", source=Script.CYRILLIC, target=Script.LATIN)
print("Latin:", t.transliterate(cyrl))
"""),
        md("## 4. Morphological Analysis (Apertium FST — Beta)"),
        code("""\
nlp = Pipeline(
    "tuk",
    processors=["tokenize", "morph"],
    morph_backend="apertium",
)
doc = nlp("Men mekdebe barýaryn.")
for w in doc.words:
    print(f"{w.text:<18} lemma={w.lemma:<12} feats={w.feats}")
"""),
        md("## 5. POS Tagging, Lemmatisation, and Dependency Parsing (Custom Stanza)\n\n"
           "TurkicNLP includes custom-trained Stanza models for Turkmen, providing "
           "POS tagging, lemmatisation, and dependency parsing."),
        code("""\
nlp_parse = Pipeline(
    "tuk",
    processors=["tokenize", "pos", "lemma", "depparse"],
)

doc = nlp_parse("Men mektebe gitdim we Murat bilen kitap okadym.")
print(f"{'Word':<20} {'UPOS':<8} {'Lemma':<20} {'Head':<5} {'Deprel'}")
print("-" * 60)
for w in doc.words:
    print(f"{w.text:<20} {w.upos:<8} {w.lemma:<20} {w.head!s:<5} {w.deprel}")
"""),
        md("## 6. Full Pipeline with CoNLL-U Export"),
        code("""\
nlp_full = Pipeline(
    "tuk",
    processors=["tokenize", "morph", "pos", "lemma", "depparse"],
    morph_backend="apertium",
)
doc = nlp_full("Türkmenistan Orta Aziýada ýerleşýän döwletdir.")
print(doc.to_conllu())
"""),
        md("## 7. Translation"),
        code("""\
turkicnlp.download("tuk", processors=["translate"])
trans = Pipeline("tuk", processors=["translate"], translate_tgt_lang="eng_Latn")
doc = trans("Türkmenistan Orta Aziýada ýerleşýän döwletdir.")
print("EN:", doc.translation)
"""),
    ]
    return cells


def _simple_lang(iso, name, script, script_enum, branch, morph_quality,
                 sample_cyrl, sample_latn, transliteration=None, notes=""):
    """Generate a minimal notebook for languages with tokenise+morph+translate."""
    cells = [
        md(f"# {name} ({iso}) — Tokenisation and Morphological Analysis\n\n"
           f"{name} ({branch} branch) is supported via "
           f"{'Cyrillic' if script == 'Cyrl' else 'Latin' if script == 'Latn' else script} script tokenisation "
           f"and {morph_quality}-quality Apertium FST morphological analysis. "
           f"NLLB-200 provides cross-lingual embeddings and machine translation.{' ' + notes if notes else ''}"),
        INSTALL,
        IMPORT,
        md("## 1. Download Models"),
        code(f"turkicnlp.download('{iso}')"),
        md("## 2. Tokenisation"),
        code(f"""\
nlp_tok = Pipeline("{iso}", processors=["tokenize"])
doc = nlp_tok("{sample_latn or sample_cyrl}")
print([w.text for w in doc.words])
"""),
    ]
    if transliteration:
        src_script, tgt_script, src_sample = transliteration
        cells += [
            md(f"## 3. Script Transliteration"),
            code(f"""\
from turkicnlp.scripts import Script
from turkicnlp.scripts.transliterator import Transliterator

t = Transliterator("{iso}", source=Script.{src_script}, target=Script.{tgt_script})
result = t.transliterate("{src_sample}")
print("{tgt_script}:", result)
"""),
        ]
    cells += [
        md(f"## {'4' if transliteration else '3'}. Morphological Analysis (Apertium FST — {morph_quality})"),
        code(f"""\
nlp = Pipeline(
    "{iso}",
    processors=["tokenize", "morph"],
    morph_backend="apertium",
)
doc = nlp("{sample_latn or sample_cyrl}")
for w in doc.words:
    print(f"{{w.text:<18}} lemma={{w.lemma:<12}} feats={{w.feats}}")
"""),
        md(f"## {'5' if transliteration else '4'}. Translation via NLLB-200"),
        code(f"""\
turkicnlp.download("{iso}", processors=["translate"])
trans = Pipeline("{iso}", processors=["translate"], translate_tgt_lang="eng_Latn")
doc = trans("{sample_latn or sample_cyrl}")
print("EN:", doc.translation)
"""),
    ]
    return cells


def _tokenize_only(iso, name, script, branch, sample, notes=""):
    """Notebook for languages with only tokenise + embeddings/translate."""
    return [
        md(f"# {name} ({iso}) — Tokenisation and Translation\n\n"
           f"{name} ({branch} branch, {script} script) currently has "
           f"rule-based tokenisation and prototype-quality morphological analysis. "
           f"NLLB-200 provides cross-lingual embeddings and machine translation.{' ' + notes if notes else ''}"),
        INSTALL,
        IMPORT,
        md("## 1. Tokenisation"),
        code(f"""\
turkicnlp.download("{iso}")
nlp_tok = Pipeline("{iso}", processors=["tokenize"])
doc = nlp_tok("{sample}")
print([w.text for w in doc.words])
"""),
        md("## 2. Morphological Analysis (Apertium FST — Prototype)"),
        code(f"""\
nlp = Pipeline(
    "{iso}",
    processors=["tokenize", "morph"],
    morph_backend="apertium",
)
doc = nlp("{sample}")
for w in doc.words:
    print(f"{{w.text:<18}} lemma={{w.lemma}} feats={{w.feats}}")
"""),
        md("## 3. Translation via NLLB-200"),
        code(f"""\
turkicnlp.download("{iso}", processors=["translate"])
trans = Pipeline("{iso}", processors=["translate"], translate_tgt_lang="eng_Latn")
doc = trans("{sample}")
print("EN:", doc.translation)
"""),
    ]


# ---------------------------------------------------------------------------
# Notebook 25 — Monolingual Semantic Similarity
# ---------------------------------------------------------------------------

def embeddings_monolingual():
    return [
        md("""\
# Monolingual Semantic Similarity with NLLB-200 Embeddings

NLLB-200 encodes text into a shared multilingual vector space via its encoder.
Sentences with similar meaning map to nearby vectors, regardless of surface form.
Cosine similarity between two such vectors is a reliable proxy for semantic
relatedness within a single language.

**Typical use cases:** duplicate-question detection, sentence clustering,
semantic search, paraphrase mining, retrieval-augmented generation (RAG).

This notebook uses Turkish as the primary language; the same API applies to
all 24 languages supported by TurkicNLP.\
"""),
        INSTALL,
        code("""\
import math
import turkicnlp
from turkicnlp import Pipeline

turkicnlp.download("tur", processors=["embeddings"])
embed = Pipeline("tur", processors=["embeddings"])
"""),

        md("## 1. Cosine Similarity Between Two Sentences"),
        code("""\
def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na  = math.sqrt(sum(x**2 for x in a))
    nb  = math.sqrt(sum(y**2 for y in b))
    return dot / (na * nb)

pairs = [
    ("Bugün hava çok güzel.",
     "Dışarısı oldukça sıcak ve güneşli.",
     "high – same topic, different words"),
    ("Ankara Türkiye'nin başkentidir.",
     "Türkiye'nin başkenti Ankara'dır.",
     "very high – near-paraphrase"),
    ("Çocuklar parkta oynuyor.",
     "Ekonomi kriz sürecine girdi.",
     "low – unrelated topics"),
    ("Bu film harikaydı.",
     "Bu film berbattı.",
     "low – opposite sentiment"),
]

print(f"{'Pair':<45} {'Cosine':>7}")
print("-" * 55)
for s1, s2, label in pairs:
    e1 = embed(s1).embedding
    e2 = embed(s2).embedding
    print(f"{label:<45} {cosine(e1, e2):>7.4f}")
"""),

        md("## 2. Semantic Ranking — Nearest-Neighbour Search"),
        code("""\
# Given a query, rank a corpus by similarity
query  = "Spor salonunda egzersiz yapıyorum."
corpus = [
    "Her sabah koşuya çıkıyorum.",
    "Futbol maçı saat 19:00'da başlıyor.",
    "Yemek pişirmek benim için bir hobi.",
    "Spor yapmak sağlık için çok önemlidir.",
    "Bugün hava yağmurlu.",
    "Fitness merkezi üyeliği aldım.",
    "Ekonomi haberleri hiç iç açıcı değil.",
    "Antrenmanımı tamamladım, çok yorgunum.",
]

q_emb = embed(query).embedding
scores = [(cosine(q_emb, embed(s).embedding), s) for s in corpus]
scores.sort(reverse=True)

print(f"Query: '{query}'\\n")
print(f"{'Rank':<5} {'Score':>6}  Sentence")
print("-" * 65)
for rank, (score, sent) in enumerate(scores, 1):
    print(f"{rank:<5} {score:>6.4f}  {sent}")
"""),

        md("## 3. Batch Embedding and Pairwise Similarity Matrix"),
        code("""\
sentences = [
    "İstanbul Türkiye'nin en kalabalık şehridir.",
    "Türkiye'de en çok insan İstanbul'da yaşar.",
    "Ankara başkent olarak idari merkez görevini üstlenmiştir.",
    "Türk mutfağı dünya genelinde çok beğenilmektedir.",
    "Türkiye'nin büyük şehirlerinden biri olan İzmir deniz kenarındadır.",
]

embeddings = [embed(s).embedding for s in sentences]

# Print the upper-triangle of the pairwise similarity matrix
header = "".join(f"  S{i+1}" for i in range(len(sentences)))
print(f"{'':>50}{header}")
for i, (s, e_i) in enumerate(zip(sentences, embeddings)):
    label = f"S{i+1}: {s[:42]:<42}"
    row   = "  " * i + "  --"
    for j in range(i + 1, len(sentences)):
        row += f" {cosine(e_i, embeddings[j]):>5.3f}"
    print(f"{label}  {row}")
"""),

        md("## 4. Embeddings for Other Turkic Languages\n\n"
           "The same `embeddings` processor is available for all languages "
           "supported by TurkicNLP. Simply change the language code."),
        code("""\
for lang, text in [
    ("kaz", "Бүгін ауа райы өте жақсы."),    # Kazakh
    ("kir", "Бүгүн аба ырайы абдан жакшы."),  # Kyrgyz
    ("uzb", "Bugun ob-havo juda yaxshi."),      # Uzbek
]:
    turkicnlp.download(lang, processors=["embeddings"])
    pipe = Pipeline(lang, processors=["embeddings"])
    emb  = pipe(text).embedding
    print(f"{lang}: dim={len(emb)},  norm={math.sqrt(sum(x**2 for x in emb)):.4f}")
"""),
    ]


# ---------------------------------------------------------------------------
# Notebook 26 — Cross-lingual Semantic Similarity
# ---------------------------------------------------------------------------

def embeddings_crosslingual():
    return [
        md("""\
# Cross-lingual Semantic Similarity across Turkic Languages

Because NLLB-200 is trained on 200 languages simultaneously, its encoder
projects sentences from different languages into a **shared** semantic space.
A sentence and its translation should therefore have a high cosine similarity
even though they are in different languages and different scripts.

This property is extremely useful for:
- Aligning parallel corpora without gold labels
- Multilingual information retrieval (query in one language, results in another)
- Assessing translation quality
- Building language-agnostic NLP models

This notebook takes a Turkish seed sentence, translates it to eight other
Turkic languages via TurkicNLP's NLLB-200 translation backend, then computes
a full 9×9 cross-lingual similarity matrix.\
"""),
        INSTALL,
        code("""\
import math
import turkicnlp
from turkicnlp import Pipeline

# FLORES-200 / NLLB language codes for nine Turkic languages
LANGS = {
    "tur": ("tur_Latn", "Turkish"),
    "kaz": ("kaz_Cyrl", "Kazakh"),
    "uzb": ("uzn_Latn", "Uzbek"),
    "aze": ("azj_Latn", "Azerbaijani"),
    "kir": ("kir_Cyrl", "Kyrgyz"),
    "tat": ("tat_Cyrl", "Tatar"),
    "tuk": ("tuk_Latn", "Turkmen"),
    "uig": ("uig_Arab", "Uyghur"),
    "bak": ("bak_Cyrl", "Bashkir"),
}

for lang in LANGS:
    turkicnlp.download(lang, processors=["embeddings", "translate"])
"""),

        md("## 1. Translate a Seed Sentence to All Languages"),
        code("""\
SEED = "Yapay zeka teknolojisi dünyayı hızla değiştiriyor."  # Turkish

translations = {"tur": SEED}

# Translate Turkish -> every other language
for lang, (nllb_code, label) in LANGS.items():
    if lang == "tur":
        continue
    trans = Pipeline("tur", processors=["translate"],
                     translate_tgt_lang=nllb_code)
    translations[lang] = trans(SEED).translation

print("Translations:")
for lang, text in translations.items():
    print(f"  [{LANGS[lang][1]:<15}] {text}")
"""),

        md("## 2. Compute Cross-lingual Embedding Similarity Matrix"),
        code("""\
def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x**2 for x in a)) *
                  math.sqrt(sum(y**2 for y in b)))

# Get one embedding per language
lang_codes = list(LANGS.keys())
embs = {}
for lang in lang_codes:
    pipe = Pipeline(lang, processors=["embeddings"])
    embs[lang] = pipe(translations[lang]).embedding

# Print similarity matrix
header = "".join(f" {LANGS[l][1][:6]:>8}" for l in lang_codes)
print(f"{'':>12}{header}")
for li in lang_codes:
    row = f"{LANGS[li][1][:12]:<12}"
    for lj in lang_codes:
        row += f" {cosine(embs[li], embs[lj]):>8.4f}"
    print(row)
"""),

        md("## 3. Heatmap Visualisation\n\n"
           "Install `matplotlib` to display the similarity matrix as a colour heatmap."),
        code("""\
try:
    import matplotlib.pyplot as plt
    import numpy as np

    labels = [LANGS[l][1] for l in lang_codes]
    matrix = np.array([[cosine(embs[li], embs[lj])
                        for lj in lang_codes]
                       for li in lang_codes])

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(matrix, vmin=0.5, vmax=1.0, cmap="YlOrRd")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center",
                    fontsize=8, color="black" if matrix[i,j] < 0.85 else "white")
    plt.colorbar(im, ax=ax, label="Cosine similarity")
    ax.set_title("Cross-lingual similarity: same sentence in 9 Turkic languages")
    plt.tight_layout()
    plt.savefig("crosslingual_similarity.png", dpi=120)
    plt.show()
    print("Figure saved to crosslingual_similarity.png")
except ImportError:
    print("Install matplotlib with: pip install matplotlib numpy")
    # Print matrix values instead
    for li in lang_codes:
        print(", ".join(f"{cosine(embs[li], embs[lj]):.3f}" for lj in lang_codes))
"""),

        md("## 4. Cross-lingual Retrieval — Query in Turkish, Retrieve in Kazakh"),
        code("""\
# Suppose we have a small Kazakh document collection.
# We query in Turkish and retrieve relevant Kazakh passages.

kazakh_docs = [
    "Жасанды интеллект технологиясы дүниені жылдам өзгертіп жатыр.",
    "Бүгін ауа райы өте жақсы болды.",
    "Қазақстан орталық Азияда орналасқан мемлекет.",
    "Ғылым мен технология адамзаттың болашағын қалыптастырады.",
    "Дәрігерлер пациенттерге мейірімді қарады.",
]

kaz_embed = Pipeline("kaz", processors=["embeddings"])
tur_embed = Pipeline("tur", processors=["embeddings"])

query     = "Teknoloji ve yapay zeka insanlığı etkiliyor."  # Turkish query
q_emb     = tur_embed(query).embedding
doc_embs  = [kaz_embed(d).embedding for d in kazakh_docs]

ranked = sorted(zip([cosine(q_emb, e) for e in doc_embs], kazakh_docs),
                reverse=True)

print(f"Query (Turkish): '{query}'\\n")
print("Ranked Kazakh documents:")
for score, doc in ranked:
    print(f"  {score:.4f}  {doc}")
"""),
    ]


# ---------------------------------------------------------------------------
# Notebook 27 — Text Classification with Embeddings
# ---------------------------------------------------------------------------

def embeddings_classifier():
    return [
        md("""\
# Text Classification with NLLB-200 Embeddings

Sentence embeddings can be used as fixed-size feature vectors for any
downstream classifier. The approach has several practical advantages:

1. **No labelled data at inference time** — embeddings are obtained from a
   pre-trained model; only the classifier head needs labelled training examples.
2. **Small training sets suffice** — 20–100 labelled examples often produce
   reasonable results, because the embedding space already encodes semantics.
3. **Domain transferability** — a classifier trained on one domain (e.g., product
   reviews) can partially generalise to related domains.

This notebook demonstrates two binary classification tasks:

- **Sentiment analysis** (positive / negative) using Turkish movie and product reviews.
- **Spam / ham detection** using Turkish SMS and email text samples.

Both tasks follow the same three-step recipe:
  1. Embed labelled sentences with the `embeddings` processor.
  2. Train a logistic regression classifier on the embeddings.
  3. Evaluate and run inference on new sentences.\
"""),
        INSTALL,
        code("""\
import math, random
import turkicnlp
from turkicnlp import Pipeline

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report
    from sklearn.model_selection import train_test_split
except ImportError:
    raise SystemExit("Install scikit-learn: pip install scikit-learn")

turkicnlp.download("tur", processors=["embeddings"])
embed = Pipeline("tur", processors=["embeddings"])

def get_embedding(text):
    return embed(text).embedding
"""),

        # ---- Sentiment ----
        md("## Part A — Sentiment Analysis"),
        md("""\
### A.1 Dataset

A small inline Turkish sentiment dataset (30 sentences per class).
Positive sentences (label 1) express satisfaction, enjoyment, or approval;
negative sentences (label 0) express disappointment, frustration, or disapproval.

For production use, replace this with a full corpus such as
[SentiTurca](https://github.com/mhbasaran/SentiTurca) (350 K Turkish tweets).\
"""),
        code("""\
SENTIMENT_DATA = [
    # Positive (1)
    ("Bu film gerçekten muhteşemdi, kesinlikle tavsiye ederim.", 1),
    ("Yemek çok lezzetliydi, restoran mükemmel.", 1),
    ("Ürün beklentilerimi tamamen karşıladı, çok memnunum.", 1),
    ("Harika bir tatildi, her şey mükemmeldi.", 1),
    ("Servis çok hızlı ve personel güler yüzlüydü.", 1),
    ("Bu kitabı okumak çok keyifliydi, tavsiye ederim.", 1),
    ("Müşteri hizmetleri sorunumu hızla çözdü.", 1),
    ("Konser fantastikti, sanatçı sahneyi terk etmek istemedi.", 1),
    ("Çocuklarım bu oyuncağı çok sevdi, tekrar alacağız.", 1),
    ("Otel çok temiz ve konforluydu.", 1),
    ("Kargo çok hızlı geldi, ürün kusursuzdu.", 1),
    ("Bu deneyim hayatımın en güzel anlarından biri oldu.", 1),
    ("Fiyatına göre kalitesi oldukça iyi.", 1),
    ("Arkadaşlarıma kesinlikle öneririm.", 1),
    ("Sipariş tam açıklandığı gibi geldi, mükemmel paketleme.", 1),
    ("Uygulama son derece kullanışlı ve hızlı.", 1),
    ("Ekip çok profesyoneldi, her konuda yardımcı oldular.", 1),
    ("Ürün kalitesi fiyatı ile tam orantılı, harika.", 1),
    ("Kafeye bayıldım, atmosfer çok sıcak.", 1),
    ("Kurs içeriği çok zengin ve öğretici.", 1),
    # Negative (0)
    ("Bu film tamamen zaman kaybıydı, berbat senaryo.", 0),
    ("Yemek soğuk geldi, tadı hiç iyi değildi.", 0),
    ("Ürün resimlerdeki gibi değildi, hayal kırıklığı.", 0),
    ("Tatil mahvoldu, otel çok kötüydü.", 0),
    ("Servis saatler sürdü, kimse ilgilenmedi.", 0),
    ("Kitap çok sıkıcıydı, yarısını okuyamadım.", 0),
    ("Müşteri hizmetleri hiç yardımcı olmadı.", 0),
    ("Konser iptal edildi, para iadesi yapılmadı.", 0),
    ("Oyuncak çok hızlı bozuldu, kalitesiz.", 0),
    ("Oda çok pistı ve kokmaya başlamıştı.", 0),
    ("Kargo 3 hafta sonra geldi, ürün hasarlıydı.", 0),
    ("Bu deneyim için para harcadığıma pişmanım.", 0),
    ("Fiyatına göre kalitesi çok düşük.", 0),
    ("Kesinlikle tavsiye etmem, berbat bir hizmet.", 0),
    ("Sipariş kayboldu, kimse ilgilenmedi.", 0),
    ("Uygulama sürekli çöküyor, berbat.", 0),
    ("Ekip kaba ve umursamaz davrandı.", 0),
    ("Ürün sahte çıktı, dolandırıldım.", 0),
    ("Kafenin hijyeni çok kötüydü, yemek yiyemedim.", 0),
    ("Kurs vaat edilen içeriği sunmadı, pişmanım.", 0),
]

texts  = [t for t, _ in SENTIMENT_DATA]
labels = [l for _, l in SENTIMENT_DATA]
print(f"Dataset: {sum(labels)} positive, {len(labels)-sum(labels)} negative")
"""),
        md("### A.2 Embed and Split"),
        code("""\
print("Embedding training sentences... (may take ~1 min)")
X = [get_embedding(t) for t in texts]
y = labels

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y)
print(f"Train: {len(X_train)}, Test: {len(X_test)}")
"""),
        md("### A.3 Train and Evaluate"),
        code("""\
clf = LogisticRegression(max_iter=1000, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred,
                             target_names=["Negative", "Positive"]))
"""),
        md("### A.4 Inference on New Sentences"),
        code("""\
new_sentences = [
    "Bu ürünü çok beğendim, tekrar satın alacağım.",
    "Hiç memnun kalmadım, para israfı.",
    "Fena değildi ama daha iyi olabilirdi.",
    "Bu kadar kötü bir hizmet beklemiyordum.",
]

for sent in new_sentences:
    emb   = get_embedding(sent)
    prob  = clf.predict_proba([emb])[0]
    label = "Positive" if prob[1] >= 0.5 else "Negative"
    print(f"[{label} | P(pos)={prob[1]:.2f}]  {sent}")
"""),

        # ---- Spam ----
        md("## Part B — Spam / Ham Detection"),
        md("""\
### B.1 Dataset

A small inline dataset of Turkish SMS and email texts. Ham (legitimate, label 0)
includes meeting reminders, casual messages, and work requests. Spam (label 1)
includes unsolicited prize notifications, phishing attempts, and promotional offers.

For a larger benchmark consider the Turkish spam dataset on Kaggle or
use machine translation to project an English SMS spam corpus (e.g., UCI SMS Spam).\
"""),
        code("""\
SPAM_DATA = [
    # Ham (0)
    ("Yarınki toplantı 10:00'da, hazır olursun değil mi?", 0),
    ("Annem seni de akşam yemeğine davet ediyor.", 0),
    ("Projenin son halini gönderebilir misin?", 0),
    ("Bugün hava çok güzel, parka gidelim mi?", 0),
    ("Raporu dün bitirdim, review edebilir misin?", 0),
    ("Doktor randevum Perşembe saat 14:00.", 0),
    ("Mağazada indirim var, gidelim mi?", 0),
    ("Ödevi bitirdim, notlarını benimle paylaşır mısın?", 0),
    ("Yeni ev güzel, taşınma için yardım eder misin?", 0),
    ("Akşam film izlemeye gidiyoruz, gelir misin?", 0),
    ("Toplantı notlarını herkes ile paylaşabilir misin?", 0),
    ("Arabanı bu hafta kullanabilir miyim?", 0),
    ("Marketten süt ve ekmek alır mısın?", 0),
    ("Sınav notun belli oldu mu?", 0),
    ("Çocukları yarın okula sen mi götürüyorsun?", 0),
    # Spam (1)
    ("TEBRİKLER! 10.000 TL nakit ödülü kazandınız, hemen tıklayın!", 1),
    ("ÜCRETSİZ iPhone 15 kazanmak için linke tıklayın!", 1),
    ("Bankanızdan acil mesaj: Hesabınız askıya alınıyor!", 1),
    ("500 TL bonus kredi kartınıza yüklendi, aktifleştirin!", 1),
    ("ÖZEL TEKLİF: Bugün üye olun, yüzde seksen indirim kazanın!", 1),
    ("Çekiliş sonucu: Siz kazandınız! Ödülünüzü almak için tıklayın.", 1),
    ("Kripto ile bir haftada büyük kazanç elde edin, kaçırmayın!", 1),
    ("Hesabınızda şüpheli işlem tespit edildi, linke tıklayın!", 1),
    ("VIP üyelik bedava! Sadece bugün, hemen kaydolun!", 1),
    ("SEÇİLDİNİZ: 1000 TL hediye çeki sizi bekliyor!", 1),
    ("Banka bilgilerinizi güncelleyin, aksi hâlde hesabınız silinecek.", 1),
    ("Özel teklifimizden yararlanmak için son gün bugün!", 1),
    ("Tebrikler, piyango çekilişini kazandınız!", 1),
    ("Ücretsiz tatil paketi için bilgilerinizi girin.", 1),
    ("Anında kredi onayı, hiç belge istenmez!", 1),
]

spam_texts  = [t for t, _ in SPAM_DATA]
spam_labels = [l for _, l in SPAM_DATA]
print(f"Dataset: {spam_labels.count(0)} ham, {spam_labels.count(1)} spam")
"""),
        md("### B.2 Train and Evaluate Spam Classifier"),
        code("""\
print("Embedding spam/ham sentences...")
Xs = [get_embedding(t) for t in spam_texts]
ys = spam_labels

Xs_train, Xs_test, ys_train, ys_test = train_test_split(
    Xs, ys, test_size=0.25, random_state=42, stratify=ys)

clf_spam = LogisticRegression(max_iter=1000, random_state=42)
clf_spam.fit(Xs_train, ys_train)

ys_pred = clf_spam.predict(Xs_test)
print(classification_report(ys_test, ys_pred, target_names=["Ham", "Spam"]))
"""),
        md("### B.3 Inference on New Messages"),
        code("""\
new_msgs = [
    "Bu ay telefon faturanı ödemeyi unutma.",
    "KAZAN! Çekilişimizde büyük ödüller sizi bekliyor!",
    "Yarın spor salonunda görüşürüz.",
    "Hesabınız güvenlik nedeniyle kısıtlandı, acil işlem yapın!",
]

for msg in new_msgs:
    emb   = get_embedding(msg)
    prob  = clf_spam.predict_proba([emb])[0]
    label = "SPAM" if prob[1] >= 0.5 else "ham"
    print(f"[{label} | P(spam)={prob[1]:.2f}]  {msg}")
"""),
    ]


# ---------------------------------------------------------------------------
# Notebook 28 — Multilingual Transfer Learning
# ---------------------------------------------------------------------------

def embeddings_multilingual_transfer():
    return [
        md("""\
# Multilingual Transfer Learning with Turkic Embeddings

Because NLLB-200 maps all 200 languages into a **single shared vector space**,
a classifier trained only on labelled data from one language can generalise—
to a remarkable degree—to other languages it has never seen during training.
This is called **zero-shot cross-lingual transfer**.

In this notebook we demonstrate three scenarios:

| Scenario | Training data | Test data |
|----------|--------------|-----------|
| Monolingual baseline | Turkish | Turkish |
| Zero-shot transfer | Turkish only | Uzbek, Azerbaijani, Kyrgyz |
| Multilingual (combined) | Turkish + Kazakh (via MT) | Uzbek, Azerbaijani, Kyrgyz |

The task is binary sentiment classification (positive / negative).

**Key insight:** even without any labelled data in Uzbek or Kyrgyz, the
shared embedding space allows the Turkish-trained classifier to identify
sentiment in those languages, because semantically similar sentences are
nearby in the embedding space regardless of language.\
"""),
        INSTALL,
        code("""\
import turkicnlp
from turkicnlp import Pipeline

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, accuracy_score
    from sklearn.model_selection import train_test_split
except ImportError:
    raise SystemExit("pip install scikit-learn")

# Download embeddings + translate for languages we will use
for lang in ["tur", "kaz", "uzb", "aze", "kir"]:
    turkicnlp.download(lang, processors=["embeddings", "translate"])

embed_tur = Pipeline("tur", processors=["embeddings"])

def embed(lang, text):
    # Use the target-language pipeline for embeddings
    return Pipeline(lang, processors=["embeddings"])(text).embedding
"""),

        md("## 1. Turkish Training Data"),
        code("""\
TUR_SENTIMENT = [
    ("Bu film gerçekten muhteşemdi, kesinlikle tavsiye ederim.", 1),
    ("Yemek çok lezzetliydi, restoran mükemmel.", 1),
    ("Ürün beklentilerimi tamamen karşıladı, çok memnunum.", 1),
    ("Harika bir tatildi, her şey mükemmeldi.", 1),
    ("Müşteri hizmetleri sorunumu hızla çözdü.", 1),
    ("Otel çok temiz ve konforluydu.", 1),
    ("Bu deneyim hayatımın en güzel anlarından biri oldu.", 1),
    ("Uygulama son derece kullanışlı ve hızlı.", 1),
    ("Ekip çok profesyoneldi, her konuda yardımcı oldular.", 1),
    ("Kurs içeriği çok zengin ve öğretici.", 1),
    ("Bu film tamamen zaman kaybıydı, berbat senaryo.", 0),
    ("Yemek soğuk geldi, tadı hiç iyi değildi.", 0),
    ("Ürün resimlerdeki gibi değildi, hayal kırıklığı.", 0),
    ("Tatil mahvoldu, otel çok kötüydü.", 0),
    ("Müşteri hizmetleri hiç yardımcı olmadı.", 0),
    ("Kargo 3 hafta sonra geldi, ürün hasarlıydı.", 0),
    ("Bu deneyim için para harcadığıma pişmanım.", 0),
    ("Fiyatına göre kalitesi çok düşük.", 0),
    ("Ekip kaba ve umursamaz davrandı.", 0),
    ("Kurs vaat edilen içeriği sunmadı, pişmanım.", 0),
]

tur_texts  = [t for t, _ in TUR_SENTIMENT]
tur_labels = [l for _, l in TUR_SENTIMENT]
print(f"Turkish data: {sum(tur_labels)} pos, {len(tur_labels)-sum(tur_labels)} neg")
"""),

        md("## 2. Build Test Sets via Machine Translation\n\n"
           "We translate the Turkish test sentences into Uzbek, Azerbaijani, and "
           "Kyrgyz using TurkicNLP's translation pipeline. The labels remain the "
           "same (translation preserves sentiment). This simulates having zero "
           "labelled data in these languages."),
        code("""\
# 5 positive + 5 negative sentences as a cross-lingual test set
TEST_TUR = [
    ("Ürün beklentilerimi tamamen karşıladı, çok memnunum.", 1),
    ("Bu deneyim hayatımın en güzel anlarından biri oldu.", 1),
    ("Otel çok temiz ve konforluydu.", 1),
    ("Kurs içeriği çok zengin ve öğretici.", 1),
    ("Uygulama son derece kullanışlı ve hızlı.", 1),
    ("Bu film tamamen zaman kaybıydı, berbat senaryo.", 0),
    ("Ürün resimlerdeki gibi değildi, hayal kırıklığı.", 0),
    ("Müşteri hizmetleri hiç yardımcı olmadı.", 0),
    ("Bu deneyim için para harcadığıma pişmanım.", 0),
    ("Kurs vaat edilen içeriği sunmadı, pişmanım.", 0),
]
test_labels = [l for _, l in TEST_TUR]

TARGET_LANGS = {
    "uzb": ("uzn_Latn", "Uzbek"),
    "aze": ("azj_Latn", "Azerbaijani"),
    "kir": ("kir_Cyrl", "Kyrgyz"),
}

print("Translating test set...")
test_translations = {}
for lang, (nllb_code, label) in TARGET_LANGS.items():
    trans_pipe = Pipeline("tur", processors=["translate"],
                          translate_tgt_lang=nllb_code)
    test_translations[lang] = [
        trans_pipe(t).translation for t, _ in TEST_TUR
    ]
    print(f"  {label}: {test_translations[lang][0]}")
"""),

        md("## 3. Scenario A — Monolingual Baseline (Train & Test on Turkish)"),
        code("""\
print("Embedding Turkish training data...")
X_tur = [embed("tur", t) for t in tur_texts]

X_train, X_test_tur, y_train, y_test_tur = train_test_split(
    X_tur, tur_labels, test_size=0.25, random_state=42, stratify=tur_labels)

clf = LogisticRegression(max_iter=1000, random_state=42)
clf.fit(X_train, y_train)

acc = accuracy_score(y_test_tur, clf.predict(X_test_tur))
print(f"Monolingual Turkish accuracy: {acc:.2%}")
"""),

        md("## 4. Scenario B — Zero-shot Transfer to Other Turkic Languages"),
        code("""\
# Train on ALL Turkish data (no held-out Turkish test set for this scenario)
clf_full = LogisticRegression(max_iter=1000, random_state=42)
clf_full.fit(X_tur, tur_labels)

print("Zero-shot cross-lingual evaluation:")
print(f"{'Language':<15} {'Accuracy':>9}")
print("-" * 26)
for lang, (_, label) in TARGET_LANGS.items():
    X_tgt = [embed(lang, t) for t in test_translations[lang]]
    acc   = accuracy_score(test_labels, clf_full.predict(X_tgt))
    print(f"{label:<15} {acc:>9.2%}")
"""),

        md("## 5. Scenario C — Multilingual Training (Turkish + Kazakh via MT)\n\n"
           "We now augment the training set by translating Turkish training "
           "sentences into Kazakh, then embedding them using the Kazakh pipeline. "
           "Combining both languages during training typically improves transfer "
           "to related Turkic languages."),
        code("""\
kaz_trans_pipe = Pipeline("tur", processors=["translate"],
                           translate_tgt_lang="kaz_Cyrl")

print("Translating training data to Kazakh...")
kaz_texts = [kaz_trans_pipe(t).translation for t in tur_texts]
X_kaz     = [embed("kaz", t) for t in kaz_texts]

# Combined training set
X_multi  = X_tur + X_kaz
y_multi  = tur_labels + tur_labels   # same labels — translation preserves sentiment

clf_multi = LogisticRegression(max_iter=1000, random_state=42)
clf_multi.fit(X_multi, y_multi)

print("\\nMultilingual classifier evaluation:")
print(f"{'Language':<15} {'Monolingual':>12} {'Multilingual':>13}")
print("-" * 42)
for lang, (_, label) in TARGET_LANGS.items():
    X_tgt  = [embed(lang, t) for t in test_translations[lang]]
    acc_m  = accuracy_score(test_labels, clf_full.predict(X_tgt))
    acc_ml = accuracy_score(test_labels, clf_multi.predict(X_tgt))
    delta  = f"+{acc_ml - acc_m:.2%}" if acc_ml > acc_m else f"{acc_ml - acc_m:.2%}"
    print(f"{label:<15} {acc_m:>12.2%} {acc_ml:>12.2%}  ({delta})")
"""),

        md("## 6. Interpreting the Results\n\n"
           "Even without any labelled Uzbek, Azerbaijani, or Kyrgyz data the "
           "Turkish-trained classifier achieves above-chance accuracy. "
           "Adding Kazakh (obtained freely via machine translation) further "
           "improves cross-lingual transfer. This demonstrates that:\n\n"
           "- The NLLB-200 embedding space is genuinely multilingual and "
           "language-agnostic for semantics.\n"
           "- Machine translation is an effective zero-cost strategy for "
           "generating multilingual training data when labelled data is scarce.\n"
           "- Combining data from multiple related Turkic languages produces "
           "a more robust cross-lingual classifier."),
    ]


# ---------------------------------------------------------------------------
# Notebook 29 — Toxicity Detection
# ---------------------------------------------------------------------------

def toxicity_detection():
    flores_turkic = [
        ("tur", "tur_Latn", "Turkish"),
        ("aze", "azj_Latn", "Azerbaijani"),
        ("kaz", "kaz_Cyrl", "Kazakh"),
        ("kir", "kir_Cyrl", "Kyrgyz"),
        ("tat", "tat_Cyrl", "Tatar"),
        ("tuk", "tuk_Latn", "Turkmen"),
        ("uig", "uig_Arab", "Uyghur"),
        ("uzb", "uzn_Latn", "Uzbek"),
    ]
    flores_table = "\\n".join(
        f"| {iso} | {nllb} | {name} |"
        for iso, nllb, name in flores_turkic
    )
    return [
        md(f"""\
# Toxicity Detection for Turkic Languages

Detecting toxic, offensive, or profane language is a critical component of
content moderation, brand safety filters, and safe-messaging platforms.
For Turkic languages, the problem is particularly challenging because:

- Most languages are **low-resource**: large annotated toxicity corpora
  simply do not exist for Tatar, Kyrgyz, Turkmen, Uyghur, or Bashkir.
- Agglutinative morphology makes **token matching fragile**: a toxic root
  may appear with dozens of different suffixes, all unseen by a simple
  word-list filter.
- Scripts differ across languages, so a single monolingual approach
  cannot be directly reused.

## Dataset: FLORES-200 Toxicity Word Lists (Toxicity-200)

Facebook Research maintains the **Toxicity-200** dataset—a curated word list
of toxic terms for all 200 NLLB-200 languages, grouped into four categories:

- Frequently used profanities
- Insults, hate speech, and demeaning language
- Pornographic terms
- Terms for body parts associated with sexual activity

The word lists are distributed as password-protected ZIP archives.

**Download URL:** `https://tinyurl.com/NLLB200TWL`
**Extraction password:** `tL4nLLb`
**File naming:** `<BCP47-code>_twl.zip`, one per language.

FLORES-200 Turkic languages covered:

| TurkicNLP ISO | FLORES BCP-47 | Language |
|:---:|:---:|:---|
{flores_table}

## Approaches Covered in This Notebook

| Section | Approach | Strengths | Weaknesses |
|---------|---------|-----------|------------|
| A | Token-based (keyword matching) | Fast, interpretable, no training needed | Misses morphological variants, context-blind |
| B | Embedding-based (per-language) | Captures context, handles morphology | Needs labelled examples per language |
| C | Unified multilingual classifier | One model for all languages, cross-lingual transfer | Requires combining training data |
\
"""),
        INSTALL,
        code("""\
import os, zipfile, math, pathlib, urllib.request
import turkicnlp
from turkicnlp import Pipeline

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, accuracy_score
    from sklearn.model_selection import train_test_split
except ImportError:
    raise SystemExit("pip install scikit-learn")

FLORES_LANGS = [
    ("tur", "tur_Latn", "Turkish"),
    ("aze", "azj_Latn", "Azerbaijani"),
    ("kaz", "kaz_Cyrl", "Kazakh"),
    ("kir", "kir_Cyrl", "Kyrgyz"),
    ("tat", "tat_Cyrl", "Tatar"),
    ("tuk", "tuk_Latn", "Turkmen"),
    ("uig", "uig_Arab", "Uyghur"),
    ("uzb", "uzn_Latn", "Uzbek"),
]

TWL_DIR = pathlib.Path("toxicity_wordlists")
TWL_DIR.mkdir(exist_ok=True)
"""),

        md("## Section A — Token-based Toxicity Detection"),
        md("""\
### A.1 Download and Load Toxicity Word Lists

We download the ZIP archive for each language, extract using the published
password, and load the word list into a Python set for O(1) lookup.

> **Note:** the files contain toxic language; treat them as sensitive data
> and do not display their contents directly in shared environments.\
"""),
        code("""\
# ---------------------------------------------------------------------------
# Download helper — fetches and extracts one language word list
# ---------------------------------------------------------------------------
TWL_URL      = "https://tinyurl.com/NLLB200TWL/{code}_twl.zip"
TWL_PASSWORD = b"tL4nLLb"

def download_wordlist(nllb_code: str) -> set:
    \"\"\"Download and extract the toxicity word list for a FLORES-200 language.\"\"\"
    zip_path = TWL_DIR / f"{nllb_code}_twl.zip"
    txt_path = TWL_DIR / f"{nllb_code}_twl.txt"

    if not txt_path.exists():
        if not zip_path.exists():
            url = f"https://tinyurl.com/NLLB200TWL"
            # The actual individual-file URL pattern; adjust if the tinyurl
            # redirects to a bulk download page requiring manual steps.
            print(f"  Downloading word list for {nllb_code}...")
            try:
                urllib.request.urlretrieve(
                    f"https://github.com/facebookresearch/flores/raw/main/"
                    f"toxicity/{nllb_code}_twl.zip",
                    zip_path,
                )
            except Exception as e:
                print(f"  Auto-download failed ({e}).")
                print(f"  Please download manually from {url}")
                print(f"  and place '{nllb_code}_twl.zip' in '{TWL_DIR}/'")
                return set()

        with zipfile.ZipFile(zip_path) as zf:
            names = zf.namelist()
            for name in names:
                zf.extract(name, TWL_DIR, pwd=TWL_PASSWORD)
        # Rename extracted file to consistent name if needed
        extracted = list(TWL_DIR.glob(f"*{nllb_code}*"))
        if extracted and extracted[0] != txt_path:
            extracted[0].rename(txt_path)

    words = set()
    if txt_path.exists():
        with open(txt_path, encoding="utf-8") as f:
            for line in f:
                w = line.strip().lower()
                if w:
                    words.add(w)
    return words

# Load word lists (downloads automatically if not cached)
wordlists = {}
for iso, nllb_code, name in FLORES_LANGS:
    wl = download_wordlist(nllb_code)
    wordlists[nllb_code] = wl
    print(f"  {name:<15}: {len(wl):>5} toxic terms loaded")
"""),

        md("### A.2 Token-based Classifier"),
        code("""\
def tokenize_simple(text: str) -> list:
    \"\"\"Whitespace + punctuation split, lowercase.\"\"\"
    import re
    return re.findall(r"[\\w']+", text.lower())

def token_classifier(text: str, wordlist: set) -> dict:
    \"\"\"
    Classify text as toxic/safe using keyword matching.
    Returns a dict with 'label', 'score', and 'matched_count'.
    \"\"\"
    tokens  = tokenize_simple(text)
    matches = [t for t in tokens if t in wordlist]
    score   = len(matches) / max(len(tokens), 1)
    return {
        "label":         "toxic" if matches else "safe",
        "score":         score,
        "matched_count": len(matches),
    }

# ---- Demo: Turkish token-based classifier ----
turkish_wl = wordlists.get("tur_Latn", set())

demo_sentences = [
    "Bugün hava çok güzel, parkta yürüyüş yaptım.",
    "Bu toplantı çok verimli geçti.",
    "Seni hiç sevmiyorum, berbat birisin.",        # mild negative
    "Bu film harika, kesinlikle izleyin.",
]

print(f"{'Sentence':<50} {'Label':>6} {'Score':>6}")
print("-" * 65)
for sent in demo_sentences:
    result = token_classifier(sent, turkish_wl)
    print(f"{sent[:49]:<50} {result['label']:>6} {result['score']:>6.3f}")
"""),

        md("### A.3 Evaluating the Token-based Approach\n\n"
           "A key weakness of keyword matching is that it misses morphologically "
           "inflected forms of toxic roots. In Turkish, for example, a toxic root "
           "can appear with dozens of case, number, and tense suffixes—all "
           "distinct tokens from the base form in the word list. "
           "The embedding-based approach in Section B addresses this limitation."),
        code("""\
# Demonstrate the morphological coverage gap
toxic_root_examples = [
    # (surface form, expected: toxic?)
    # Assume 'kötü' (bad/mean) is in the word list as a mild example
    ("kötüsün", True),    # 'you are bad' — suffix -sün
    ("kötüleştir", True), # 'make it bad' — causative
    ("kötülük", True),    # 'badness' — nominal
]
example_wl = {"kötü"}  # minimal demo wordlist
for form, expected in toxic_root_examples:
    result = token_classifier(form, example_wl)
    hit = result["label"] == "toxic"
    print(f"'{form}':  token match={hit}  (should be {expected})  "
          f"-> {'OK' if hit == expected else 'MISSED'}")
"""),

        # ---- Embedding-based ----
        md("## Section B — Embedding-based Toxicity Classifier (Per-language)"),
        md("""\
**Approach:**

1. Construct a training set by inserting words from the toxicity word list
   into simple sentence templates, creating contextually toxic examples.
   Safe examples are everyday benign sentences.
2. Obtain NLLB-200 embeddings for each training sentence.
3. Train a logistic regression classifier on the embeddings.
4. At inference time, embed the input sentence and classify.

Because the embeddings capture **morphological and semantic context** rather
than surface tokens, the classifier can recognise toxic intent even when
the exact word list entry is not present verbatim.\
"""),
        code("""\
# ---------------------------------------------------------------------------
# Construct training data from word list + templates
# ---------------------------------------------------------------------------
SAFE_TEMPLATES = [
    "Bugün hava çok güzel ve parkta yürüyüş yaptım.",
    "Toplantı çok verimli geçti, ekip harika çalıştı.",
    "Bu kitabı okumak çok keyifliydi.",
    "Arkadaşlarımla güzel bir akşam geçirdik.",
    "Yemek çok lezzetliydi, restoran mükemmeldi.",
    "Çocuklar bahçede mutlu bir şekilde oynuyor.",
    "Film gerçekten etkileyici ve düşündürücüydü.",
    "İş yerimiz çok verimli bir ortama sahip.",
    "Tatil boyunca harika anılar biriktirdik.",
    "Sınav sonuçları beklediğimden çok daha iyiydi.",
]

TOXIC_TEMPLATES = [
    "Sen tam bir {}!",
    "Bu {} bir insan.",
    "Onlar {} insanlardır.",
    "Ne kadar {} birisin.",
    "Böyle {} biriyle konuşmak istemiyorum.",
]

def build_training_data(wordlist: set, n_toxic: int = 30):
    \"\"\"Sample n_toxic words from the word list and inject into templates.\"\"\"
    import random
    words   = random.sample(list(wordlist), min(n_toxic, len(wordlist)))
    toxic_X, toxic_y = [], []
    for w in words:
        tmpl = random.choice(TOXIC_TEMPLATES)
        toxic_X.append(tmpl.format(w))
        toxic_y.append(1)
    safe_X  = SAFE_TEMPLATES * (max(1, n_toxic // len(SAFE_TEMPLATES)))
    safe_y  = [0] * len(safe_X)
    return toxic_X + safe_X, toxic_y + safe_y

# --- Train per-language embedding classifier for Turkish ---
turkicnlp.download("tur", processors=["embeddings"])
tur_embed = Pipeline("tur", processors=["embeddings"])

if turkish_wl:
    X_texts, y_tox = build_training_data(turkish_wl, n_toxic=40)
    print(f"Training set: {y_tox.count(1)} toxic, {y_tox.count(0)} safe")

    print("Embedding training sentences...")
    X_emb = [tur_embed(t).embedding for t in X_texts]

    clf_tox = LogisticRegression(max_iter=1000, random_state=42)
    clf_tox.fit(X_emb, y_tox)
    print("Classifier trained.")
else:
    print("Word list not available; run Section A.1 to download.")
    clf_tox = None
"""),
        code("""\
# --- Inference: morphological variants caught by embedding approach ---
if clf_tox:
    test_cases = [
        ("Bugün hava çok güzel.", "safe"),
        ("Toplantı çok verimli geçti.", "safe"),
        ("Sen gerçekten kötüsün.", "ambiguous-neg"),
        ("Bu film harika, izleyin!", "safe"),
    ]
    print(f"{'Sentence':<45} {'Pred':>6}  {'P(toxic)':>9}  {'Expected':>10}")
    print("-" * 75)
    for sent, expected in test_cases:
        emb  = tur_embed(sent).embedding
        prob = clf_tox.predict_proba([emb])[0]
        pred = "toxic" if prob[1] >= 0.5 else "safe"
        print(f"{sent[:44]:<45} {pred:>6}  {prob[1]:>9.3f}  {expected:>10}")
"""),

        # ---- Unified multilingual model ----
        md("## Section C — Unified Multilingual Toxicity Classifier"),
        md("""\
**Approach:**

Because NLLB-200 embeddings are language-agnostic, we can train a **single
logistic regression model** on combined training data from all nine FLORES
Turkic languages simultaneously. This model then works for all languages
at inference time, including those that contributed no or few training examples.

Steps:
1. For each language, build training sentences using the word list and templates.
2. Embed all sentences using the language-specific TurkicNLP pipeline.
3. Concatenate all embeddings into one large training matrix.
4. Train a single logistic regression.
5. Evaluate per-language accuracy on held-out test sentences.\
"""),
        code("""\
for iso, _, _ in FLORES_LANGS:
    turkicnlp.download(iso, processors=["embeddings"])

# Safe sentences translated to each language for diverse safe training data
SAFE_EN = [
    "Today the weather is very nice and I went for a walk in the park.",
    "The meeting was very productive, the team worked great.",
    "This book was very enjoyable to read.",
    "We spent a great evening with friends.",
    "The food was delicious and the restaurant was excellent.",
]

# Build combined multilingual training data
all_X, all_y = [], []

for iso, nllb_code, name in FLORES_LANGS:
    wl = wordlists.get(nllb_code, set())
    if not wl:
        print(f"  {name}: word list missing, skipping.")
        continue

    pipe = Pipeline(iso, processors=["embeddings"])
    trans_pipe = Pipeline("tur", processors=["translate"],
                          translate_tgt_lang=nllb_code)

    # Toxic examples
    import random
    words = random.sample(list(wl), min(20, len(wl)))
    for w in words:
        tmpl = random.choice(TOXIC_TEMPLATES)
        emb  = pipe(tmpl.format(w)).embedding
        all_X.append(emb)
        all_y.append(1)

    # Safe examples — translate generic English safe sentences
    for safe_en in SAFE_EN:
        safe_tgt = trans_pipe(safe_en).translation
        emb = pipe(safe_tgt).embedding
        all_X.append(emb)
        all_y.append(0)

    print(f"  {name:<15}: +{min(20,len(wl))} toxic, +{len(SAFE_EN)} safe")

print(f"\\nTotal training samples: {len(all_X)}"
      f" ({all_y.count(1)} toxic, {all_y.count(0)} safe)")
"""),
        code("""\
if len(all_X) > 10:
    X_tr, X_te, y_tr, y_te = train_test_split(
        all_X, all_y, test_size=0.2, random_state=42, stratify=all_y)

    clf_multi = LogisticRegression(max_iter=2000, random_state=42)
    clf_multi.fit(X_tr, y_tr)

    print("Overall evaluation on held-out multilingual test set:")
    print(classification_report(y_te, clf_multi.predict(X_te),
                                 target_names=["Safe", "Toxic"]))
else:
    print("Not enough data to train. Download word lists first.")
    clf_multi = None
"""),
        code("""\
# Per-language accuracy on a small held-out set per language
if clf_multi:
    print(f"{'Language':<15} {'Acc (toxic)':>12} {'Acc (safe)':>11}")
    print("-" * 40)
    for iso, nllb_code, name in FLORES_LANGS:
        wl = wordlists.get(nllb_code, set())
        if not wl:
            continue
        pipe = Pipeline(iso, processors=["embeddings"])
        trans_pipe = Pipeline("tur", processors=["translate"],
                              translate_tgt_lang=nllb_code)

        # 5 toxic + 5 safe test sentences
        test_words = random.sample(list(wl), min(5, len(wl)))
        toxic_embs = [pipe(random.choice(TOXIC_TEMPLATES).format(w)).embedding
                      for w in test_words]
        safe_embs  = [pipe(trans_pipe(s).translation).embedding
                      for s in SAFE_EN[:5]]

        toxic_preds = clf_multi.predict(toxic_embs)
        safe_preds  = clf_multi.predict(safe_embs)
        acc_tox  = sum(p == 1 for p in toxic_preds) / len(toxic_preds)
        acc_safe = sum(p == 0 for p in safe_preds)  / len(safe_preds)
        print(f"{name:<15} {acc_tox:>12.0%} {acc_safe:>11.0%}")
"""),
        md("## Summary\n\n"
           "| Approach | Strengths | Recommended when |\n"
           "|----------|-----------|------------------|\n"
           "| Token-based | Zero training data needed, fully interpretable | Quick baseline; language has a good word list |\n"
           "| Embedding (per-lang) | Context-aware, handles morphology | You have labelled examples for each target language |\n"
           "| Unified multilingual | One model for all 9 languages, cross-lingual transfer | Low-resource languages; unified deployment |\n\n"
           "For production systems, consider combining approaches: use the token "
           "list as a hard filter for known toxic terms, and the embedding "
           "classifier to catch novel or morphologically inflected forms."),
    ]


# ---------------------------------------------------------------------------
# Notebook 30 — Multilingual Neural Models (Glot500)
# ---------------------------------------------------------------------------

def multilingual_models():
    return [
        md("""\
# Multilingual Neural Models for Turkic Languages (Glot500)

TurkicNLP includes multilingual neural models based on the **Glot500** backbone
that provide POS tagging, dependency parsing, morphological analysis, and
lemmatisation across many Turkic languages using a single shared model.

These models are particularly powerful because they support:

- **Trained languages** (10): Turkish, Azerbaijani, Uzbek, Turkmen, Kazakh,
  Kyrgyz, Bashkir, Tatar, Uyghur, Ottoman Turkish
- **Zero-shot languages** (via proxy embeddings): Karakalpak, Kumyk, Sakha

This notebook demonstrates three capabilities:

| Feature | Processor | Backend |
|---------|-----------|---------|
| POS tagging + Dependency parsing | `pos`, `depparse` | `multilingual_glot500` |
| Morphological analysis (UPOS + UD features + lemma) | `morph_neural` | multilingual Glot500 morph |
| Backend comparison | Stanza vs Glot500 | side-by-side |\
"""),
        INSTALL,
        IMPORT,

        md("## 1. Download Models"),
        code("""\
# Download for multiple languages — the Glot500 backbone is shared
for lang in ["tur", "kaz", "uzb", "kaa"]:
    turkicnlp.download(lang)
"""),

        md("## 2. Multilingual POS Tagging and Dependency Parsing\n\n"
           "The Glot500-based POS tagger and dependency parser use a shared model "
           "with per-language embeddings. Use `pos_backend=\"multilingual_glot500\"` "
           "and `depparse_backend=\"multilingual_glot500\"` to select this backend."),
        code("""\
# Turkish
nlp_tur = Pipeline(
    "tur",
    processors=["tokenize", "pos", "depparse"],
    pos_backend="multilingual_glot500",
    depparse_backend="multilingual_glot500",
)
doc = nlp_tur("Ahmet bugün okula gitti.")
print("=== Turkish ===")
print(f"{'Word':<15} {'UPOS':<8} {'Head':<5} {'Deprel'}")
print("-" * 40)
for w in doc.words:
    print(f"{w.text:<15} {w.upos:<8} {w.head!s:<5} {w.deprel}")
"""),
        code("""\
# Kazakh (Cyrillic)
nlp_kaz = Pipeline(
    "kaz",
    processors=["tokenize", "pos", "depparse"],
    pos_backend="multilingual_glot500",
    depparse_backend="multilingual_glot500",
    script="Cyrl",
)
doc = nlp_kaz("Ахмет бүгін мектепке барды.")
print("=== Kazakh ===")
print(f"{'Word':<15} {'UPOS':<8} {'Head':<5} {'Deprel'}")
print("-" * 40)
for w in doc.words:
    print(f"{w.text:<15} {w.upos:<8} {w.head!s:<5} {w.deprel}")
"""),
        code("""\
# Uzbek (Latin)
nlp_uzb = Pipeline(
    "uzb",
    processors=["tokenize", "pos", "depparse"],
    pos_backend="multilingual_glot500",
    depparse_backend="multilingual_glot500",
)
doc = nlp_uzb("Ahmat bugun maktabga ketdi.")
print("=== Uzbek ===")
print(f"{'Word':<15} {'UPOS':<8} {'Head':<5} {'Deprel'}")
print("-" * 40)
for w in doc.words:
    print(f"{w.text:<15} {w.upos:<8} {w.head!s:<5} {w.deprel}")
"""),

        md("## 3. Zero-shot Parsing for Unseen Languages\n\n"
           "The multilingual model can parse languages it was never directly "
           "trained on, using proxy embeddings from related languages. "
           "Karakalpak (Kipchak, close to Uzbek) is one such zero-shot language."),
        code("""\
# Karakalpak — zero-shot via Uzbek proxy embedding
nlp_kaa = Pipeline(
    "kaa",
    processors=["tokenize", "pos", "depparse"],
    pos_backend="multilingual_glot500",
    depparse_backend="multilingual_glot500",
)
doc = nlp_kaa("Qiz dostina xat jazdi.")
print("=== Karakalpak (zero-shot) ===")
print(f"{'Word':<15} {'UPOS':<8} {'Head':<5} {'Deprel'}")
print("-" * 40)
for w in doc.words:
    print(f"{w.text:<15} {w.upos:<8} {w.head!s:<5} {w.deprel}")
print("\\nCoNLL-U:\\n", doc.to_conllu())
"""),

        md("## 4. Neural Morphological Analysis (Glot500 Morph)\n\n"
           "The `morph_neural` processor provides UPOS tags, UD morphological "
           "features, and lemmatisation for 21 Turkic languages using the Glot500 "
           "morph model. This is broader than the Stanza-based models."),
        code("""\
turkicnlp.download("tur", processors=["tokenize", "morph_neural"])

nlp_morph = Pipeline(
    "tur",
    processors=["tokenize", "morph_neural"],
)
doc = nlp_morph("Çocuklar okula gidiyorlar.")
print("=== Turkish — Neural Morphology ===")
print(f"{'Word':<20} {'UPOS':<8} {'Lemma':<15} {'Features'}")
print("-" * 70)
for w in doc.words:
    print(f"{w.text:<20} {w.upos:<8} {w.lemma:<15} {w.feats}")
"""),
        code("""\
# Neural morph for low-resource languages
for lang, text, label in [
    ("sah", "Мин оскуолаҕа бардым.", "Sakha"),
    ("kaa", "Men mektepke bardim.", "Karakalpak (zero-shot)"),
]:
    turkicnlp.download(lang, processors=["tokenize", "morph_neural"])
    nlp = Pipeline(lang, processors=["tokenize", "morph_neural"])
    doc = nlp(text)
    print(f"\\n=== {label} ===")
    for w in doc.words:
        print(f"  {w.text:<20} upos={w.upos:<8} lemma={w.lemma:<15} feats={w.feats}")
"""),

        md("## 5. Backend Comparison: Stanza vs Glot500\n\n"
           "For languages with both Stanza and Glot500 models (e.g., Turkish, Kazakh), "
           "you can compare outputs side-by-side. Stanza models are language-specific "
           "and typically more accurate for high-resource languages, while Glot500 "
           "provides broader coverage."),
        code("""\
text = "Ahmet bugün okula gitti."

# Stanza backend (language-specific, trained on Turkish IMST treebank)
nlp_stanza = Pipeline(
    "tur",
    processors=["tokenize", "pos", "lemma", "depparse"],
)
doc_stanza = nlp_stanza(text)

# Glot500 backend (multilingual)
nlp_glot = Pipeline(
    "tur",
    processors=["tokenize", "pos", "depparse"],
    pos_backend="multilingual_glot500",
    depparse_backend="multilingual_glot500",
)
doc_glot = nlp_glot(text)

print(f"{'Word':<15} {'Stanza UPOS':<13} {'Glot500 UPOS':<14} {'Stanza Dep':<12} {'Glot500 Dep'}")
print("-" * 70)
# Both backends use the same rule-based tokenizer, so word counts match
for ws, wg in zip(doc_stanza.words, doc_glot.words):
    match_pos = "✓" if ws.upos == wg.upos else "✗"
    match_dep = "✓" if ws.deprel == wg.deprel else "✗"
    print(f"{ws.text:<15} {ws.upos:<13} {wg.upos:<14} {ws.deprel:<12} {wg.deprel} {match_pos}{match_dep}")
"""),

        md("## 6. Processing Multiple Languages in a Loop\n\n"
           "The multilingual backend makes it easy to process text from many "
           "Turkic languages in a uniform way."),
        code("""\
sentences = [
    ("tur", "Bugün hava güzel.", "Turkish"),
    ("kaz", "Бүгін ауа райы жақсы.", "Kazakh"),
    ("uzb", "Bugun ob-havo yaxshi.", "Uzbek"),
    ("tat", "Бүген һава матур.", "Tatar"),
    ("kir", "Бүгүн аба ырайы жакшы.", "Kyrgyz"),
]

for lang, text, label in sentences:
    nlp = Pipeline(
        lang,
        processors=["tokenize", "pos", "depparse"],
        pos_backend="multilingual_glot500",
        depparse_backend="multilingual_glot500",
    )
    doc = nlp(text)
    tags = " ".join(f"{w.text}/{w.upos}" for w in doc.words)
    print(f"[{label:<10}] {tags}")
"""),
    ]


# ---------------------------------------------------------------------------
# Notebook 31 — Morpheme Tokenizer
# ---------------------------------------------------------------------------

def morpheme_tokenizer_demo():
    return [
        md("""\
# Morpheme Tokenizer — Hybrid Neural + FST Morpheme Segmentation

Turkic languages are agglutinative: a single word can carry many suffixes
encoding grammatical features like number, case, possession, tense, and more.
Understanding the internal morpheme structure of words is critical for:

- **Morphology-aware NLP:** better tokenisation for language models
- **Linguistic analysis:** automatic morpheme glossing
- **Educational tools:** breaking words into meaningful units
- **Low-resource MT:** morpheme-level translation strategies

TurkicNLP's `MorphemeTokenizer` uses a **hybrid approach**:

1. **Neural backbone** (Glot500 morph model) — provides UPOS, UD features, and lemma
2. **Apertium HFST transducer** — adds derivational morphology tags
3. **Language-specific suffix tables** — maps UD features to surface allomorphs
   using phonological rules (vowel harmony, consonant context)

**Supported languages (16):** Turkish, Azerbaijani, Kazakh, Uzbek, Kyrgyz,
Tatar, Bashkir, Turkmen, Crimean Tatar, Sakha, Khakas, Tuvan, Southern Altai,
Northern Altai, Chuvash, Gagauz\
"""),
        INSTALL,
        code("""\
import turkicnlp
from turkicnlp.processors.morpheme_tokenizer import MorphemeTokenizer
"""),

        md("## 1. Basic Usage — Kazakh"),
        code("""\
tok = MorphemeTokenizer(lang="kaz")
tok.load()

# "houses" — stem + plural suffix
result = tok.segment("үйлер")
print(f"Word:     {result.word}")
print(f"Segments: {result.segments}")
print(f"Labels:   {[m.label for m in result.morphemes]}")
print(f"Labeled:  {result.labeled}")
"""),
        code("""\
# More complex Kazakh examples
words = [
    ("мектепке", "to school — stem + dative"),
    ("баладан", "from child — stem + ablative"),
    ("кітабым", "my book — stem + possessive"),
    ("үйлеріңізде", "in your (formal) houses — stem + plural + poss + locative"),
    ("бардым", "I went — stem + past tense + 1sg"),
]

print(f"{'Word':<25} {'Segments':<35} {'Labels'}")
print("-" * 80)
for word, gloss in words:
    result = tok.segment(word)
    segs = " + ".join(result.segments)
    labs = " + ".join(m.label for m in result.morphemes)
    print(f"{word:<25} {segs:<35} {labs}")
    print(f"  {'':25} ({gloss})")
"""),

        md("## 2. Turkish Morpheme Segmentation"),
        code("""\
tok_tur = MorphemeTokenizer(lang="tur")
tok_tur.load()

words = [
    ("evlerde", "in houses — stem + plural + locative"),
    ("gidiyordum", "I was going — stem + progressive + past + 1sg"),
    ("okumuşlardır", "they have read — stem + evidential + plural + copula"),
    ("güzelleştirilmek", "to be beautified — stem + become + causative + passive + infinitive"),
    ("kitaplarımızdan", "from our books — stem + plural + possessive + ablative"),
]

print(f"{'Word':<25} {'Segments':<40} {'Labels'}")
print("-" * 90)
for word, gloss in words:
    result = tok_tur.segment(word)
    segs = " + ".join(result.segments)
    labs = " + ".join(m.label for m in result.morphemes)
    print(f"{word:<25} {segs:<40} {labs}")
    print(f"  {'':25} ({gloss})")
"""),

        md("## 3. Comparing Morpheme Segmentation Across Languages\n\n"
           "The same grammatical concept (e.g., plural + dative) is expressed "
           "with different allomorphs across Turkic languages due to vowel "
           "harmony and consonant assimilation rules."),
        code("""\
# "to schools" in different Turkic languages
examples = [
    ("tur", "okullara",    "Turkish"),
    ("kaz", "мектептерге", "Kazakh"),
    ("uzb", "maktablarga", "Uzbek"),
    ("kir", "мектептерге", "Kyrgyz"),
    ("tat", "мәктәпләргә", "Tatar"),
    ("aze", "məktəblərə",  "Azerbaijani"),
]

for lang, word, label in examples:
    tok = MorphemeTokenizer(lang=lang)
    tok.load()
    result = tok.segment(word)
    segs = " + ".join(result.segments)
    labs = " + ".join(m.label for m in result.morphemes)
    print(f"[{label:<12}] {word:<20} → {segs}")
    print(f"{'':16} labels: {labs}")
"""),

        md("## 4. Processing Full Sentences\n\n"
           "The `MorphemeTokenizer` can also process entire documents via "
           "its `.process()` method, which adds morpheme annotations to each word."),
        code("""\
from turkicnlp import Pipeline

# First create a pipeline to tokenize the text
nlp = Pipeline("kaz", processors=["tokenize", "morph_neural"])
doc = nlp("Мен мектепке бардым.")

# Then apply morpheme tokenizer to each word
tok_kaz = MorphemeTokenizer(lang="kaz")
tok_kaz.load()
doc = tok_kaz.process(doc)

print(f"{'Word':<20} {'Morphemes':<35} {'Labels'}")
print("-" * 70)
for w in doc.words:
    # _morphemes is set by MorphemeTokenizer.process() on each Word
    morphemes = getattr(w, '_morphemes', None)
    if morphemes:
        segs = " + ".join(m.text for m in morphemes)
        labs = " + ".join(m.label for m in morphemes)
    else:
        segs = w.text
        labs = "STEM"
    print(f"{w.text:<20} {segs:<35} {labs}")
"""),

        md("## 5. Vowel Harmony in Action\n\n"
           "One of the key features of the morpheme tokenizer is its awareness "
           "of phonological rules. The same suffix has different surface forms "
           "depending on the vowel harmony class of the stem."),
        code("""\
# Turkish dative suffix: -a/-e (palatal harmony)
# Turkish plural suffix: -lar/-ler (palatal harmony)
tok_tur2 = MorphemeTokenizer(lang="tur")
tok_tur2.load()

harmony_examples = [
    ("evlere",     "to houses — front vowel stem → -ler, -e"),
    ("okullara",   "to schools — back vowel stem → -lar, -a"),
    ("kitaplarda",  "in books — back vowel stem → -lar, -da"),
    ("defterlerde", "in notebooks — front vowel stem → -ler, -de"),
]

for word, note in harmony_examples:
    result = tok_tur2.segment(word)
    segs = " + ".join(result.segments)
    print(f"{word:<20} → {segs}")
    print(f"{'':20}   ({note})")
"""),
    ]


# ---------------------------------------------------------------------------
# Define all 24 per-language notebooks + 7 thematic notebooks
# ---------------------------------------------------------------------------
NOTEBOOKS = [
    ("01_turkish",          "Turkish",           turkish()),
    ("02_kazakh",           "Kazakh",            kazakh()),
    ("03_kyrgyz",           "Kyrgyz",            kyrgyz()),
    ("04_uyghur",           "Uyghur",            uyghur()),
    ("05_uzbek",            "Uzbek",             uzbek()),
    ("06_azerbaijani",      "Azerbaijani",       azerbaijani()),
    ("07_tatar",            "Tatar",             tatar()),
    ("08_ottoman_turkish",  "Ottoman Turkish",   ottoman()),
    ("09_bashkir",          "Bashkir",           bashkir()),
    ("10_turkmen",          "Turkmen",           turkmen()),
    ("11_crimean_tatar",    "Crimean Tatar",
     _simple_lang("crh", "Crimean Tatar", "Latn", "LATIN", "Kipchak",
                  "Beta", None, "Men mektepke baraman.",
                  ("CYRILLIC", "LATIN", "Мен мектепке бараман."))),
    ("12_karakalpak",       "Karakalpak",
     _simple_lang("kaa", "Karakalpak", "Latn", "LATIN", "Kipchak",
                  "Beta", None, "Men mektepke baraman.",
                  ("CYRILLIC", "LATIN", "Мен мектепке бараман."))),
    ("13_nogai",            "Nogai",
     _simple_lang("nog", "Nogai", "Cyrl", "CYRILLIC", "Kipchak",
                  "Prototype", "Мен мектепке бараман.", None,
                  ("CYRILLIC", "LATIN", "Мен мектепке бараман."))),
    ("14_kumyk",            "Kumyk",
     _simple_lang("kum", "Kumyk", "Cyrl", "CYRILLIC", "Kipchak",
                  "Prototype", "Мен мектепке бараман.", None)),
    ("15_karachay_balkar",  "Karachay-Balkar",
     _simple_lang("krc", "Karachay-Balkar", "Cyrl", "CYRILLIC", "Kipchak",
                  "Prototype", "Мен школгъа барама.", None)),
    ("16_sakha",            "Sakha (Yakut)",
     _simple_lang("sah", "Sakha", "Cyrl", "CYRILLIC", "Siberian",
                  "Prototype", "Мин оскуолаҕа барабын.", None,
                  notes="Sakha (Yakut) is spoken in the Sakha Republic, Russia.")),
    ("17_chuvash",          "Chuvash",
     _simple_lang("chv", "Chuvash", "Cyrl", "CYRILLIC", "Oghur",
                  "Prototype", "Эпĕ шкулта вĕренетĕп.", None,
                  notes="Chuvash belongs to the Oghur branch and is distinct from all other Turkic languages.")),
    ("18_altai",            "Altai",
     _tokenize_only("alt", "Altai", "Cyrillic", "Siberian",
                    "Мен школго барадым.")),
    ("19_tuvan",            "Tuvan",
     _tokenize_only("tyv", "Tuvan", "Cyrillic", "Siberian",
                    "Мен школага баар мен.")),
    ("20_khakas",           "Khakas",
     _tokenize_only("kjh", "Khakas", "Cyrillic", "Siberian",
                    "Мин чалтырарға парам.")),
    ("21_gagauz",           "Gagauz",
     _simple_lang("gag", "Gagauz", "Latn", "LATIN", "Oghuz",
                  "Prototype", None, "Ben mektebi gidiyom.",
                  notes="Gagauz is spoken primarily in Moldova.")),
    ("22_iranian_azerbaijani", "Iranian Azerbaijani",
     _tokenize_only("azb", "Iranian Azerbaijani", "Arabic", "Other",
                    "من مکتبه گئدیرم.",
                    notes="Iranian Azerbaijani is written in Perso-Arabic script.")),
    ("23_old_turkish",      "Old Turkish",
     _tokenize_only("otk", "Old Turkish", "Old Turkic Runic", "Historical",
                    "𐰼𐰀𐰭 𐰕𐰇𐰤𐱅𐰞𐰺𐰀",
                    notes="Old Turkish is used for Orkhon and Yenisei runic inscriptions.")),
    ("24_khalaj",           "Khalaj",
     _tokenize_only("klj", "Khalaj", "Arabic", "Arghu",
                    "من مکتبه گئدیرم.",
                    notes="Khalaj belongs to the Arghu branch, the most divergent Turkic branch.")),
    ("25_embeddings_monolingual",
     "Embeddings: Monolingual Semantic Similarity",
     embeddings_monolingual()),
    ("26_embeddings_crosslingual",
     "Embeddings: Cross-lingual Semantic Similarity",
     embeddings_crosslingual()),
    ("27_embeddings_classifier",
     "Embeddings: Text Classification (Sentiment & Spam)",
     embeddings_classifier()),
    ("28_embeddings_multilingual_transfer",
     "Embeddings: Multilingual Transfer Learning",
     embeddings_multilingual_transfer()),
    ("29_toxicity_detection",
     "Toxicity Detection for Turkic Languages",
     toxicity_detection()),
    ("30_multilingual_models",
     "Multilingual Neural Models (Glot500)",
     multilingual_models()),
    ("31_morpheme_tokenizer",
     "Morpheme Tokenizer (Hybrid Neural + FST)",
     morpheme_tokenizer_demo()),
]


# ---------------------------------------------------------------------------
# Write notebooks
# ---------------------------------------------------------------------------
for stem, display, cells in NOTEBOOKS:
    path = os.path.join(NOTEBOOKS_DIR, f"{stem}.ipynb")
    notebook = nb(cells)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, ensure_ascii=False, indent=1)
    print(f"  Written: {path}")

print(f"\nDone — {len(NOTEBOOKS)} notebooks created in ./{NOTEBOOKS_DIR}/")
