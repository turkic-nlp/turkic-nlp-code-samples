# TurkicNLP Code Samples

Companion Jupyter notebooks for the paper
**"TurkicNLP: An Open-Source NLP Toolkit for Turkic Languages"**

All notebooks are self-contained and installable with a single `pip` command.

---

## Installation

```bash
pip install turkicnlp                    # core (tokenisation, transliteration)
pip install "turkicnlp[stanza]"          # + neural POS, lemma, depparse, NER
pip install "turkicnlp[nllb]"            # + cross-lingual embeddings & translation
pip install "turkicnlp[all]"             # all optional extras
pip install scikit-learn matplotlib      # required for notebooks 27–29
```

---

## Part 1 — Per-language Notebooks (24 notebooks)

One notebook per Turkic language, demonstrating every processing component
available for that language.

| # | Notebook | ISO | Script(s) | Branch | Components |
|---|----------|-----|-----------|--------|------------|
| 01 | [Turkish](notebooks/01_turkish.ipynb) | `tur` | Latin | Oghuz | tok · morph · pos · lemma · dep · NER · emb · trans |
| 02 | [Kazakh](notebooks/02_kazakh.ipynb) | `kaz` | Cyrillic / Latin | Kipchak | tok · morph · pos · lemma · dep · NER · emb · trans |
| 03 | [Kyrgyz](notebooks/03_kyrgyz.ipynb) | `kir` | Cyrillic | Kipchak | tok · morph · pos · lemma · dep · emb · trans |
| 04 | [Uyghur](notebooks/04_uyghur.ipynb) | `uig` | Arabic / Latin (ULY) | Karluk | tok · morph · pos · lemma · dep · translit · emb · trans |
| 05 | [Uzbek](notebooks/05_uzbek.ipynb) | `uzb` | Latin / Cyrillic | Karluk | tok · morph · translit · emb · trans |
| 06 | [Azerbaijani](notebooks/06_azerbaijani.ipynb) | `aze` | Latin / Cyrillic | Oghuz | tok · morph · translit · emb · trans |
| 07 | [Tatar](notebooks/07_tatar.ipynb) | `tat` | Cyrillic / Zamanälif | Kipchak | tok · morph · translit · emb · trans |
| 08 | [Ottoman Turkish](notebooks/08_ottoman_turkish.ipynb) | `ota` | Arabic | Historical | tok · pos · lemma · dep · translit · emb · trans |
| 09 | [Bashkir](notebooks/09_bashkir.ipynb) | `bak` | Cyrillic | Kipchak | tok · morph · emb · trans |
| 10 | [Turkmen](notebooks/10_turkmen.ipynb) | `tuk` | Latin / Cyrillic | Oghuz | tok · morph · translit · emb · trans |
| 11 | [Crimean Tatar](notebooks/11_crimean_tatar.ipynb) | `crh` | Latin / Cyrillic | Kipchak | tok · morph · translit · emb · trans |
| 12 | [Karakalpak](notebooks/12_karakalpak.ipynb) | `kaa` | Latin / Cyrillic | Kipchak | tok · morph · translit · emb · trans |
| 13 | [Nogai](notebooks/13_nogai.ipynb) | `nog` | Cyrillic | Kipchak | tok · morph · emb · trans |
| 14 | [Kumyk](notebooks/14_kumyk.ipynb) | `kum` | Cyrillic | Kipchak | tok · morph · emb · trans |
| 15 | [Karachay-Balkar](notebooks/15_karachay_balkar.ipynb) | `krc` | Cyrillic | Kipchak | tok · morph · emb · trans |
| 16 | [Sakha (Yakut)](notebooks/16_sakha.ipynb) | `sah` | Cyrillic | Siberian | tok · morph · emb · trans |
| 17 | [Chuvash](notebooks/17_chuvash.ipynb) | `chv` | Cyrillic | Oghur | tok · morph · emb · trans |
| 18 | [Altai](notebooks/18_altai.ipynb) | `alt` | Cyrillic | Siberian | tok · emb · trans |
| 19 | [Tuvan](notebooks/19_tuvan.ipynb) | `tyv` | Cyrillic | Siberian | tok · emb · trans |
| 20 | [Khakas](notebooks/20_khakas.ipynb) | `kjh` | Cyrillic | Siberian | tok · emb · trans |
| 21 | [Gagauz](notebooks/21_gagauz.ipynb) | `gag` | Latin | Oghuz | tok · morph · emb · trans |
| 22 | [Iranian Azerbaijani](notebooks/22_iranian_azerbaijani.ipynb) | `azb` | Arabic | Oghuz | tok · emb · trans |
| 23 | [Old Turkish](notebooks/23_old_turkish.ipynb) | `otk` | Old Turkic Runic | Historical | tok · emb |
| 24 | [Khalaj](notebooks/24_khalaj.ipynb) | `klj` | Arabic | Arghu | tok · emb · trans |

**Key:** tok = tokenise, morph = morphological analysis (Apertium FST),
pos = POS tagging, lemma = lemmatisation, dep = dependency parsing,
NER = named entity recognition, translit = script transliteration,
emb = NLLB-200 sentence embeddings, trans = machine translation.

---

## Part 2 — Thematic Embedding Notebooks (5 notebooks)

Advanced use cases built on NLLB-200 sentence embeddings.

| # | Notebook | Task | Methods |
|---|----------|------|---------|
| 25 | [Monolingual Similarity](notebooks/25_embeddings_monolingual.ipynb) | Semantic similarity within one language | Cosine similarity, nearest-neighbour search, pairwise matrix |
| 26 | [Cross-lingual Similarity](notebooks/26_embeddings_crosslingual.ipynb) | Semantic similarity across Turkic languages | MT-based alignment, 9-language similarity matrix, heatmap, cross-lingual retrieval |
| 27 | [Text Classification](notebooks/27_embeddings_classifier.ipynb) | Sentiment analysis + spam/ham detection | Embeddings as features, logistic regression, inline Turkish datasets |
| 28 | [Multilingual Transfer](notebooks/28_embeddings_multilingual_transfer.ipynb) | Zero-shot cross-lingual classification | Turkish-trained classifier tested on Uzbek/Azerbaijani/Kyrgyz; MT-augmented multilingual training |
| 29 | [Toxicity Detection](notebooks/29_toxicity_detection.ipynb) | Safe/toxic text classification | Token-based keyword matching; embedding-based per-language classifier; unified multilingual model for 8 FLORES-200 Turkic languages |


---

## Toxicity Detection Setup (Notebook 29)

Notebook 29 uses the [FLORES-200 Toxicity-200 word lists](https://github.com/facebookresearch/flores/tree/main/toxicity) published by Meta Research. The word lists for each language are distributed as password-protected ZIP archives.

**Download URL:** `https://tinyurl.com/NLLB200TWL`
**Extraction password:** `tL4nLLb`
**File naming:** `<BCP47-code>_twl.zip`

FLORES-200 Turkic languages covered by notebook 29:

| TurkicNLP ISO | FLORES BCP-47 | Language |
|:---:|:---:|:---|
| tur | tur_Latn | Turkish |
| aze | azj_Latn | Azerbaijani |
| kaz | kaz_Cyrl | Kazakh |
| kir | kir_Cyrl | Kyrgyz |
| tat | tat_Cyrl | Tatar |
| tuk | tuk_Latn | Turkmen |
| uig | uig_Arab | Uyghur |
| uzb | uzn_Latn | Uzbek |

> **Note:** The word lists contain toxic and offensive language. Treat them as sensitive data and do not display their contents in shared or public environments.

---

## Citation

```bibtex
@inproceedings{hakimov2026turkicnlp,
  title     = {{TurkicNLP}: An Open-Source {NLP} Toolkit for {Turkic} Languages},
  author    = {Hakimov, Sherzod},
  year      = {2026}
}
```

## Links

- **Library:** [https://turkic-nlp.github.io](https://turkic-nlp.github.io)
- **pip:** `pip install turkicnlp`
- **Paper:** TurkicNLP: An Open-Source NLP Toolkit for Turkic Languages
