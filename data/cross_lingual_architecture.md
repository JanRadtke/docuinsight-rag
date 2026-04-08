# Cross-Lingual RAG Architecture

## Problem

Frage in Sprache X + Dokumente in Sprachen Y, Z, W = unvollständige Ergebnisse.

**Zwei Bruchstellen:**
1. **Retrieval:** BM25 matcht nur gleiche Sprache. Vektorsuche ist etwas robuster, aber nicht perfekt.
2. **Extraction:** Reader scheitert wenn Prompt-Sprache ≠ Chunk-Sprache (0 Facts statt 19).

### Messung (v2.0 — vor Fix)

| Frage | Sprache | Chunks gefunden | Facts extrahiert | Antwort |
|-------|---------|-----------------|------------------|---------|
| "What is CBT..." | EN | 8 | **19** | Vollständig |
| "Was ist CBT..." | DE | 8 | **0** | Fallback |

---

## Ziel-Architektur: Sprachagnostisches RAG

Das System muss **vollständig sprachagnostisch** funktionieren:
- User fragt auf **Hebräisch**
- Dokumente liegen auf **Chinesisch, Englisch, Deutsch**
- System findet relevante Chunks in **allen 3 Sprachen**
- Reader extrahiert Facts aus **jeder Sprache**
- Writer antwortet auf **Hebräisch**

---

## Architektur-Überblick

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER QUERY                               │
│                   "מהי CBT וילו טכניקות..."                      │
│                     (Hebräisch)                                  │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 1: LANGUAGE ROUTER                                        │
│                                                                 │
│  Input:  User-Frage (beliebige Sprache)                         │
│  Output: {                                                      │
│    "user_language": "he",                                       │
│    "doc_languages": ["en", "de", "zh"],  ← aus Ingestion-Meta   │
│    "english_question": "What is CBT and which techniques..."    │
│  }                                                              │
│                                                                 │
│  Woher kommen die doc_languages?                                │
│  → Ingestion speichert pro Dokument metadata["language"]        │
│  → Language Router liest die unique Languages aus ChromaDB      │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 2: MULTILINGUAL QUERY EXPANSION                           │
│                                                                 │
│  Für JEDE doc_language eine optimierte Suchanfrage generieren:  │
│                                                                 │
│  {                                                              │
│    "en": "CBT cognitive behavioural therapy techniques          │
│           depression treatment methods",                        │
│    "de": "KVT kognitive Verhaltenstherapie Techniken            │
│           Depression Behandlung Methoden",                      │
│    "zh": "认知行为疗法 CBT 技术 抑郁症 治疗方法"                    │
│  }                                                              │
│                                                                 │
│  1 LLM-Call: "Translate these search terms into [en, de, zh]"   │
│  (gpt-4o-mini, ~100 Tokens — alle Sprachen in einem Call)       │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 3: PARALLEL MULTILINGUAL RETRIEVAL                        │
│                                                                 │
│  Pro Sprache ein eigener Retrieval-Durchlauf:                   │
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │ EN Query    │  │ DE Query    │  │ ZH Query    │             │
│  │ BM25+Vector │  │ BM25+Vector │  │ BM25+Vector │             │
│  │ → 10 Hits   │  │ → 10 Hits   │  │ → 10 Hits   │             │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘             │
│         │                │                │                     │
│         └────────────────┼────────────────┘                     │
│                          ▼                                      │
│               ┌─────────────────────┐                           │
│               │   CROSS-LINGUAL RRF │                           │
│               │                     │                           │
│               │ Alle Hits mergen:   │                           │
│               │ score += 1/(k+rank) │                           │
│               │ pro Sprach-Durchlauf│                           │
│               │                     │                           │
│               │ Dokument das in 2   │                           │
│               │ Sprachen rankt →    │                           │
│               │ höherer Score!      │                           │
│               └──────────┬──────────┘                           │
│                          │                                      │
│                  Top-K Chunks (gemischt EN/DE/ZH)               │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 4: READER (Englisch-intern)                               │
│                                                                 │
│  Input:  english_question + Chunks (gemischte Sprachen)         │
│                                                                 │
│  Prompt: "Extract facts from these chunks. The question is in   │
│           English. Chunks may be in ANY language — extract       │
│           facts regardless of chunk language."                   │
│                                                                 │
│  Output: Facts auf Englisch (interne Lingua Franca)             │
│                                                                 │
│  Warum Englisch intern?                                         │
│  → LLMs performen am besten auf Englisch                        │
│  → Konsistente Fact-Qualität unabhängig von Chunk-Sprache       │
│  → Nur 1 Übersetzung am Ende (statt N Übersetzungen im Reader) │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 5: WRITER + LANGUAGE OUTPUT                               │
│                                                                 │
│  Input:  english_facts + original_question (Hebräisch)          │
│                                                                 │
│  Prompt: "Write your response in the language of the user's     │
│           question. User language: Hebrew.                       │
│           Base your answer on these English facts."              │
│                                                                 │
│  Output: Vollständige Antwort auf Hebräisch                     │
│          mit Quellenverweisen [1], [2], [3]                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Ingestion: Language Detection

Voraussetzung für Multilingual Query Expansion — wir müssen wissen, welche Sprachen in der DB liegen.

```
PDF Upload → Text extrahieren → detect_language(first_500_chars)
                                         │
                                         ▼
                              metadata["language"] = "de"
                              → ChromaDB speichern

Beim Start / Lazy:
  → unique_languages = set(all metadata["language"] values)
  → ["en", "de", "zh"]
```

**Language Detection:** Kein LLM nötig — `langdetect` Library. ✅ Implementiert in `ingest.py`.
`retriever.get_document_languages()` liest die unique Languages aus ChromaDB.

---

## Implementierungsstufen

### Stufe 1: Reader-Translation (v2.1) ✅ IMPLEMENTIERT

```
Reader: translate_to_english(question) → englische Extraktion
Writer: antwortet in User-Sprache (Prompt-Instruktion)
```

**Löst:** Extraction-Problem (0 → 19 Facts)
**Löst NICHT:** Retrieval-Bias (BM25 findet nur gleiche Sprache)

### Stufe 2: Language Detection bei Ingestion ✅ IMPLEMENTIERT

```
Ingestion: langdetect auf erste Seite → metadata["language"] = "en" / "de" / "zh-cn" / ...
           Alle Chunks, Parents, Images desselben Dokuments erben die Sprache.
Retriever: get_document_languages() → ["de", "en", "zh-cn"] aus ChromaDB
```

**Aufwand:** Erledigt (~30 Zeilen in ingest.py + retriever.py)
**Voraussetzung für:** Stufe 3
**Randfall:** Dokument mit mehreren Sprachen → erste Seite gewinnt (pragmatisch)

### Stufe 3: Multilingual Query Expansion

```
Planner: 1 LLM-Call → Suchbegriffe in allen doc_languages
Retriever: N parallele Hybrid-Suchen → Cross-Lingual RRF
```

**Aufwand:** Mittel (Planner + Retriever erweitern)
**Löst:** Retrieval-Bias vollständig

### Stufe 4: Lingua Franca Embeddings (optional, Enterprise)

```
Ingestion: Nicht-englische Chunks → englisch übersetzen → englisch embedden
           Original-Text in metadata["original_text"] behalten
Retrieval: Immer englische Suche → perfekte Vektorähnlichkeit
```

**Aufwand:** Hoch (Ingestion-Pipeline, Übersetzungskosten)
**Wann:** Wenn multilinguale Dokumente der Regelfall sind

---

## Kostenabschätzung pro Query

| Stufe | Extra LLM-Calls | Extra Tokens | Latenz |
|-------|-----------------|-------------|--------|
| Stufe 1 (Reader-Translation) | +1 mini | ~50 | +200ms |
| Stufe 3 (Query Expansion) | +1 mini | ~150 | +300ms |
| Stufe 3 (N Retrieval-Calls) | 0 (lokal) | 0 | +N×100ms |
| **Gesamt Stufe 1+3** | **+2 mini** | **~200** | **+600ms** |

Bei gpt-4o-mini Preisen: ~$0.003 pro Query extra. Vernachlässigbar.

---

## Warum RRF über Sprachgrenzen funktioniert

Beispiel: Hebräische Frage, Dokumente in EN + DE + ZH

```
EN-Suche:  Doc_A (Rank 1), Doc_B (Rank 2), Doc_C (Rank 5)
DE-Suche:  Doc_A (Rank 3), Doc_D (Rank 1), Doc_B (Rank 4)
ZH-Suche:  Doc_E (Rank 1), Doc_A (Rank 2)

RRF Scores (k=60):
  Doc_A: 1/61 + 1/63 + 1/62 = 0.0487  ← rankt in ALLEN 3 Sprachen → Top!
  Doc_D: 1/61              = 0.0164
  Doc_E: 1/61              = 0.0164
  Doc_B: 1/62 + 1/64       = 0.0317
```

→ Dokumente die in **mehreren Sprachen** relevant sind, steigen automatisch hoch.
→ Kein manuelles Gewichten, kein Sprach-Bias.

---

## Interview-Talking Points

1. **Problem:** Cross-Lingual Retrieval + Extraction — zwei separate Bruchstellen
2. **Architektur:** Sprachagnostisch — beliebige User-Sprache × beliebige Doc-Sprachen
3. **Multilingual Query Expansion:** 1 LLM-Call generiert Suchterme in N Sprachen
4. **Cross-Lingual RRF:** Dokumente die in mehreren Sprachen ranken steigen automatisch
5. **Interne Lingua Franca:** Reader arbeitet auf Englisch (beste LLM-Performance), Writer übersetzt zurück
6. **Trade-off-Bewusstsein:** Stufenweise Implementierung — Portfolio braucht nicht Enterprise-Grade CLIR
7. **Kostenanalyse:** +2 Mini-Calls pro Query, ~$0.003, vernachlässigbar vs. Qualitätsgewinn
