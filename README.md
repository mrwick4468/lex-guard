# LEX-GUARD

Code, legal resources and aggregate results for the article

> **Explainable AI for Detecting and Assessing Online Child Sexual Exploitation: A Retrieval-Augmented Legal Intelligence Framework**
> Trinh Thi Yen, Vietnam Women's Academy, Hanoi. Submitted to *Frontiers in Artificial Intelligence* (Technology and Law).

LEX-GUARD screens chat users for online grooming in four steps:

1. A lexical screen: TF-IDF features with a linear SVM.
2. An open LLM that labels grooming stages and behaviours and cites the chat lines behind its decision.
3. A conformal three-way decision (flag / human review / clear) that bounds the false-flag rate.
4. Retrieval-augmented, element-by-element legal analysis under EU, US and Vietnamese criminal law, with a verifier pass.

**This repository contains no chat text and no per-user outputs.** Only text data was used; no image material of any kind.

## Contents

| Path | What it is |
|---|---|
| `code/pan12_eda.py` | Corpus statistics; overlap of the Perverted Justice release with PAN 2012 |
| `code/baselines.py` | Stage 1: TF-IDF + logistic regression / linear SVM, with out-of-fold scores (Kaggle CPU kernel) |
| `code/lexguard.py` | Stages 2–4 and experiments E1–E7 (Kaggle kernel, 2 × T4 GPU, vLLM) |
| `code/build_legal_corpus.py` | Builds the legal corpus from the downloaded legal sources |
| `code/make_figures.py` | Builds the article's figures, tables and `numbers.json` from the kernel outputs |
| `legal/reference_map.json` | Behaviour → provisions that most directly criminalise it (used to score retrieval and citations) |
| `legal/elements.json` | Elements of the eight offences assessed (EU Dir. 2011/93/EU Art. 3(2), 6(1), 6(2); 18 U.S.C. §§ 1470, 2251(a), 2422(b); Vietnam Criminal Code Art. 145, 147) |
| `legal/legal_corpus_public.jsonl` | The 43 provisions used for retrieval. Vietnamese texts are omitted (see below). |
| `results/` | Aggregate metrics of every run (`metrics_*.json`), baselines, corpus statistics and the numbers quoted in the article |

## Data

* **PAN 2012 sexual predator identification corpus.** Available on request from Zenodo: https://doi.org/10.5281/zenodo.3713280. The kernels attach the Kaggle copy `randmansour/pan12-data`.
* **Perverted Justice transcripts (cleaned).** https://github.com/helenkoutli/PervertedJusticeDataset
* **Legal sources:**
  * Directive 2011/93/EU, from EUR-Lex.
  * 18 U.S.C. §§ 1470, 2251, 2252A, 2256, 2422 and 2425, from Cornell LII.
  * Vietnam Criminal Code No. 100/2015/QH13, in English translation:
    * Articles 15, 141–147 and 151: https://apolatlegal.com/wp-content/uploads/2023/12/Criminal-Code-2015.pdf
    * Article 14 as amended by Law No. 12/2017/QH14: https://english.luatvietnam.vn

  The Vietnamese translations are third-party works, so their text is not redistributed here. `build_legal_corpus.py` rebuilds the full corpus from the downloaded files in `datasets/legal_raw/`.

## Reproducing the results

1. Run `code/pan12_eda.py` and `code/baselines.py` as Kaggle kernels with the two datasets attached. The baselines write `scores_tfidf_linsvm.npy`.
2. Run `code/lexguard.py` as a Kaggle GPU kernel (2 × T4) with these attached:
   * PAN 2012
   * the Perverted Justice release
   * the legal corpus (`legal_corpus.jsonl`, `reference_map.json`, `elements.json`)
   * the baselines kernel output

   Settings used in the article:
   * **Primary model:** `MODEL='Qwen/Qwen3-8B'`, `FULL=True`.
   * **Comparison models:** `FULL=False`, with `MODEL` set in turn to `Qwen/Qwen2.5-7B-Instruct`, `ibm-granite/granite-3.3-8b-instruct`, `mistralai/Mistral-7B-Instruct-v0.3` and `microsoft/Phi-4-mini-instruct`.
   * **Resuming:** `RESUME=True` reruns the legal stages, re-using the detection outputs of a previous full run attached as a kernel source.
3. Run `code/make_figures.py` to rebuild the figures from the kernel outputs. It also needs the PAN 2012 ground-truth label files.

Decoding is greedy. A detection-only run takes 80–165 minutes per model on 2 × T4.

## Responsible use

LEX-GUARD is research software for decision support. Its legal output is a structured checklist for a qualified person, not a legal finding. The article shows that the LLM's element-level legal claims are often unsupported, and that LLMs used alone mistake adult decoys for offenders. Do not deploy it without:
* calibrating it on data from the target platform,
* human review of every flag, and
* legal review of every legal output.

## Licence

Code: MIT (see `LICENSE`). Legal texts remain subject to the terms of their sources.

## Citation

If you use this code, please cite the article (details to be added on publication).
