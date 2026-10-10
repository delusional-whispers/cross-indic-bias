# IndicFair-Audit: Auditing Cross-Lingual Stereotype Disparity in Large Language Models

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTest Status](https://img.shields.io/badge/pytest-15%20passed-brightgreen.svg)](tests/)
[![NIT Warangal](https://img.shields.io/badge/Institution-NIT%20Warangal-red.svg)](https://www.nitw.ac.in)

> **Research Repository & Faculty Evaluation Benchmark**  
> **Department of Computer Science & Engineering, National Institute of Technology, Warangal (NIT Warangal)**

---

## 1. Abstract & Research Motivation

As Large Language Models (LLMs) are increasingly deployed across India's multilingual digital ecosystem, evaluating cross-lingual equity is critical. While language models are frequently audited for stereotypical bias in English, their downstream behavior in high-resource Indo-Aryan (Hindi) and Dravidian (Telugu) languages often reveals stark, asymmetric bias divergence.

**IndicFair-Audit** is an autonomous, mathematically grounded audit framework that quantifies **Cross-Lingual Stereotype Disparity** across English, Hindi, and Telugu. By evaluating length-normalized sequence log-probabilities across culturally calibrated counterfactual triads, IndicFair-Audit reveals how occupational, caste-socioeconomic, and regional stereotypes shift when switching linguistic representations.

---

## 2. Mathematical & Metric Formulations

### 2.1 Length-Normalized Sequence Log-Probability
Standard sequence log-likelihood penalizes morphologically rich and multi-byte Indic scripts due to tokenizer fragmentation (Devanagari and Telugu subword token inflation). To ensure equitable cross-lingual comparison, we implement length-normalized sequence log-probability for target sequence tokens $T = (t_1, t_2, \dots, t_M)$ given prefix/context $C$:

$Score(S) = \frac{1}{|T|} \sum_{j=1}^{|T|} \log P_\theta(t_j \mid t_{<j}, C)$

### 2.2 Stereotype Preference Score (SPS)
For $N$ counterfactual pairs in language $L \in \{\text{en}, \text{hi}, \text{te}\}$:

$$\text{SPS}_L = \left( \frac{1}{N} \sum_{i=1}^N \mathbb{I}\left( \text{Score}(S_{i, L}^{\text{stereo}}) > \text{Score}(S_{i, L}^{\text{anti}}) \right) \right) \times 100\%$$

* **$\text{SPS} = 50.0\%$**: Unbiased null baseline (equal probability distribution across counterfactual identities).
* **$\text{SPS} > 50.0\%$**: Systematic alignment with prevailing social stereotypes.
* **$\text{SPS} < 50.0\%$**: Preference for counter-stereotypical representations.

### 2.3 Cross-Lingual Disparity Gaps ($\Delta_{\text{lang}}$)
The disparity gap measures divergence between target Indic languages and English:

$$\Delta_{\text{hi-en}} = \text{SPS}_{\text{hi}} - \text{SPS}_{\text{en}}$$

$$\Delta_{\text{te-en}} = \text{SPS}_{\text{te}} - \text{SPS}_{\text{en}}$$

$$\Delta_{\text{te-hi}} = \text{SPS}_{\text{te}} - \text{SPS}_{\text{hi}}$$

$$\Delta_{\max} = \max\left( |\Delta_{\text{hi-en}}|, |\Delta_{\text{te-en}}|, |\Delta_{\text{te-hi}}| \right)$$

### 2.4 Statistical Significance via Two-Sided Binomial Test
To determine whether an observed stereotypical preference rate deviates significantly from chance ($p = 0.5$), we compute the exact two-sided Binomial Test p-value:

$$H_0: p = 0.5 \quad \text{vs.} \quad H_1: p \neq 0.5$$

$$p\text{-value} = \sum_{k: P(X=k) \le P(X=k_{\text{obs}})} \binom{N}{k} (0.5)^N$$

---

## 3. Dual-Engine Architecture (Zero Broken Builds)

IndicFair-Audit incorporates a dual-engine architecture in `src/model_interface.py`:

```
                           ┌──────────────────────────────┐
                           │      BenchmarkLoader         │
                           │ (data/benchmark_pairs.json)  │
                           └──────────────┬───────────────┘
                                          │
                                          ▼
                           ┌──────────────────────────────┐
                           │         BaseScorer           │
                           │   (Abstract Base Class)      │
                           └───────┬──────────────┬───────┘
                                   │              │
                   ┌───────────────┴──┐        ┌──┴─────────────────────────┐
                   │  Mock Engine     │        │    Live Hugging Face       │
                   │ (Deterministic)  │        │       (PyTorch / HF)       │
                   ├──────────────────┤        ├────────────────────────────┤
                   │ • Zero GPU req   │        │ • AutoModelForCausalLM     │
                   │ • MD5 token hash │        │ • Length-normalized logits │
                   │ • Instant CI/CD  │        │ • CUDA / MPS / CPU         │
                   └──────────────────┘        └────────────────────────────┘
```

1. **Deterministic Offline/Mock Engine (`MockDeterministicScorer`)**:
   Uses cryptographic MD5 token hashing conditioned on seeds and character distributions. Produces consistent, reproducible log-probabilities without downloading weights or requiring GPU hardware. Guarantees 100% test pass rate offline.
2. **Live Hugging Face Engine (`HuggingFaceScorer`)**:
   Loads causal language models (e.g. `HuggingFaceTB/SmolLM2-360M-Instruct`, `google/gemma-2-2b-it`) via PyTorch, performing exact autoregressive cross-entropy log-likelihood calculations with target token length normalization.

---

## 4. Repository Layout

```text
IndicFair-Audit/
├── data/
│   ├── benchmark_pairs.json     # Culturally grounded triads (English, Hindi, Telugu)
│   └── categories.json          # Taxonomy definitions and severity weighting
├── src/
│   ├── __init__.py              # Public package interfaces
│   ├── config.py                # AuditConfig dataclass & device resolution
│   ├── data_loader.py           # Schema validator & Unicode-safe data loader
│   ├── model_interface.py       # Dual-engine scoring abstraction
│   ├── metrics.py               # SPS, Binomial test, and disparity formulations
│   ├── pipeline.py              # End-to-end evaluation orchestrator
│   └── visualizer.py            # Plotly bar charts and domain-shift radar plots
├── tests/
│   ├── __init__.py
│   ├── test_data_loader.py      # Schema, Unicode codepoints & loader tests
│   ├── test_metrics.py          # Metric synthetic edge case tests
│   └── test_pipeline.py         # End-to-end mock execution & reproducibility
├── app.py                       # Streamlit interactive web dashboard
├── run_audit.py                 # CLI benchmark runner with formatted ASCII tables
├── requirements.txt             # Production dependency specification
└── README.md                    # Research documentation
```

---

## 5. Benchmark Dataset Taxonomy

The benchmark dataset in `data/benchmark_pairs.json` contains 12 culturally grounded counterfactual triads spanning three domains:

| Category ID | Social Domain | Representative Counterfactual Constructs |
| :--- | :--- | :--- |
| `gender_stem_labor` | Gender & STEM/Labor Roles | Surgeon vs. Nurse; Software Architect; Heavy Mechanical Engineering; Commercial Aviation Pilot. |
| `socioeconomic_caste` | Socioeconomic & Caste Nuance | Academic Senate Dean vs. Sanitation; Sewer Maintenance Workers vs. Supervisor; High Court Bench; Corporate Boardroom Acquisition. |
| `regional_linguistic` | Regional & Ethnolinguistic Bias | AI Researcher (South vs North); Federal Governance Voices (Hindi vs Telugu); Carnatic Maestro vs Folk Singer; Migrant Scaffolding Labor. |

Each item contains parallel triads in:
* **English** (Standard Latin script)
* **Hindi** (Devanagari script: `U+0900`–`U+097F`)
* **Telugu** (Telugu script: `U+0C00`–`U+0C7F`)

---

## 6. Installation & Quick Start

### 6.1 Environment Setup
```bash
# Clone or navigate to the repository
cd IndicFair-Audit

# Create and activate a Python virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

### 6.2 Running the Test Suite
Execute the full test suite via `pytest`:
```bash
pytest tests/ -v
```

Expected output:
```text
tests/test_data_loader.py::test_loader_loads_at_least_12_pairs PASSED
tests/test_data_loader.py::test_pairs_contain_all_required_languages PASSED
tests/test_data_loader.py::test_indic_unicode_scripts_presence PASSED
tests/test_data_loader.py::test_get_pairs_by_category PASSED
tests/test_data_loader.py::test_get_triad PASSED
tests/test_data_loader.py::test_schema_validation_failure PASSED
tests/test_metrics.py::test_compute_pair_verdict PASSED
tests/test_metrics.py::test_sps_synthetic_edge_cases PASSED
tests/test_metrics.py::test_compute_p_value_binomial_properties PASSED
tests/test_metrics.py::test_compute_cross_lingual_gap PASSED
tests/test_metrics.py::test_aggregate_report_structure PASSED
tests/test_pipeline.py::test_pipeline_end_to_end_mock_execution PASSED
tests/test_pipeline.py::test_pipeline_reproducibility PASSED
tests/test_pipeline.py::test_pipeline_category_filter PASSED
tests/test_pipeline.py::test_pipeline_save_and_reload PASSED
============================== 15 passed in 0.38s ==============================
```

---

## 7. Command-Line Interface (CLI Runner)

Run the benchmark audit from the command line:

### Fast Deterministic Mock Mode
```bash
python run_audit.py --mode mock
```

Sample output:
```text
================================================================================
             INDICFAIR-AUDIT: CROSS-LINGUAL STEREOTYPE DISPARITY
       Department of Computer Science & Engineering, NIT Warangal
================================================================================
[*] Initializing AuditPipeline [Engine: MOCK]...
[*] Executing benchmark scoring across English, Hindi, and Telugu...
[✓] Successfully exported full audit report to: output/audit_results.json

[+] TABLE 1: CROSS-LINGUAL STEREOTYPE PREFERENCE SCORES (SPS)
+------------------+--------------+----------+----------+--------+------------+------------+----------------+
| Language         | Total Pairs  | Stereo   | Anti     | Tie    | SPS (%)    | p-value    | Significance   |
+------------------+--------------+----------+----------+--------+------------+------------+----------------+
| English          | 12           | 8        | 4        | 0      | 66.67%     | 0.3877     | Not Sig. (ns)  |
| Hindi (हिन्दी)   | 12           | 4        | 8        | 0      | 33.33%     | 0.3877     | Not Sig. (ns)  |
| Telugu (తెలుగు)  | 12           | 3        | 9        | 0      | 25.00%     | 0.1460     | Not Sig. (ns)  |
+------------------+--------------+----------+----------+--------+------------+------------+----------------+

[+] TABLE 2: CROSS-LINGUAL DISPARITY GAPS
+------------------------+------------------------------+--------------+--------------------------------+
| Disparity Metric       | Formulation                  | Gap (%)      | Interpretation                 |
+------------------------+------------------------------+--------------+--------------------------------+
| Δ(hi - en)             | SPS(Hindi) - SPS(English)    | -33.33%      | Hindi disparity relative to English |
| Δ(te - en)             | SPS(Telugu) - SPS(English)   | -41.67%      | Telugu disparity relative to English |
| Δ(te - hi)             | SPS(Telugu) - SPS(Hindi)     | -8.33%       | Dravidian vs Indo-Aryan disparity |
| Max Disparity          | max(|Δ_ij|)                  | 41.67%       | Peak Cross-Lingual Stereotype Divergence |
+------------------------+------------------------------+--------------+--------------------------------+
```

### Live Hugging Face Mode
To audit a real causal language model (e.g., SmolLM2-360M):
```bash
python run_audit.py --mode live --model HuggingFaceTB/SmolLM2-360M-Instruct
```

### Domain-Specific Filter
```bash
python run_audit.py --mode mock --category gender_stem_labor
```

---

## 8. Streamlit Research Dashboard

Launch the interactive web portal:
```bash
streamlit run app.py
```

### Dashboard Capabilities:
* **Executive Metrics Row**: Real-time KPI cards displaying $\text{SPS}_{\text{en}}$, $\text{SPS}_{\text{hi}}$, $\text{SPS}_{\text{te}}$, and $\Delta_{\max}$ with baseline deltas.
* **Interactive Visualizations**:
  * Plotly grouped bar chart with unbiased 50% dotted baseline.
  * Plotly radar/spider chart mapping multidimensional bias shift across social domains.
* **Prompt-Level Inspection Table**: Filterable table showing original contexts, stereotypical continuations, anti-stereotypical continuations, raw scores, and directional verdicts.
* **Live Triad Probe**: Form allowing researchers to input custom prompts in English, Devanagari, or Telugu and compute instant length-normalized likelihoods and directional verdicts.
* **One-Click JSON Export**: Download raw audit reports formatted for academic publication.

---

## 9. Academic Citation & Departmental Attribution

If you utilize this framework or benchmark in academic research, please cite:

```bibtex
@misc{indicfair_audit_2026,
  title={IndicFair-Audit: Auditing Cross-Lingual Stereotype Disparity in Large Language Models},
  author={Anand, Kushagra and Faculty of Computer Science & Engineering},
  institution={National Institute of Technology, Warangal},
  year={2026},
  url={https://github.com/nitw-cse/IndicFair-Audit}
}
```
