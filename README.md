# Knowledge-Grounded Multi-Agent RAG for Hindi Counter-Speech

Official implementation and evaluation benchmark for a multi-agent Retrieval-Augmented Generation (RAG) framework mitigating online hate speech in Hindi across three strategies: **Factual Refutation**, **Counter-Questioning**, and **Positive De-escalation**.

## 📊 Benchmark Results (N = 632 Prompts)

| Metric Dimension | Evaluation Metric | Baseline Hate Speech | Factual Refutation | Counter-Questioning | Positive De-escalation | Optimal Direction |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Reference-Based** | **mBERT-F1** | — | 0.6626 | 0.6738 | **0.6766** | Higher (↑) |
| | **BLEU-2** | — | 0.0638 | **0.0931** | 0.0703 | Higher (↑) |
| | **ROUGE-L** | — | 0.1444 | **0.1769** | 0.1690 | Higher (↑) |
| **Safety** | **MuRIL Toxicity** | 0.8844 | 0.1673 | 0.2375 | **0.0987** | Lower (↓) |
| | **Toxicity Reduction (%)** | 0.0% | **81.08%** | **73.14%** | **88.84%** | Higher (↑) |
| **Informativeness** | **Substantive Words** | — | **45.25** | 22.65 | 24.82 | Higher (↑) |
| | **Poisson Mean RIDF** | — | **+0.0706** | +0.0361 | -0.0210 | Higher (↑) |
| | **Information Gain** | — | 0.1354 | 0.1352 | **0.2007** | Higher (↑) |
| **Contextual Relevance** | **Dense Cosine Coherence** | — | 0.4020 | **0.4291** | 0.3685 | Target [0.35-0.45] |
| **Linguistic Accessibility** | **Sinha et al. RH1 (Grade)** | 2.8797 | 3.1760 | **2.6771** | 3.0678 | Lower/Accessible (↓) |
| | **Sinha et al. RH2 (Grade)** | 4.1430 | 5.1985 | 4.4655 | **4.1478** | Lower/Accessible (↓) |

## 🛠️ Repository Layout

- `evaluation/`: Standalone scripts for toxicity, readability, RIDF informativeness, and dense cosine coherence.
- `data/`: Sample inputs and linguistic lexicons.
- `requirements.txt`: Python package dependencies.
