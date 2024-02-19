# Information Extraction of Political Statements at the Passage Level

## Overview

Briefly introduce your thesis topic, its significance, and the main research question(s) it addresses. This section should provide a high-level overview of the themes and objectives of your work.

### Author

- **Name:** Juan-Francisco Reyes
- **Institution:** Brandenburgische Technische Universität Cottbus-Senftenberg
- **Department:** Institute of Computer Science
- **Supervisor(s):** Gerd Wagner
- **Contact Information:** pacoreyes@protonmail.com

## Abstract

This thesis presents the development of an information extraction (IE) system designed to identify and extract political statements at the passage level, a task that is located amid computational linguistics and discourse analysis.

The core of this research is the creation and refinement of Natural Language Processing (NLP) models that serve distinct functions within the IE pipeline. These models have been meticulously crafted, employing both

- rule-based models, which leverage linguistic structures and features at the morphologic, syntactic, semantics, and pragmatics levels, and

- deep-learning models harness the performance of neural networks.

The rule-based models offer the advantage of explainability, a crucial aspect when dealing with the nuanced domain of political text, allowing for the extraction process to be transparent and interpretable through linguistic rules. On the other hand, deep-learning models contribute robustness and adaptability, learning from data to capture complex patterns that may elude rule-based approaches.
## Table of Contents

- [Thesis Overview](#overview)
- [Abstract](#abstract)
- [Table of Contents](#table-of-contents)
- [Paper 1: Performance and Explainability of Rule-based, Machine Learning, and BERT Models in Monologic and Dialogic Classification](#paper-1-title)
- [Paper 2: Title](#paper-2-title)
- [Paper 3: Title](#paper-3-title)
- [Conclusion](#conclusion)
- [Acknowledgments](#acknowledgments)
- [References](#references)

## Paper 1: Performance and Explainability of Rule-based, Machine Learning, and BERT Models in Monologic and Dialogic Classification

### Abstract

This study introduces an approach to this classification task, exploring the performance and explainability of rule-based and deep-learning models, focusing on the BERT model. We leverage this taxonomy for classifying political discourse, providing linguistic interpretations of defining features and model behaviors. The research aims to balance accuracy and explainability in NLP models while investigating the critical linguistic features vital for classification and their behavior across different models and exploring the possibility of developing a hybrid model that amalgamates their strengths.
### Key Contributions

- Bullet point list of the paper's key contributions to the field.

### How to Cite

Provide the citation format for this paper.

## Paper 2: Title

### Abstract

Summarize the second paper, highlighting the research problem, methodology, findings, and conclusions.

### Key Contributions

- Bullet point list of significant contributions made by this paper.

### How to Cite

Offer the citation details for this paper.

## Paper 3: Title

### Abstract

Outline the third paper's main points, including its research question, methods, results, and conclusions.

### Key Contributions

- List the critical contributions of this paper to the broader research field.

### How to Cite

Provide citation information for this paper.

## Conclusion

Summarize the overarching findings of your thesis and its contributions to the field. Discuss potential future research directions stemming from your work.

## Acknowledgments

Express gratitude to those who have helped you during your research and writing process, including advisors, peers, and any funding bodies.

## References

List all references cited throughout the README, formatted appropriately for your field.

---

This README provides an overview of my thesis and the associated papers. For further information or inquiries, please contact me at [Your Email].




install 
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_md
python -m spacy download en_core_web_lg
python -m spacy download en_core_web_trf



pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

Step 1: pip install https://github.com/explosion/spacy-experimental/releases/download/v0.6.0/en_coreference_web_trf-3.4.0a0-py3-none-any.whl
Step 2: python -m spacy download en_core_web_lg
Step 3: python -m spacy download en_core_web_md
Step 4: python -m spacy download en_core_web_trf
Step 5: python -m spacy download en_core_web_sm
