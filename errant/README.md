# FERRANTI

This repository contains the Factual ERRor ANnotation ToolkIt (FERRANTI) designed for factual error correction described in:

> Mingqi Gao, Xiaojun Wan, Jia Su, Zhefeng Wang, and Baoxing Huai. 2023. [**Reference Matters: Benchmarking Factual Error Correction for Dialogue Summarization with Fine-grained Evaluation Framework**](https://aclanthology.org/2023.acl-long.779/). In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 13932–13959, Toronto, Canada. Association for Computational Linguistics.

We are very grateful to the authors of ERRANT. Most of this code comes from the repository of [ERRANT](https://github.com/chrisjbryant/errant), which is described in:

> Christopher Bryant, Mariano Felice, and Ted Briscoe. 2017. [**Automatic annotation and evaluation of error types for grammatical error correction**](https://www.aclweb.org/anthology/P17-1074/). In Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). Vancouver, Canada.

> Mariano Felice, Christopher Bryant, and Ted Briscoe. 2016. [**Automatic extraction of learner errors in ESL sentences using linguistically enhanced alignments**](https://www.aclweb.org/anthology/C16-1079/). In Proceedings of COLING 2016, the 26th International Conference on Computational Linguistics: Technical Papers. Osaka, Japan.

# Overview

Inspired by ERRANT, FERRANTI is designed to annotate parallel English sentences with factual error type information. Specifically, given an original summary and a corrected summary, FERRANTI will extract the edits that transform the former to the latter and classify them according to a rule-based factual error type framework. Similar to ERRANT, annotated output files are in M2 format.

### Example:  
**Original**: Greg needs to stay after hours . Betsy can ' t pick him up .  
**Corrected**: Greg needs to stay after hours . Betsy can ' t pick Johnny up .  
**Output M2**:  
S Greg needs to stay after hours . Betsy can ' t pick him up .
A 12 13|||R:CorefE|||Johnny|||REQUIRED|||-NONE-|||0

For more information, please see the repository of ERRANT.

# Installation

## Source Install

If you prefer to install FERRANTI from source, you can instead run the following commands:
```
git clone https://github.com/kite99520/DialSummFactCorr.git
cd DialSummFactCorr/errant
python3 -m venv errant_env
source errant_env/bin/activate
pip3 install -U pip setuptools wheel
pip3 install -e .
python3 -m spacy download en
```

# Usage

## CLI

Three main commands are provided with FERRANTI: `errant_parallel`, `errant_m2` and `errant_compare`. You can run them from anywhere on the command line without having to invoke a specific python script.  

1. `errant_parallel`  

     This is the main annotation command that takes an original text file and at least one parallel corrected text file as input, and outputs an annotated M2 file. By default, it is assumed that the original and corrected text files are word tokenised with one sentence per line.  
	 Example:
	 ```
	 errant_parallel -orig <orig_file> -cor <cor_file1> [<cor_file2> ...] -out <out_m2>
	 ```

2. `errant_m2`  

     This is a variant of `errant_parallel` that operates on an M2 file instead of parallel text files. This makes it easier to reprocess existing M2 files. You must also specify whether you want to use gold or auto edits; i.e. `-gold` will only classify the existing edits, while `-auto` will extract and classify automatic edits. In both settings, uncorrected edits and noops are preserved.  
     Example:
	 ```
	 errant_m2 {-auto|-gold} m2_file -out <out_m2>
	 ```

3. `errant_compare`  

     This is the evaluation command that compares a hypothesis M2 file against a reference M2 file. The default behaviour evaluates the hypothesis overall in terms of span-based correction. The `-cat {1,2,3}` flag can be used to evaluate error types at increasing levels of granularity, while the `-ds` or `-dt` flag can be used to evaluate in terms of span-based or token-based detection (i.e. ignoring the correction). All scores are presented in terms of Precision, Recall and F-score (default: F0.5), and counts for True Positives (TP), False Positives (FP) and False Negatives (FN) are also shown.  
	 Examples:
	 ```
     errant_compare -hyp <hyp_m2> -ref <ref_m2> 
     errant_compare -hyp <hyp_m2> -ref <ref_m2> -cat {1,2,3}
     errant_compare -hyp <hyp_m2> -ref <ref_m2> -ds
     errant_compare -hyp <hyp_m2> -ref <ref_m2> -ds -cat {1,2,3}
	 ```	

All these scripts also have additional advanced command line options which can be displayed using the `-h` flag. 

