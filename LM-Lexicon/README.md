🤖 📚 LM-Lexicon-Bench---
WIP 🚧
---
# Available Lexicon Resources
## Datasets

- **3D-EX (2023)**: [[Paper](https://aclanthology.org/2023.ranlp-1.8/)] [[GitHub](https://github.com/F-Almeman/3D-EX?tab=readme-ov-file)] [[Data](https://drive.google.com/uc?export=download&id=1gnRFRKISVxLVGhwpOWg6ZfjYNdW6Nad-)]
- **WordNet-Oxford-Urban-Wikipedia (2021)**: [[Paper](https://aclanthology.org/2021.emnlp-main.194/)]  [[GitHub](https://github.com/amanotaiga/Definition_Modeling_Project)] [[Data](http://www.tkl.iis.u-tokyo.ac.jp/~ishiwatari/naacl_data.zip)]
- :wrench: _**TODO**: Find more datasets' link._

## Eval-tools

- :wrench: _**TODO**: Add Metrics eval tool._

# Foundation Models

| Model         | Architecture | # Parameters |
|---------------|--------------|--------------|
| T5-small      | Enc-Dec      | 60M          |
| T5-base       | Enc-Dec      | 220M         |
| T5-large      | Enc-Dec      | 770M         |
| Flan-T5-small | Enc-Dec      | 80M          |
| Flan-T5-Base  | Enc-Dec      | 250M         |
| Flan-T5-Large | Enc-Dec      | 780M         |
| Flan-T5-XL    | Enc-Dec      | 3B           |
| Flan-T5-xxl   | Enc-Dec      | 11B          |
| Phi-2         | Dec.         | 2.7B         |
| Gemma-7B      | Dec.         | 7B           |
| Llama-2-7B    | Dec.         | 7B           |
| Llama-2-13B   | Dec.         | 13B          |
| Llama-2-70B   | Dec.         | 70B          |
| Mistral-7B    | Dec.         | 7B           |
| Mixtral-8x7B  | Dec.         | 46.7B        |

# Related Work
## Survey

- Definition modeling: literature review and dataset analysis (Applied Computing and Intelligence 2022)

## Dataset/Benchmark

- **3D-EX: A Unified Dataset of Definitions and Dictionary Examples (RANLP 2023)**
- COMPILING: A Benchmark Dataset for Chinese Complexity Controllable Definition Generation (CCL 2022)
- Graphine: A Dataset for Graph-aware Terminology Definition Generation (EMNLP 2021)
- JADE: Corpus for Japanese Definition Modelling (LREC 2022)
- DORE: DORE: A Dataset For Portuguese Definition Generation (LREC-COLING 2024)
- :wrench: _**TODO**: Add Datasets' papers._

## Metrics 

- :wrench: _**TODO**: Add evaluation metrics tool's paper._

- BLEU [[paper](https://aclanthology.org/P02-1040)]
- Perpexity
- ROUGE-L
- NIST
- METEOR [[paper](https://aclanthology.org/W05-0909/)]
- BERT-SCORE
- Cosine-Similarity
- Human-Eval (Win rate)

## Definition Modeling

- Beyond Perplexity: Examining Temporal Generalization in Large Language Models via Definition Generation (Computational Linguistics in the Netherlands Journal, 2024)
- Lexical Semantics with Large Language Models: A Case Study of English "break" (EACL 2023, Findings)
- Fantastic Semantics and Where to Find Them: Investigating Which Layers of Generative LLMs Reflect Lexical Semantics （Arxiv 2024）
- MISGENDERED: Limits of Large Language Models in Understanding Pronouns (ACL 2023)
- **Interpretable Word Sense Representations via Definition Generation: The Case of Semantic Change Analysis (ACL 2023)**
- **Definition Modelling for Appropriate Specificity (EMNLP 2021)**
- **Word Definitions from Large Language Models (Arxiv 2023 | ACL 2024 filed)**
- Evaluating Large Language Models' Understanding of Financial Terminology via Definition Modeling (IJCNLP 2023)
- Explicit Semantic Decomposition for Definition Generation (ACL 2020)
- Probing Pretrained Language Models for Lexical Semantics (EMNLP 2020)
- Fine-grained Contrastive Learning for Definition Generation (AACL 2022)
- Exploiting Correlations Between Contexts and Definitions with Multiple Definition Modeling (Arxiv 2023)
- Multitasking Framework for Unsupervised Simple Definition Generation (ACL 2022)
- Decompose, Fuse and Generate: A Formation-Informed Method for Chinese Definition Generation (NAACL 2021)
- Understanding Jargon: Combining Extraction and Generation for Definition Modeling (EMNLP 2022)
- Toward Cross-Lingual Definition Generation for Language Learners (Arxiv 2020)
- “Definition Modeling : To model definitions.” Generating Definitions With Little to No Semantics (IWCS 2023)
- Auto-Encoding Dictionary Definitions into Consistent Word Embeddings (EMNLP 2018)
- Generationary or “How We Went beyond Word Sense Inventories and Learned to Gloss” (EMNLP 2020)
- What Does This Word Mean? Explaining Contextualized Embeddings with Natural Language Definition (EMNLP | IJCNLP 2019)
- Conditional Generators of Words Definitions (ACL 2018)
- Combining Extraction and Generation for Definition Modeling (EMNLP 2022)
- Learning to Describe Unknown Phrases with Local and Global Contexts (NAACL 2019)
- Evaluating a Multi-sense Definition Generation Model for Multiple Languages (Text, Speech, and Dialogue: 23rd International Conference 2020)
- Mark my Word: A Sequence-to-Sequence Approach to Definition Modeling (Proceedings of the First NLPL Workshop on Deep Learning for Natural Language Processing 2019)
- Definition Modeling: Learning to define word embeddings in natural language (arxiv 2017)
- VCDM: Leveraging Variational Bi-encoding and Deep Contextualized Word Representations for Improved Definition Modeling (EMNLP 2020)
- Bridging the Defined and the Defining: Exploiting Implicit Lexical Semantic Relations in Definition Modeling (EMNLP-IJCNLP 2019)
- Learning to Explain Non-Standard English Words and Phrases (IJCNLP 2017)
- Multi-sense Definition Modeling using Word Sense Decompositions (Arxiv 2019)

## In-Context Learning
- Larger language models do in-context learning differently (ICLR 2024, Withdrawn Submission)
- Rethinking the Role of Demonstrations: What Makes In-Context Learning Work? (EMNLP 2022)
- In-Context Learning Creates Task Vectors (EMNLP 2023)
- Pre-Training to Learn in Context (ACL 2023)
- Understanding In-Context Learning via Supportive Pretraining Data (ACL 2023)

## Instruction Tuning
- TaxoLLaMA: WordNet-based Model for Solving Multiple Lexical Sematic Tasks (Arxiv 2024)
- Finetuned Language Models are Zero-Shot Learners (ICLR 2022, oral)
- The Flan Collection: Designing Data and Methods for Effective Instruction Tuning (ICML 2023)

## LM Distillation
- LLM2LLM: Boosting LLMs with Novel Iterative Data Enhancement
- UniversalNER: Targeted Distillation from Large Language Models for Open Named Entity Recognition
- Distilling Step-by-Step! Outperforming Larger Language Models with Less Training Data and Smaller Model Sizes
- Self-Distillation Bridges Distribution Gap in Language Model Fine-Tuning

## LLMs-as-Evaluators
- G-Eval: NLG Evaluation using Gpt-4 with Better Human Alignment (EMNLP 2023)
- GPTScore: Evaluate as You Desire (Arxiv 2023)
- LLM-Eval: Unified Multi-Dimensional Automatic Evaluation for Open-Domain Conversations with Large Language Models (NLP4ConvAI Workshop 2023)
- AlignScore: Evaluating Factual Consistency with A Unified Alignment Function (ACL 2023)
- A Survey on Evaluation of Large Language Models (TIST 2024)

## LM Interpretability
- What In-Context Learning "Learns" In-Context: Disentangling Task Recognition and Task Learning (ACL 2023, Findings)
- Towards Understanding Chain-of-Thought Prompting: An Empirical Study of What Matters (ACL 2023)
- Analyzing Transformers in Embedding Space (ACL 2023)
- Impact of Co-occurrence on Factual Knowledge of Large Language Models (EMNLP 2023, Findings)
- Label Words are Anchors: An Information Flow Perspective for Understanding In-Context Learning (EMNLP 2023)
- Dissecting Recall of Factual Associations in Auto-Regressive Language Models (EMNLP 2023)
- A Mechanistic Interpretation of Arithmetic Reasoning in Language Models using Causal Mediation Analysis (EMNLP 2023)
- Transformer Feed-Forward Layers Build Predictions by Promoting Concepts in the Vocabulary Space (EMNLP 2022)
- Rethinking the Role of Demonstrations: What Makes In-Context Learning Work? (EMNLP 2022)
- Transformer Feed-Forward Layers Are Key-Value Memories (EMNLP 2021)
- Are Emergent Abilities of Large Language Models a Mirage? (NeurIPS 2023)
- How does GPT-2 compute greater-than?: Interpreting mathematical abilities in a pre-trained language model (NeurIPS 2023)
- Does Localization Inform Editing? Surprising Differences in Causality-Based Localization vs. Knowledge Editing in Language Models (NeurIPS 2023, Spotlight)
- Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small (ICLR 2023)
- Mass-Editing Memory in a Transformer (ICLR 2023)
- Locating and Editing Factual Associations in GPT (NeurIPS 2022)
- Towards a Unified View of Parameter-Efficient Transfer Learning (ICLR 2022)
- Progress measures for grokking via mechanistic interpretability (ICLR 2023)
- Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets (Arxiv 2022)
