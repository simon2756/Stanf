# Project Plan: LLMs from first principles to production (Swiss finance focus)

A hands-on portfolio that follows Stanford CME 295 (Transformers & Large Language Models, Autumn 2026).
Each lecture gets a small, self-contained project. All of them build toward one flagship system for a
real problem that Swiss banks, insurers and data companies care about.

> **Positioning:** *"I can build, adapt, evaluate and run LLM systems for regulated, multilingual
> environments, and I understand them down to the attention kernel."*

---

## 1. Why this angle works for Swiss data and finance employers

Swiss employers (UBS, Julius Bär, Vontobel, Pictet, LGT, Swiss Re, Zurich, Swiss Life, SIX,
PostFinance, cantonal banks, and data consultancies) all have the same constraints. The portfolio
answers each one directly:

| Their constraint | What the portfolio shows |
|---|---|
| **Data residency and confidentiality** (banking secrecy, revised FADP/nDSG) | Open-weight models that run locally or on-prem, no client data sent to third-party APIs |
| **Regulation** (FINMA Guidance 08/2024 on AI governance; EU AI Act for EU-facing business) | Audit logs, reproducible evaluations, model cards, documented risks and limits |
| **Multilingual documents** (DE / FR / IT / EN, plus Swiss German in client contact) | Tokenizer analysis, multilingual evaluation, Swiss open model (Apertus) as a baseline |
| **Numbers have to be right** | Verifiable-reward RL on financial reasoning, numeric accuracy metrics, citations to sources |
| **Cost and latency** | Measured serving benchmarks (quantization, speculative decoding, batching) |

**Rule:** use only public data. Never use data from your employer (Rolls-Royce Power Systems / MTU) or
any other confidential source in this repository.

---

## 2. Overview: projects mapped to lectures

| # | Lecture | Project | Core deliverable | Effort |
|---|---|---|---|---|
| P1 | L1 Transformers | **Swiss-multilingual mini-GPT from scratch** | BPE tokenizer + modern decoder trained on a DE/FR/IT/EN corpus | 2 weeks |
| P2 | L2 LLMs | **LLM internals lab** | GQA/RoPE/MoE/sampling implemented; real open-weight model loaded into your own code with matching logits | 2 weeks |
| P3 | L3 Training | **Domain adaptation: financial document extraction** | LoRA SFT, then DPO, then distillation of a small model that turns annual reports into validated JSON | 3 weeks |
| P4 | L4 RL | **RLVR for financial numeric reasoning** | GRPO-family training with verifiable rewards on FinQA-style tasks, plus a reward-hacking analysis | 2–3 weeks |
| P5 | L5 Systems | **Inference performance lab** | KV cache and speculative decoding from scratch; vLLM/SGLang benchmark; one Triton kernel | 2 weeks |
| P6 | L6 Agents | **Compliant research agent with MCP** | MCP servers (SNB data, document search, Python sandbox), RAG, memory, compaction, audit log | 3 weeks |
| P7 | L7 Evaluation | **Evaluation harness** | LLM-as-a-judge calibrated against human labels, agent trajectory evals, CI regression gate | 2 weeks |
| P8 | L8 Diffusion | **Masked diffusion LM vs autoregressive** | Small MDLM-style model trained on the P1 data, compared with the P1 GPT | 1–2 weeks |
| P9 | L9 Multimodal | **Charts and tables in PDFs** | Visual document retrieval (ColPali-style) + VLM extraction, plugged into the agent | 1–2 weeks |
| ★ | All | **Flagship: HelvetiQ** (working title) | Sovereign, multilingual financial document intelligence system with a public demo | 4–6 weeks |

Rough total at **6–10 h/week** next to work: about 6 months. P1–P2 run on a laptop or free GPUs;
P3, P4 and the flagship need a few paid GPU hours (see §6).

---

## 3. Shared engineering standard (applies to every project)

Recruiters and senior engineers open the repository before they read your CV. The engineering quality
counts as much as the ML.

- **Monorepo layout**
  ```
  stanf/
  ├── PROJECT_PLAN.md
  ├── README.md                 # portfolio landing page with result tables and links
  ├── pyproject.toml            # uv workspace
  ├── packages/
  │   ├── core/                 # shared model code (attention, RoPE, sampling, ...)
  │   └── evals/                # shared evaluation utilities
  ├── projects/
  │   ├── p1_mini_gpt/
  │   ├── p2_internals/
  │   ├── ...
  │   └── flagship_helvetiq/
  └── .github/workflows/        # lint, type-check, tests, eval regression
  ```
- **Tooling:** `uv` for environments and lockfiles, `ruff` (lint and format), `pyright` or `mypy`,
  `pytest`, `pre-commit`, GitHub Actions.
- **Configs and tracking:** typed configs (`pydantic-settings` or Hydra), experiment tracking with
  Weights & Biases or MLflow, fixed seeds, recorded hardware and runtimes.
- **Every project README follows the same template:** problem, approach, results table, what did not
  work, what I learned, how to reproduce (one command).
- **Artifacts:** models and datasets on the Hugging Face Hub, each with a model or dataset card that
  covers intended use and limitations.
- **Tests:** unit tests for the maths (e.g. "RoPE preserves the norm", "the causal mask blocks
  future tokens", "KV-cache decoding equals full recompute").

---

## 4. The projects in detail

### P1: Swiss-multilingual mini-GPT from scratch (Lecture 1)

**Goal:** understand every line of a modern transformer by writing and training one.

**Build**
1. **Tokenizer:** a byte-level BPE trainer written by you (then compared with the Hugging Face
   `tokenizers` library). Train it on a mixed DE/FR/IT/EN corpus, e.g. a FineWeb-2 subset for DE/FR/IT,
   FineWeb-Edu for EN, and Swiss sources such as SNB press releases or admin.ch texts.
2. **Tokenizer study (good portfolio material):** measure *fertility* (tokens per word) and
   compression across languages for your tokenizer, GPT-style tokenizers, Qwen, Llama and Apertus.
   Question: *how much more does it cost to process French or Italian text than English?*
3. **Embeddings warm-up:** a short word2vec (skip-gram with negative sampling) notebook, then the
   question of why static embeddings fall short and attention is needed.
4. **Model:** a decoder-only transformer with the **current standard design**, not the 2017 original:
   - pre-norm with **RMSNorm**
   - **SwiGLU** MLP
   - **RoPE** position embeddings
   - **QK-norm** for training stability
   - weight tying, no biases
5. **Training:** a ~10–50M parameter model, AdamW with warmup and cosine or WSD
   (warmup-stable-decay) schedule, bf16 mixed precision, `torch.compile`, gradient accumulation.
   Bonus: try the **Muon** optimizer (popularised by the modded-nanoGPT speedrun) and compare loss
   curves with AdamW.

**Results to report:** loss curves, tokenizer fertility table, generated samples in all four languages,
and an ablation (e.g. RoPE vs learned absolute positions, or with and without QK-norm).

**Skills:** PyTorch internals, tokenization, attention, training loops, mixed precision.

---

### P2: LLM internals lab (Lecture 2)

**Goal:** implement the architectural choices of current LLMs and *prove* they are correct.

**Build**
1. **Attention variants:** MHA, MQA, GQA in one module with a `num_kv_heads` parameter. Measure KV
   cache memory against sequence length for each.
2. **Multi-head Latent Attention (MLA, as in DeepSeek):** a simplified version that compresses KV into
   a latent vector. Compare cache size with GQA.
3. **Position embeddings:** RoPE plus context extension through **position interpolation, NTK-aware
   scaling and YaRN**. Experiment: train at context 512, evaluate perplexity at 2k/4k with each method.
4. **Mixture of Experts:** a top-k routed MoE layer with **shared experts** and load balancing, using
   both an auxiliary loss and the **auxiliary-loss-free bias method**. Plot expert utilisation.
5. **Sampling:** greedy, temperature, top-k, top-p, **min-p**, repetition penalty, and constrained
   decoding with a JSON schema (logit masking). An interactive notebook shows how the next-token
   distribution changes.
6. **Showpiece:** load the weights of a real small open model (e.g. a Qwen3 0.6B–1.7B class model, or
   pick the newest small open model when you start) **into your own implementation**, and show that
   your logits match the Hugging Face reference to within floating-point tolerance.

**Results to report:** KV-cache memory chart (MHA/GQA/MQA/MLA), context-extension perplexity table,
expert load plots, and a logit-equivalence test that runs in CI.

**Skills:** reading model code and papers, numerical debugging, modern architecture variants.

---

### P3: Domain adaptation, financial document extraction (Lecture 3)

**Goal:** turn a small open model into a reliable specialist and measure every training stage.

**Task:** given a page or section of a Swiss company's annual report (PDF, DE/FR/EN), extract key
figures (revenue, net profit, CET1 ratio, AuM, combined ratio, ...) into a **Pydantic-validated JSON
schema** with source references, and answer short questions about it.

**Data**
- Public annual reports of SMI and SPI companies, parsed with **Docling** (layout-aware PDF to
  Markdown/JSON).
- Public benchmarks for a sanity check: FinQA, TAT-QA, ConvFinQA, FinanceBench.
- **Synthetic data:** a large teacher model generates candidate labels; you hand-check a held-out set
  of about 200 examples. That human-labelled set is your gold standard and reused in P7.

**Training stages (each one is an experiment with a before/after table)**
1. **Baseline:** zero-shot and few-shot with the base model.
2. **SFT with LoRA / QLoRA** (Hugging Face TRL + PEFT, or Unsloth for speed). Follow current
   practice: LoRA on *all* linear layers including the MLP, a higher learning rate than full
   fine-tuning, and a rank sweep (8/32/128).
3. **Preference tuning with DPO:** build chosen/rejected pairs from model outputs that pass or fail
   schema validation and numeric checks.
4. **Distillation:** distil the large teacher into a 1–2B student, comparing
   - offline distillation (SFT on teacher outputs)
   - **on-policy distillation** (the student generates, the teacher scores per token with reverse KL,
     e.g. TRL's GKD trainer or your own loop).
5. **Base model choice:** compare a Qwen3-class model with **Apertus** (the Swiss open LLM from
   EPFL/ETH Zurich/CSCS). A Swiss sovereign model is a strong talking point in interviews.

**Results to report:** a table with exact-match / F1 per field × training stage × model, schema
validity rate, cost (GPU hours) per stage.

**Skills:** SFT, LoRA/QLoRA, DPO, synthetic data generation, distillation, data quality work.

---

### P4: RLVR for financial numeric reasoning (Lecture 4)

**Goal:** apply reinforcement learning with **verifiable rewards** (the method behind current
reasoning models) to a task where correctness can be checked automatically.

**Build**
1. **Environment:** FinQA/ConvFinQA-style questions plus your own generated Swiss-flavoured tasks
   (interest calculations, CHF/EUR conversions with SNB rates, percentage changes from report tables).
   The answer can be verified numerically within a tolerance.
2. **Reward design:** correctness (with tolerance), format reward (reasoning followed by a final
   answer tag), and a length penalty. Document every reward function and why it is there.
3. **Algorithms:** start with **GRPO** (TRL `GRPOTrainer`, or a minimal implementation you write),
   then compare at least one recent variant:
   - **Dr. GRPO** (removes the length and std normalisation biases)
   - **DAPO** (clip-higher, dynamic sampling, token-level loss)
   - **GSPO** (sequence-level importance ratios)
4. **Reward hacking study:** deliberately weaken a reward (e.g. only check the format) and document
   how the model exploits it. Employers value people who understand where RL goes wrong.
5. **Small PPO/RLHF demo:** a reward model trained on preference pairs from P3, then PPO for a few
   steps, mainly to explain why GRPO drops the critic.

**Results to report:** accuracy vs training steps, response length over training, pass@k before and
after RL, examples of reward hacking.

**Skills:** policy gradients, GRPO family, reward design, RL debugging, vLLM-backed rollouts.

---

### P5: Inference performance lab (Lecture 5)

**Goal:** know what makes LLM inference fast and cheap, and back every claim with measurements.

**Build (from scratch, on your P1/P2 code)**
1. **KV cache** incremental decoding, with a test that it matches full recompute exactly.
2. **Speculative decoding:** draft model (the P1 model or a tiny Qwen) plus target model, with
   acceptance rate and speed-up measured. Read about EAGLE-style draft heads and Medusa.
3. **One custom kernel in Triton:** fused RMSNorm or fused softmax, benchmarked against PyTorch eager
   and `torch.compile`.
4. **FlashAttention:** write a tiled, online-softmax attention in plain PyTorch to understand the
   algorithm, then benchmark it against PyTorch SDPA (which uses FlashAttention kernels).

**Benchmark (production stack)**
- Serve the P3 model with **vLLM** and **SGLang**. Measure throughput, time-to-first-token and
  inter-token latency under load, comparing:
  - BF16 vs **FP8** vs 4-bit (AWQ/GPTQ) quantization, plus the quality impact on the P3 gold set
  - prefix caching on/off (realistic for RAG with shared system prompts)
  - speculative decoding on/off
- Short write-up: **cost per 1M tokens on a rented GPU vs a commercial API**, and the break-even point.
  This is exactly the question a bank's IT architecture team asks.

**Skills:** profiling, GPU memory reasoning, Triton, serving engines, quantization, cost modelling.

---

### P6: Compliant research agent with MCP (Lecture 6)

**Goal:** an agent that answers analyst questions about Swiss financial documents and data with
tools, sources and a full audit trail.

**Build**
1. **MCP servers** (Model Context Protocol, the open standard for connecting tools to models):
   - `snb-data`: fetches time series (rates, FX, monetary aggregates) from the SNB data portal
   - `doc-search`: hybrid retrieval over the annual-report corpus
   - `python-sandbox`: isolated code execution for calculations and charts
2. **Retrieval, built properly:**
   - Docling parsing with structure-aware chunking
   - **hybrid search** (BM25 + dense embeddings with a multilingual model) and a **cross-encoder
     reranker**
   - **contextual retrieval**: prepend an LLM-generated context sentence to each chunk before
     embedding
   - citations required: every number in an answer must link to a chunk
3. **Agent harness:** a tool-calling loop (a lightweight framework such as PydanticAI or LangGraph, or
   your own loop) with:
   - **memory** (session notes plus long-term facts)
   - **context compaction** (summarising old turns when the context fills up)
   - **skills**: reusable instruction bundles, e.g. "how to analyse a bank's capital ratios"
   - **guardrails**: input/output checks, a PII filter, refusal of investment advice
4. **Governance layer:** structured audit log of every prompt, tool call and source (OpenTelemetry
   traces in Langfuse or MLflow), plus a model selector that switches between a local open model and
   an API model per data sensitivity level.
5. **Coding-agent practice:** use a coding agent for parts of this project and write a short honest
   note on where it helped and where you had to correct it. That is a real skill question in
   interviews now.

**Skills:** MCP, RAG, tool calling, agent design, observability, AI governance.

---

### P7: Evaluation harness (Lecture 7)

**Goal:** make quality measurable, so every change to P3–P6 shows up as a number.

**Build**
1. **Gold sets:** the human-checked P3 set plus about 100 hand-written agent tasks with expected
   tools and answers, in DE/FR/EN.
2. **LLM-as-a-judge, done carefully:**
   - rubric-based grading with reference answers
   - mitigate **position bias** (swap order and average), **verbosity bias** (length-controlled
     comparison) and **self-preference bias** (judge from a different model family)
   - **calibrate against your own human labels** and report agreement (Cohen's κ / Spearman)
3. **Agent evaluation:** task success, tool-call accuracy, number of steps, cost, and multi-turn
   consistency (re-run each task several times, report pass^k in the style of τ-bench).
4. **Framework:** build it on **Inspect AI** (the UK AISI eval framework) or a lightweight custom
   runner; also run a public benchmark slice through `lm-evaluation-harness` for comparison.
5. **CI regression gate:** a small eval runs on each pull request and fails if key metrics drop.

**Skills:** experimental design, statistics (confidence intervals, bootstrap), judge calibration,
benchmark literacy.

---

### P8: Masked diffusion LM vs autoregressive (Lecture 8)

**Goal:** understand the newest alternative to left-to-right generation.

**Build**
1. A small **masked diffusion language model** (MDLM / LLaDA-style objective: random masking ratio,
   predict masked tokens, iterative unmasking at inference) on the **same data and similar parameter
   count as P1**.
2. Compare with the P1 GPT on perplexity bound, sample quality and generation speed as a function of
   the number of denoising steps.
3. Run inference with a public diffusion LLM (e.g. LLaDA or Dream) and test **parallel decoding** and
   infilling, which autoregressive models cannot do naturally.

**Skills:** generative modelling beyond AR, research-style comparison under equal compute.

---

### P9: Charts and tables in PDFs (Lecture 9)

**Goal:** handle what text-only pipelines miss in financial reports.

**Build**
1. **Visual document retrieval** with ColPali/ColQwen-style late-interaction models: embed page
   images directly, compare with the text-based retrieval from P6.
2. **VLM extraction:** a small open vision-language model (Qwen-VL class) reads charts and complex
   tables into structured data; evaluate on a hand-labelled set of 50 pages.
3. Add this as an extra MCP tool to the P6 agent.

**Skills:** multimodal models, document AI.

---

## 5. Flagship: HelvetiQ (working title)

**One sentence:** *a sovereign, multilingual assistant that answers questions about Swiss listed
companies and macro data with verified numbers, sources and an audit trail, running on open models.*

**Brings together:** the P3 extraction model (distilled, quantized), P4 numeric reasoning, P5
serving, the P6 agent and MCP tools, the P7 evaluation and P9 visual retrieval.

**Example questions it should handle**
- "Compare UBS's and Julius Bär's CET1 ratio over the last three years and explain the trend."
- "Wie hat sich der SNB-Leitzins seit 2022 entwickelt, und wie hat der EUR/CHF darauf reagiert?"
- "Quels risques climatiques Swiss Re mentionne-t-il dans son dernier rapport annuel ?"

**Deliverables**
- A public demo (Hugging Face Spaces or a small cloud VM) with a 2-minute video
- An architecture diagram and a one-page **model risk note** in the style of a bank's model
  governance documentation: intended use, limitations, evaluation results, monitoring plan
- A technical blog post (Medium, Substack or a personal site) and a LinkedIn post with results
- An eval dashboard showing quality, latency and cost per question

---

## 6. Compute and budget

| Stage | Hardware | Options |
|---|---|---|
| P1, P2, P8 | Laptop / free GPU | Kaggle (free weekly GPU quota), Google Colab |
| P3, P4 | 1× 24–80 GB GPU for a few hours per experiment | RunPod, Lambda, Vast.ai, Modal; Unsloth plus QLoRA keeps 4–8B models on 24 GB |
| P5 | 1× modern GPU for benchmarks | Same providers, H100 or L40S for FP8 |
| Flagship demo | CPU or small GPU | Quantized 1–4B model, or a hosted endpoint |

Plan for roughly **CHF 100–250 in total** if you shut instances down and run small sweeps first. Check
for student credit programmes (cloud providers, Hugging Face, Modal) before paying. GPU prices change
quickly, so compare at the time you rent.

---

## 7. Timeline (aligned with the lecture schedule)

| Weeks | Dates (2026/27) | Work |
|---|---|---|
| 0 | Oct 21–26 | Repo setup (uv, ruff, CI, README template). Midterm on Oct 23 takes priority. |
| 1–2 | Oct 27–Nov 9 | **P1** mini-GPT plus tokenizer study. Watch L3 and L4 (already released). |
| 3–4 | Nov 10–23 | **P2** internals lab |
| 5–7 | Nov 24–Dec 14 | **P3** domain adaptation (start data collection for P6/P7 alongside) |
| 8–9 | Dec 15–Jan 4 | **P4** RLVR (holiday period, good for longer GPU runs) |
| 10–11 | Jan 5–18 | **P5** inference lab |
| 12–14 | Jan 19–Feb 8 | **P6** agent with MCP |
| 15–16 | Feb 9–22 | **P7** evaluation harness |
| 17 | Feb 23–Mar 1 | **P8** diffusion (light) |
| 18 | Mar 2–8 | **P9** multimodal |
| 19–24 | Mar 9–Apr 19 | **Flagship** integration, demo, write-ups, applications |

If time gets tight, the minimum viable portfolio is **P1 + P3 + P6 + P7 + a small flagship demo**.
That already covers "built from scratch", "fine-tuned", "built an agent" and "evaluated properly".
P2, P4 and P5 then add depth.

---

## 8. Skill matrix

| Skill area | P1 | P2 | P3 | P4 | P5 | P6 | P7 | P8 | P9 | ★ |
|---|---|---|---|---|---|---|---|---|---|---|
| PyTorch / model internals | ● | ● | | ○ | ● | | | ● | | |
| Modern architectures (GQA, MLA, RoPE/YaRN, MoE) | ○ | ● | | | ○ | | | | | |
| Fine-tuning (LoRA, DPO, distillation) | | | ● | ○ | | | | | | ● |
| RL for LLMs (GRPO family, reward design) | | | | ● | | | | | | ○ |
| Inference systems (vLLM, quantization, kernels) | | ○ | | ○ | ● | | | | | ● |
| RAG and agents (MCP, memory, compaction) | | | | | | ● | | | ● | ● |
| Evaluation and statistics | ○ | ○ | ● | ● | ● | ○ | ● | ● | ○ | ● |
| MLOps and engineering (CI, tracking, cards) | ● | ● | ● | ● | ● | ● | ● | ○ | ○ | ● |
| Governance and risk (audit, model risk notes) | | | ○ | ○ | | ● | ● | | | ● |
| Multilingual / Swiss context | ● | | ● | ○ | | ● | ● | | ○ | ● |

● main focus · ○ secondary

---

## 9. Recent methods covered (and where)

Methods that are current as of 2025/26, as opposed to textbook-only material:

- **Architecture:** RMSNorm + SwiGLU + RoPE + QK-norm (P1), GQA and **MLA** (P2), **MoE with shared
  experts and aux-loss-free balancing** (P2), YaRN context extension (P2)
- **Optimisation:** WSD schedules, **Muon** optimizer (P1)
- **Sampling:** **min-p**, constrained/structured decoding (P2)
- **Post-training:** LoRA on all layers with rank sweeps, DPO, **on-policy distillation** (P3)
- **RL:** RLVR with **GRPO, Dr. GRPO, DAPO, GSPO**, reward-hacking analysis (P4)
- **Systems:** FP8 quantization, prefix caching, speculative decoding, Triton kernels, FlashAttention
  algorithm (P5)
- **Agents:** **MCP**, skills, context compaction, contextual retrieval, hybrid search + reranking,
  OpenTelemetry tracing (P6)
- **Evaluation:** calibrated LLM-as-a-judge, pass^k agent metrics, Inspect AI, CI eval gates (P7)
- **New paradigms:** masked diffusion LMs (P8), late-interaction visual retrieval (P9)
- **Swiss context:** Apertus, FINMA AI guidance, multilingual DE/FR/IT evaluation (throughout)

The field moves fast. At the start of each project, spend 30 minutes checking for newer small open
models and library releases, and note in the README which versions you used and why.

---

## 10. Turning this into applications

- **CV bullets with numbers**, for example:
  - "Distilled a 1.5B model for multilingual financial-report extraction: 91% field-level F1 vs 94%
    for an 8× larger teacher, at 6× lower serving cost" (fill in your real results)
  - "Implemented GRPO and DAPO for numeric reasoning; +18 pp accuracy on a FinQA-style benchmark"
- **Pinned GitHub repos:** this monorepo plus the flagship. The top-level README is a short landing
  page with the result tables and the demo video.
- **Interview stories:** for each project, prepare one *"what went wrong and how I found it"* story
  (e.g. a RoPE bug found through the logit-equivalence test, reward hacking in P4).
- **Swiss specifics:** mention FINMA, the FADP, data residency and multilingual handling in cover
  letters. Those topics show you understand their business, not just the models.
- **Connect it to your current role:** you already work on data science and AI in an industrial
  setting. Present this portfolio as the LLM-specialist layer on top of that experience, without
  sharing anything confidential from work.

---

## 11. Progress tracker

- [ ] Week 0: repo, tooling, CI, README template
- [ ] P1 Swiss-multilingual mini-GPT
- [ ] P2 LLM internals lab
- [ ] P3 Domain adaptation
- [ ] P4 RLVR
- [ ] P5 Inference lab
- [ ] P6 Agent with MCP
- [ ] P7 Evaluation harness
- [ ] P8 Diffusion LM
- [ ] P9 Multimodal documents
- [ ] Flagship HelvetiQ: demo, model risk note, blog post
