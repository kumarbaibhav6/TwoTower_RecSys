# PROJECT PLAN: Distributed Two-Tower Recommendation System

> **For the agent picking this up:** this file holds (A) background, (B) the original plan exactly as written,
> (C) the review of that plan and why it changed, (D) the **revised plan to execute**, and (E) the Azure budget plan.
> **Execute Section D, not Section B.** Do one phase at a time, run its verification step, and don't start the next
> phase until verification passes. Tick the checkboxes `[x]` in this file as you go, and record anything that
> deviates from the plan under "Implementation notes" at the bottom.

---

## A. Background

- **Owner:** an AI Engineer (about 1.5 years: production RAG on Azure, eval frameworks, document automation)
  targeting **AI Engineer / Applied LLM Engineer** and **ML Engineer** roles. This is a portfolio project.
- **The gap it fills on the resume:** training a deep model yourself, distributed training, offline ranking evaluation,
  ANN retrieval at scale. The companion project (vLLM serving benchmark,
  github.com/kumarbaibhav6/LLM_Inference_Project) covers LLM inference. Together: *serving LLMs* plus
  *training embeddings and retrieving with them at scale*.
- **Compute available:** free Colab (T4/L4), free Kaggle notebooks (**2× T4**, which makes real DDP possible),
  **$200 Azure credits** (see Section E).
- **Standard of work:** senior-MLE quality. Every claim needs a number, every number needs a baseline or control,
  and nothing goes on the resume unless it was actually run.

---

## B. Original plan (as written by the owner, kept for reference)

### 1. Purpose
The objective of this project is to build an end-to-end MLOps pipeline for a Two-Tower recommendation system using the Kaggle "H&M Personalized Fashion Recommendations" dataset (comprising 1.37M customers, 106K articles, and 31.8M transactions). This project bridges offline deep learning with real-time vector retrieval, demonstrating the ability to engineer large-scale candidate generation pipelines.

### 2. Learning Goals & Architectural Focus
*   **Shared Embedding Space:** Map high-cardinality user and item features into a unified 64-dimensional latent space.
*   **Implicit Feedback & Negative Sampling:** Train on historical transactions using in-batch negatives to optimize a contrastive loss function (InfoNCE).
*   **Distributed Training & MLOps:** Utilize PyTorch DataLoaders, integrate experiment tracking (MLflow/Weights & Biases), and structure the code for PyTorch Distributed Data Parallel (DDP).
*   **ANN Retrieval:** Export frozen Item Tower embeddings into a FAISS index to simulate real-world, millisecond-latency candidate generation.

### 3. Execution Phases & Action Items (original)

**Phase 1: Environment & Data Ingestion**
- Initialize project structure: `data/`, `src/`, `models/`, `notebooks/`.
- Write `requirements.txt` including: `torch`, `pandas`, `numpy`, `scikit-learn`, `faiss-cpu` (or `faiss-gpu`), `mlflow` (or `wandb`), `tqdm`.
- Create a data ingestion script `src/data_loader.py` to load `articles.csv`, `customers.csv`, and `transactions_train.csv`.
- Verification: Dependencies install successfully. The data loader successfully reads the CSVs (or a sampled subset for local testing) into Pandas DataFrames.

**Phase 2: Feature Engineering & Dataset Preparation**
- Filter dataset sparsity: Keep only users with >5 purchases and articles with >20 purchases to ensure embedding quality.
- Map categorical features to numerical IDs (Label Encoding) for PyTorch embedding layers.
  - User Features: `customer_id`, `age_bucket`, `club_member_status`.
  - Item Features: `article_id`, `product_group_name`, `colour_group_name`.
- Implement a custom PyTorch `Dataset` (`src/dataset.py`) that yields `(user_features, item_features)` pairs based on positive transactions.
- Verification: The PyTorch `DataLoader` outputs batched tensors of the correct shape without memory leaks.

**Phase 3: PyTorch Model Architecture (`src/model.py`)**
- Define `UserTower`: `nn.Embedding` layers for user features -> concatenated -> `nn.Sequential` (MLP) -> 64-dim output vector.
- Define `ItemTower`: `nn.Embedding` layers for item features -> concatenated -> `nn.Sequential` (MLP) -> 64-dim output vector.
- Define `TwoTowerModel`: Takes user and item inputs, passes them through their respective towers, and computes the dot product similarity.
- Verification: A dummy forward pass with random tensor inputs successfully outputs a tensor of similarity scores without shape mismatch errors.

**Phase 4: Training Pipeline with In-Batch Negatives**
- Write the training loop (`src/train.py`).
- Implement the loss function: Treat the diagonal of the batch similarity matrix as positive pairs, and all off-diagonal elements as negative pairs (Cross-Entropy / InfoNCE loss).
- Integrate MLflow or Weights & Biases to log training loss, learning rate, and epoch duration.
- Save the best model weights (`.pt` or `.pth`) based on validation loss.
- Verification: The model trains for at least 1 epoch, the loss decreases, and metrics are successfully logged to the tracking server.

**Phase 5: FAISS Vector Indexing & Online Serving**
- Write an extraction script (`src/export_embeddings.py`) that passes all unique items from `articles.csv` through the trained, frozen `ItemTower`.
- Build a FAISS Index (e.g., `IndexFlatIP` for Inner Product / Dot Product) and insert the item embeddings. Save the index to disk.
- Write an inference script (`src/serve.py`) that simulates an online request:
  - Input: A `customer_id`.
  - Action 1: Pass the user's features through the `UserTower` to get their 64-dim vector.
  - Action 2: Query the FAISS index with this vector to retrieve the Top-10 recommended `article_id`s.
- Verification: The inference script successfully executes in under 50 milliseconds and returns 10 valid article IDs for a given user.

---

## C. Review: what changed and why

| # | Gap in the original | Why it matters | Change |
|---|---|---|---|
| 1 | **No ranking evaluation.** Success means "loss decreases" | A falling InfoNCE loss doesn't mean good recommendations. This is the first thing an interviewer probes | Time-based split (test on the last week), **MAP@12** (the competition metric) and Recall@k, and **baselines** (recent popularity, repeat purchase). Build the evaluation harness *before* the model |
| 2 | Random split implied | Future purchases leak into training and inflate metrics | Train ≤ 2020-09-08, validate 09-09 → 09-15, test 09-16 → 09-22 (the last week, as in the competition) |
| 3 | **Popularity bias in in-batch negatives** | Popular items show up as negatives far more often, so the model learns to push them down | **logQ correction** (Yi et al., *Sampling-Bias-Corrected Neural Modeling*, RecSys 2019): subtract log(item frequency) from the logits |
| 4 | Accidental hits | The same article twice in a batch means a positive gets labelled as a negative | Mask duplicate item IDs off the diagonal |
| 5 | No temperature, no normalization | InfoNCE on raw dot products trains unstably | L2-normalize both towers' outputs, temperature τ ≈ 0.05–0.1 (tune or learn it) |
| 6 | **User tower dominated by the `customer_id` embedding** | That's a lookup table: no generalization and no cold-start | Add **purchase-history features** (mean of the last N item embeddings, recency, purchase count); treat `customer_id` embedding as an ablation |
| 7 | **"Structure for DDP"** | You can't claim DDP if you never ran it | Actually run it on **Kaggle 2× T4** (free) or Azure (Section E): `DistributedSampler`, **gather embeddings across GPUs so negatives cover the global batch**, save checkpoints from rank 0 only, report 1 → 2 GPU scaling |
| 8 | **Flat FAISS over 106K items, < 50 ms** | Exact search takes about 1 ms, so the target proves nothing | Compare **Flat vs. IVF vs. HNSW** with a recall-vs-latency curve; FastAPI endpoint with **p50/p99 under concurrent load** |
| 9 | Pandas on a 31.8M-row CSV | Default dtypes (object strings, int64) can exceed Colab's ~12 GB RAM | Convert to **Parquet once** with small integer codes, `category` columns and `float32` |
| 10 | Sparsity filters hide cold-start | Filtering users with ≤5 purchases and articles with ≤20 hides the hardest case | Report how many users/items each filter removes; evaluate cold-start items separately |
| 11 | Weak link to AI Engineer roles | Recommender systems alone don't read as LLM-adjacent | **Text embeddings of `detail_desc`** (sentence-transformer) as an item-tower feature, with an ablation on cold-start items. This connects to the owner's RAG/embedding background |
| 12 | Model selected on validation loss | Loss and ranking quality can disagree | Early-stop and select on **validation Recall@100 / MAP@12** |

---

## D. Revised plan (execute this)

### Phase 1: Environment, data and Parquet
- [x] Project structure: `data/` (gitignored), `src/`, `models/` (gitignored), `notebooks/`, `benchmarks/`, `docs/`.
- [x] `requirements.txt` with pinned versions: `torch`, `pandas`, `pyarrow`, `numpy`, `scikit-learn`, `faiss-cpu`, `mlflow`, `sentence-transformers`, `fastapi`, `uvicorn`, `httpx`, `tqdm`, `matplotlib`. (`faiss-gpu` is optional; CPU is fine for 106K items.)
- [x] Download **only the CSVs**. The full competition archive is ~30 GB because of the images:
      `kaggle competitions download -c h-and-m-personalized-fashion-recommendations -f transactions_train.csv` (and the same for `articles.csv` and `customers.csv`). The competition rules must be accepted on Kaggle first.
- [x] `src/data_loader.py`: read the CSVs once, then write Parquet with compact dtypes (`article_id` → int32, `customer_id` hash → int64 code, `price` → float32, categoricals → `category`). Support `--sample-frac` for local testing.
- [x] Print dataset statistics (rows, unique users/items, date range, purchases per user and per item).
- [x] **Verification:** Parquet files exist; memory in use after loading transactions is < 2 GB; statistics match the dataset (~31.8M transactions, ~1.37M customers, ~106K articles, 2018-09-20 → 2020-09-22).

### Phase 2: Split, filtering and baselines (before any model)
- [x] Time-based split: train ≤ 2020-09-08, val 2020-09-09 → 09-15, test 2020-09-16 → 09-22. Evaluate only users who purchased in the target week.
- [x] Sparsity filter on the **training** window only (users > 5 purchases, articles > 20). Log how many users, items and transactions each filter removes.
- [x] `src/metrics.py`: MAP@12 (Kaggle definition), Recall@k (k = 12, 100), with unit tests on hand-built examples.
- [x] `src/baselines.py`:
  - recent popularity (top 12 items in the last training week)
  - repeat purchase (the user's most recent items, filled in with popularity)
- [x] **Verification:** MAP@12 / Recall@k for both baselines on val and test, saved to `benchmarks/baselines.json`. Sanity check: last-week popularity should score roughly 0.02 MAP@12. If it's far off, the split or the metric is wrong. *(Corrected in Phase 2: ~0.02 is what repeat-purchase scores; plain popularity is ~0.007-0.009, consistent with public Kaggle notebooks.)*

### Phase 3: Features and Dataset
- [ ] Label-encode categoricals (index 0 reserved for unknown/padding):
  - User: `customer_id`, `age_bucket`, `club_member_status`, `fashion_news_frequency`
  - Item: `article_id`, `product_group_name`, `colour_group_name`, `product_type_name`, `index_group_name`
- [ ] **User history features:** the last N (e.g. 20) purchased article IDs before each transaction's timestamp (padded and masked), purchase count, days since last purchase. **No leakage:** history must only include purchases strictly before the target.
- [ ] **Item text features:** embed `detail_desc` with a small sentence-transformer (e.g. `all-MiniLM-L6-v2`), cache it to `data/item_text_emb.npy`, and feed it into the item tower through a linear projection.
- [ ] Item sampling frequencies for logQ (from training transactions).
- [ ] `src/dataset.py`: a PyTorch `Dataset` yielding `(user_features, item_features, item_id)` for each positive transaction; `num_workers > 0`, `pin_memory=True`.
- [ ] **Verification:** batch shapes are correct; one full pass over the DataLoader runs with flat RSS memory (no leak); a leakage test asserts every history item's timestamp is < the target timestamp.

### Phase 4: Model (`src/model.py`)
- [ ] `ItemTower`: ID + categorical embeddings + projected text embedding → MLP → 64-d → L2-normalize.
- [ ] `UserTower`: categorical embeddings + **history encoder** (mean-pool the history item embeddings, sharing the item tower's ID embedding table) + numeric features → MLP → 64-d → L2-normalize. Make the `customer_id` embedding switchable on/off for the ablation.
- [ ] `TwoTowerModel`: returns user and item vectors; scores are `U @ I.T / τ`.
- [ ] **Verification:** a dummy forward pass gives a `[B, B]` score matrix; no shape errors; parameter count printed.

### Phase 5: Training (`src/train.py`)
- [ ] In-batch softmax / InfoNCE loss: labels = diagonal.
  - **logQ correction:** `logits -= log(q_item)` (applied to the columns)
  - **accidental-hit masking:** set duplicate item IDs off the diagonal to −inf
  - temperature τ (config option or learnable)
- [ ] AdamW, warmup + cosine LR schedule, AMP (fp16 on T4), gradient clipping.
- [ ] MLflow logging: loss, LR, epoch time, throughput (samples/s), val Recall@100 and MAP@12 each epoch. Log the config and git SHA as params.
- [ ] Early stopping and checkpoint selection on **val MAP@12** (not val loss).
- [ ] Ablations, each a separate MLflow run:
  - with vs. without logQ
  - with vs. without history features
  - with vs. without text features (also report the cold-start-items slice)
  - `customer_id` embedding on vs. off
- [ ] **Verification:** at least 1 epoch completes; loss decreases; val MAP@12 is logged; the best model **beats the recent-popularity baseline on Recall@100**. (Candidate generation is judged on recall. If MAP@12 doesn't beat popularity, report it honestly and explain why.)

### Phase 6: Distributed training (real DDP)
- [ ] `torchrun`-compatible entry point: `init_process_group` (NCCL), `DistributedSampler` with `set_epoch`, DDP wrapper, rank-0-only logging and checkpointing.
- [ ] **Cross-GPU negatives:** `all_gather` the item embeddings (with gradients via `torch.distributed.nn.functional.all_gather`) so each GPU computes the loss against the full global batch.
- [ ] Run on **Kaggle 2× T4** (free) and/or **Azure** (Section E).
- [ ] **Verification:** `benchmarks/ddp_scaling.json` with samples/s and epoch time for 1 vs. 2 (vs. 4) GPUs, scaling efficiency, and confirmation that val metrics match single-GPU within noise. Also compare local vs. global negatives on Recall@100.

### Phase 7: Retrieval and serving
- [ ] `src/export_embeddings.py`: frozen item tower → embeddings for all articles → `models/item_emb.npy` + ID map.
- [ ] `src/build_index.py`: FAISS `IndexFlatIP` (exact, the ground truth), `IndexIVFFlat` (sweep `nlist`/`nprobe`), `IndexHNSWFlat` (sweep `efSearch`).
- [ ] `benchmarks/ann_tradeoff`: **recall@100 against exact search vs. per-query latency** for each index; chart it.
- [ ] `src/serve.py`: FastAPI `POST /recommend {customer_id, k}`. Look up user features (precomputed in memory), run the user tower on CPU, query FAISS, filter items the user already bought (configurable), return the top-k article IDs. Add `/health`.
- [ ] Load test: async client at concurrency 1/8/32/64 (reuse the pattern from LLM_Inference_Project `load_test.py`).
- [ ] Dockerfile for the serving app.
- [ ] **Verification:** returns k valid article IDs; **p99 < 50 ms at concurrency 32** with the chosen index; recall vs. exact search ≥ 95%.

### Phase 8: Results and write-up
- [ ] `src/plot_results.py`, producing:
  - a model vs. baselines table
  - an ablation table
  - the DDP scaling chart
  - the ANN recall-latency chart
  - the serving latency chart
- [ ] README: architecture diagram (mermaid), results table, charts, "what the numbers say", design decisions, how to run (Kaggle/Colab/Azure).
- [ ] `docs/concepts.md`: two-tower vs. ranking models, in-batch negatives and logQ, temperature, why cross-GPU negatives help, IVF vs. HNSW, cold-start, likely interview questions.
- [ ] **Resume bullet** filled in from measured numbers only, e.g.:
      *"Two-tower retrieval model on 31.8M H&M transactions: MAP@12 X vs. Y for the popularity baseline; logQ correction +Z% Recall@100; text features +W% recall on cold-start items; N× training speed-up with DDP; HNSW at R% recall with P ms p99 serving."*

---

## E. Azure plan ($200 credits)

**Rule:** develop and debug on free compute (Kaggle/Colab). Spend Azure credits only on things that look good on the
resume (the owner's experience is Azure-heavy, so Azure ML fits) or that free tiers can't do.

### Before spending anything
- [x] **Set a budget alert:** Cost Management → Budgets → $150 with alerts at 50/80/100%.
- [x] **Check GPU quota:** new and credit-based subscriptions often start at **0 vCPUs for GPU families**. Request quota for `Standard NCASv3_T4 Family` (and spot quota if using spot) in the chosen region. Approval can take hours or days, so request it early. Some offer types (free trial, Azure for Students) don't allow GPU VMs at all. If that's the case, use Kaggle for DDP and Azure for tracking and serving only.
- [x] Use **one resource group** for the whole project so cleanup is a single `az group delete`.
- [ ] Compute clusters: **min nodes = 0** and idle scale-down of about 120 s. Compute instances: **auto-shutdown on**.

### Where credits go (approximate pay-as-you-go prices in East US; **check them in the Azure pricing calculator**, and use spot/low-priority VMs where possible at roughly 60–90% off)

| Use | Resource | Est. hours | Est. cost |
|---|---|---|---|
| Experiment tracking (MLflow) | **Azure ML workspace**: its built-in MLflow tracking URI replaces a self-hosted MLflow server. Includes a storage account, Key Vault, and an optional container registry | whole project | ~$5–15 (storage + ACR Basic ~$5/mo) |
| Single-GPU training runs | Azure ML compute cluster, `Standard_NC4as_T4_v3` (1× T4, ~$0.53/h) | ~15–20 h | ~$8–11 |
| **Multi-GPU DDP** | `Standard_NC64as_T4_v3` (4× T4, ~$4.35/h), or 2 nodes of NC4as for multi-node DDP | ~3–4 h | ~$13–18 (much less on spot) |
| Serving demo | **Azure Container Apps** (consumption, CPU, scale-to-zero) running the FastAPI + FAISS container | demo period | ~$0–10 |
| Buffer for mistakes and reruns | | | ~$50+ |
| **Total planned** | | | **~$40–70 of $200** |

**What to show from Azure:** Azure ML job submission (`az ml job create` with a YAML job spec), the MLflow runs and
ablation comparison in Azure ML Studio, a DDP job on a multi-GPU SKU, and a container deployed to Container Apps.
These match the Azure ML, IaC and CI/CD keywords already on the resume.

**Cleanup at the end:** delete the compute cluster and Container App (or the whole resource group once results and
screenshots are saved).

---

## Implementation notes & deviations
_(Fill in during execution: anything that differed from the plan, and why.)_

### Phase 1 (2026-10-24)
- **Customer IDs:** dense `customer_idx` int32 (row order of `customers.csv`) instead of an int64 hash; the string mapping lives in `customers.parquet`. Dense codes plug straight into `nn.Embedding`.
- **Sampling:** `--sample-frac` samples customers, not rows, so each sampled customer's full history is kept (Phase 3 history features need complete histories).
- **Memory:** transactions DataFrame 0.67 GB, process RSS growth 1.2 GB (pyarrow `self_destruct`). One-off conversion peaks at 2.9 GB RSS, 22 s.
- **Duplicates:** 2,974,905 duplicate rows (same customer/article/day, i.e. quantity > 1) kept as-is; Phase 2 decides how to count them.
- **Extra deps:** `psutil` (memory measurement), `pytest`, Azure ML SDK (`azure-ai-ml`, `azure-identity`, `azureml-mlflow`). `mlflow` pinned to 3.15.0 to match the `mlflow-skinny` that `azureml-mlflow` resolves to.
- **Azure is a Free Trial:** 4 vCPU limit and 0 T4 quota in every region checked. **GPU training moves to Colab (single GPU) and Kaggle 2x T4 (DDP)**; Azure is used for MLflow tracking, data storage and the serving demo. Section E's GPU rows no longer apply.
- **Azure resources:** resource group `rg-twotower-recsys` (eastus); budget `twotower-150` ($150/month, alerts at 50/80/100% actual + 100% forecast); workspace `mlw-twotower` (created by `infra/azure_setup.py`); data asset `hm-processed:1` (all of `data/processed/`, including the 5% sample). Test MLflow run in experiment `phase1-data`.
- **No Databricks/Spark:** the largest table is 0.67 GB in memory; a cluster adds cost without speed-up.


### Phase 2 (2026-10-24)
- **Full-data runs moved off the laptop** (8 GB RAM; the Phase 1 conversion caused heavy swapping). Local work uses the 5% sample; full data runs as Azure ML command jobs (`infra/submit_job.py`) on cluster `cpu-e4` (Standard_E4s_v3, 4 vCPU / 32 GB, 0-1 nodes, 120 s idle scale-down). Environment is the public `docker.io/library/python:3.12-slim` image plus `requirements-job.txt`, so no container registry is needed yet.
- **Test-week history includes the val week** (history = everything before the target week); model weights still train on <= 2020-09-08 only.
- **Ground truth** counts each article once per user per week (Kaggle definition); duplicate purchases are not double-counted.
- **Full-data results** (`benchmarks/baselines.json`, Azure ML job `red_cushion_gnzz92zsyn`):
  - popularity: val MAP@12 0.0066, Recall@100 0.099; test MAP@12 0.0087, Recall@100 0.118
  - repeat purchase: val MAP@12 0.0241, Recall@100 0.111; test MAP@12 0.0246, Recall@100 0.122
- **Sparsity filter on the full training window** removes 36.8% of users, 31.0% of items, but only 4.9% of transactions. 15-16% of evaluated users would be dropped by the filter and 7.5-8.6% have no training history at all (cold start).
- **The 5% sample overstates item sparsity** (74% of items removed vs 31% on full data) because per-item counts shrink 20x; thresholds must be judged on full data.
- **Full-data candidate pool (train window):** 102,967 articles bought; 71,024 with > 20 purchases; only 29,278 sold in the last 4 weeks. 913 val-week and 1,424 test-week articles never appear in training (cold-start items).
