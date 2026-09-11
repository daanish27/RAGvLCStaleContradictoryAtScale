# RAGvLCStaleContradictoryAtScale
Benchmarking RAG vs. long-context LLM generation for robustness to stale and contradictory evidence, using a temporal CEO-succession corpus at scale.

A small experiment testing whether retrieval-augmented generation (RAG) handles **temporal/contradictory knowledge better than dumping everything into a longcontext window**, using a synthetic corpus of real CEO succession facts pulled
from Wikipedia.

## What this tests

Given a company, "who is the CEO as of a fixed reference date?" where the context may contain:
- the current, correct answer
- **stale paraphrases** of former CEOs (same true fact, reworded, wrong era)
- **contradictions** (an outright false CEO/date pairing, framed as a
  retracted/disputed report)

Three conditions are compared on the *same* query, using the *same* generator model (`mistral-small-latest`) throughout, so only the context-construction strategy varies:

| Condition | Description |
|---|---|
| `long_context` | Full corpus for the company is stuffed into the prompt |
| `hybrid_rag` | Top-k docs via Reciprocal Rank Fusion (TF-IDF + dense embeddings) |
| `recency_weighted_rag` | Same hybrid RRF, then re-weighted by exponential recency decay (180-day half-life default overrided in code to median 2,574 days) |

## Pipeline

1. **Fetch real data**: Pulls a short Wikipedia extract for each CEO in `CEO_SUCCESSION_SEED` and builds one "primary" document per CEO-era.
2. **Assemble a synthetic corpus**: For each company, the most recent era (relative to `AS_OF_DATE`) is marked ground truth; every other era is "stale" and gets expanded into paraphrases via `PARAPHRASE_TEMPLATES`. A fixed set of `CONTRADICTION_SEED` entries inject false CEO/date claims.
3. **Retrieve + generate**: For each of the three conditions, build a prompt from the relevant context docs and get an answer from the generator.
4. **Evaluate**: The answer is classified as `correct`, `stale`, `contradiction_echoed`, or `miss` (string matching first, LLM-as-judge as fallback), and cross-referenced with whether the current/correct document was actually retrieved, giving a failure taxonomy:
   - `success`: correct doc retrieved and used correctly
   - `retrieval_failure`: correct doc never made it into context
   - `generation_failure`: correct doc was in context, model still got it wrong
   - `correct_without_evidence`: right answer despite the correct doc not
     being retrieved (likely parametric/pretraining leakage, not grounded)
5. **Sweep**: Repeated across corpus sizes (`CORPUS_SIZE_CONFIGS`), stale document ratios (`STALE_RATIO_CONFIGS`), and multiple trials per config, to see how each condition's accuracy and stability hold up as the context gets larger and noisier.

## Requirements

- Python 3
- `openai`, `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `requests`
- A Mistral API key, stored in Colab secrets under `mistral_kay` (used via
  `google.colab.userdata`): Swap this out for `os.environ` if running
  outside Colab

## Running

Designed to run top-to-bottom in a single notebook/script. It will:
- fetch Wikipedia data (rate-limited, cached, with retry/backoff)
- run the full sweep (`run_full_sweep`), which makes many Mistral API calls (chat completions + embeddings), also rate-limited with retry/backoff
- write `rag_vs_longcontext_results.csv` and `rag_vs_longcontext_corpus_meta.csv`
- produce three plots: `accuracy_clean.png`, `stability_three_bars.png`, `rag_vs_longcontext_failure_taxonomy.png`

Note the sweep is expensive: `CORPUS_SIZE_CONFIGS x STALE_RATIO_CONFIGS x N_TRIALS` corpora, each running 3 conditions per company. Reduce `N_TRIALS` or the config lists for a quick smoke test.

## Key parameters

- `AS_OF_DATE`: Fixed "today" the whole experiment reasons about, for
  reproducibility
- `GEN_MODEL` / `EMBED_MODEL`: Single generator/embedder used across all
  conditions so results aren't confounded by model choice
- `half_life_days`: Recency decay half-life for `retrieve_recency_weighted`;
  defaults to 180 days. Note `run_single_corpus_trial` calls it with `half_life_days=2574` for the `recency_weighted_rag` condition in the sweep  actually in effect for your results
- `n_paraphrases_per_stale_doc` — how many reworded versions each stale fact gets, controlling noise/stale ratio in the corpus

## Caveats

- Ground truth and "stale" labeling both key off Wikipedia extract text and hand-maintained seed dates: errors in `CEO_SUCCESSION_SEED` or `CONTRADICTION_SEED` propagate directly into the eval.
- LLM-as-judge fallback in `classify_answer_with_judge` is itself a Mistral call, so judge errors are possible on ambiguous answers.
- Results are specific to `mistral-small-latest` / `mistral-embed` as of the run date; re-running later may shift absolute numbers even with the same code.
