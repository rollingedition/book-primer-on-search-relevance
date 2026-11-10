---
description: "Use when evaluating or improving search ranking: grading relevance judgments, scoring rankings with nDCG, precision and MRR, deciding with a permutation test and bootstrap interval whether one ranker beats another, checking LLM relevance judgments against people, tuning BM25, semantic retrieval with embeddings and HNSW, hybrid retrieval with min-max blends or reciprocal rank fusion, query spelling, synonyms, classification and LLM rewrites, learning to rank with LambdaMART, cross-encoder and LLM reranking, evaluating retrieval for RAG answers and agents, and testing a search change before shipping with URisk, A/B tests, power and interleaving. Teaches product search on a public furniture and home-goods product-search dataset, in Python."
---

## Grade relevance and prepare judgments

**When:** you are about to evaluate a search system and need to know what its judgments can support.

1. Before blaming a ranker for missing an exact product, check whether any field it reads says what the judges saw ([Judge relevance against the shopper's need](#relevance-is-a-judgment)).
2. Keep one grade per pair, prefer the lower one, and write down the rule ([Expect judges to disagree on some pairs](#judges-disagree-on-some-pairs)).
3. Collect graded judgments ([Grade relevance instead of yes or no](#graded-relevance)). Fix the gains before comparing rankers and report them with every score ([Choose the gains on purpose](#gains-are-a-choice)).
4. Print each query's label mix before averaging any metric ([Read the label mix of each query](#label-mix-per-query)).
5. Use judgments, not clicks, as the reference ([Treat clicks as biased evidence](#clicks-are-not-judgments)), and find out which systems built the judgment pool ([Know where the judgments came from](#pooled-judgments)).
6. Report judged@10 next to every nDCG@10, and compute condensed scores only as a diagnostic, never as the headline; when two rankers' judged@10 differ, have their unjudged top-10 products judged ([Count unjudged products before you compare scores](#unjudged-results)).

**Check:** run [the example](examples/unjudged-in-bm25-top-10.py) and compare its `mean judged@10` line, {{out:unjudged-in-bm25-top-10:mean judged@10}}, and its check that the hand-built value matches the library: {{out:unjudged-in-bm25-top-10:queries where the hand-built judged@10 equals lib.metrics}}.

**Pitfalls:** judgments go stale, so date them and track judged@10 in production; guidelines shape the whole set; judges may see more than the ranker ([What goes wrong in production](#what-relevance-means-in-production)).

## Score a ranking with nDCG and other metrics

**When:** you need a score per query, and a mean, for a ranking against graded judgments.

1. Use precision@10 for how much of the first page is right; don't make recall@10 a headline when queries have many relevant products ([Count the relevant products in the top k](#count-relevant-in-top-k)).
2. Use MRR as the headline only when the user needs one right product ([Score the first relevant result with reciprocal rank](#score-the-first-relevant-result)).
3. Compute nDCG@10 with linear gains, a log2(i + 1) discount and an ideal ranking from every judged product, and state the variant with every number ([Build nDCG from discounted gains](#build-ndcg-from-dcg)).
4. Check it against scikit-learn's `ndcg_score` with every judged product in each row, after checking each run holds at least k products and, when passing raw scores, breaking ties by product ID ([Match scikit-learn's `ndcg_score`](#match-scikit-learn-ndcg)).
5. Write down the headline metric, gains and cutoff before comparing, and never change them after seeing which ranker they favour ([Choose a metric before you compare](#choose-a-ranking-metric)).

**Check:** run [the example](examples/ndcg-matches-scikit-learn.py) and compare its `largest difference` line: {{out:ndcg-matches-scikit-learn:largest difference}}.

**Pitfalls:** tied scores make runs non-repeatable; unjudged products lower a ranker's score ([What goes wrong in production](#measuring-search-in-production)).

## Decide whether one ranker beats another

**When:** rankers have per-query scores on the same judged queries, and someone wants to act on the gap.

1. Print the mean and median difference, wins, losses and ties, and the largest losses, then read those queries ([Read per-query differences, not only the mean](#read-per-query-differences)).
2. List the planned comparisons before reporting any. A difference is established only when its permutation p-value is below the significance level divided by the number of planned comparisons; label later comparisons exploratory ([Decide in advance what counts as established](#when-a-difference-is-established)).
3. Run a paired permutation test on the per-query differences, and read a p-value above the threshold as "no evidence of a difference" ([Test a gap with a permutation test](#permutation-test-on-queries)).
4. Report the paired t-test beside it, and say when the two disagree ([Get the p-value from a paired t-test](#paired-t-test-on-queries)).
5. Report the bootstrap interval and effect size, and decide before testing what gap justifies the change's cost ([Report how big the gap is](#report-the-size-of-a-gap)).
6. Report each run's nDCG@10 and judged@10, then one column per comparison ([Compare runs in an evaluation harness](#compare-runs-in-a-harness)).

**Check:** run [the example](examples/bootstrap-interval-by-hand.py) and compare its `semantic minus bm25 bca interval and 0` line: {{out:bootstrap-interval-by-hand:semantic minus bm25 bca interval and 0}}.

**Pitfalls:** one split can mislead, and reusing test queries is testing many hypotheses, so log every comparison on them ([What goes wrong in production](#measuring-search-in-production)).

## Get judgments from an LLM and check them against people

**When:** you want an LLM to grade query-product pairs, or to trust scores from its grades.

1. Give the LLM the guidelines people use, plus a rule for what it can't see ([Write judging guidelines](#write-judging-guidelines)).
2. On pairs drawn the way you will use the judge, report kappa with observed agreement ([Measure agreement beyond chance with kappa](#agreement-beyond-chance)), linear weighted kappa with far misses counted apart ([Give near misses partial credit with weighted kappa](#weighted-kappa-for-grades)), and the label mix ([Read kappa against the label mix](#kappa-depends-on-label-mix)).
3. Read the confusion matrix before kappa and name the errors' direction for that pool ([Read the direction of the LLM's errors](#read-llm-error-direction)); check whether the evidence was in what the LLM saw ([Compare what the LLM saw with what the judges saw](#what-the-llm-saw)).
4. Record every prompt word for word with the model version, run two prompts, and trust a conclusion only when both give it ([Measure how much the prompt changes the labels](#prompt-sensitivity)).
5. Fill unjudged pairs with LLM grades only as a diagnostic, beside the human-only score ([Fill unjudged pairs with LLM grades](#fill-unjudged-with-llm-grades)).
6. When checking whether LLM grades order rankers as people do, report Kendall's tau with the list of swapped pairs, never alone, then test whether the swapped pairs differ at all ([Check whether LLM grades order the rankers the same way](#system-order-human-versus-llm)).
7. Never ship on LLM grades alone ([Rerun the planned comparisons with LLM grades](#llm-graded-verdicts)); correct a score from LLM grades with a random human-judged sample of queries ([Correct LLM scores with a human sample](#calibrate-on-a-human-sample)).

**Check:** run [the example](examples/kappa-by-hand.py) and compare its `kappa by hand, short prompt` line, {{out:kappa-by-hand:kappa by hand, short prompt}}, with `kappa from scikit-learn, short prompt`: {{out:kappa-by-hand:kappa from scikit-learn, short prompt}}.

**Pitfalls:** a different model version or prompt is a different judge; systems tuned to the judge score well on it; product text can steer it; choose prompts on train queries ([What goes wrong in production](#judgments-from-people-and-llms-in-production)).

## Build and tune lexical retrieval with BM25

**When:** you are building, debugging or tuning keyword search.

1. Print the terms queries and products become, through one tokenizer ([Split product text into terms](#split-text-into-terms)), and look up a failing query's terms in the index before changing scoring ([Build an inverted index](#build-an-inverted-index)).
2. Serve BM25 from a library and check it against a small version of your own, comparing tokenizers before formulas ([Rebuild BM25 by hand to match bm25s](#bm25-from-scratch)).
3. Stem products and queries alike, and read the products a stem adds ([Stem terms so word forms match](#stem-word-forms)).
4. Choose each change on train queries and keep it only on an established test gain ([Check stemming on the test queries](#check-stemming-on-test)); tune `k1` and `b` on train when what you index changes ([Tune BM25's parameters on train queries](#tune-bm25-parameters)).
5. Weight a field before saturation, with BM25F or by repeating it, not by adding field scores ([Weight the product name above the description](#weight-the-product-name)). Index attribute values without their names ([Index product attributes as a field](#index-attributes-as-a-field)), and prefer a measured phrase bonus to a filter ([Match phrases with term positions](#match-phrases-with-positions)).

**Check:** run [the example](examples/bm25-by-hand.py) and compare its `same top 100 for every query` line: {{out:bm25-by-hand:same top 100 for every query}}.

**Pitfalls:** index and queries drift apart; sizes and codes don't survive tokenizing; IDF depends on the catalog; a BM25 score has no fixed scale, so cut by rank ([What goes wrong in production](#lexical-retrieval-in-production)).

## Retrieve by meaning with embeddings and HNSW

**When:** you are adding or checking embedding-based retrieval.

1. Embed with the same model, revision and prompts on both sides, recorded with the vectors, and check the longest product fits the input limit ([Turn text into embeddings](#turn-text-into-embeddings)).
2. Normalize vectors when stored and after any change, then rank by dot product ([Compare embeddings with cosine similarity](#compare-with-cosine-similarity)).
3. Use exact search while it fits memory and latency; it is the ground truth for any index ([Find the exact nearest neighbours with NumPy](#exact-nearest-neighbours-numpy)).
4. Tune an HNSW index's `efSearch` on train queries by recall against exact search, choosing by the worst query ([Trade recall for speed with HNSW](#hnsw-recall-for-speed)), then check it on held-out queries ([Check the index on the test queries](#hnsw-on-test-queries)).
5. List the queries that hold sizes, quantities or codes and compare both rankers on them ([Check queries that contain digits](#embeddings-match-digits)); match codes lexically ([Expect a code to carry little weight](#codes-carry-no-weight)); don't rely on retrieval for negation ([Test embeddings on negation](#embeddings-on-negation)).
6. Decide between models or vector sizes on your train queries with a paired test ([Truncate dimensions, then test the loss](#truncate-dimensions-test-loss)).

**Check:** run [the example](examples/exact-search-numpy.py) and compare its `same top 10 as faiss IndexFlatIP` line: {{out:exact-search-numpy:same top 10 as faiss IndexFlatIP}}.

**Pitfalls:** a similarity is not a confidence; mismatched vectors fail silently; an index doesn't follow the catalog; long texts are cut short; typos move vectors ([What goes wrong in production](#semantic-retrieval-in-production)).

## Combine lexical and semantic retrieval

**When:** you are considering a hybrid of two retrievers.

1. On train queries, count where each retriever wins and what their union adds ([Find the queries each retriever wins](#queries-each-retriever-wins)).
2. Check each retriever alone; give a weaker one a tuned weight, not an equal vote ([Check each retriever before you fuse](#check-each-retriever-before-fusing)).
3. Never add raw scores; normalize per query with min-max and tune α on train ([Normalize scores before you add them](#normalize-scores-before-adding)).
4. For reciprocal rank fusion, keep k = 60 and fuse lists at least 100 deep ([Test reciprocal rank fusion's settings on train queries](#rrf-settings-on-train)).
5. Tune one weight on a coarse grid ([Tune the blend's weight on train queries](#tune-blend-on-train)), then bootstrap and reverse ties to check it holds ([Check that the tuned weight is stable](#tuned-weight-stability)).
6. Adopt a blend only when it beats the best single retriever on held-out queries ([Report the tuned blends on test queries](#report-blends-on-test)); start with weighted RRF when you have ranks only ([Decide when hybrid retrieval pays](#when-hybrid-helps)).

**Check:** run [the example](examples/tuned-blends-on-test.py) and compare its `weighted sum established` line: {{out:tuned-blends-on-test:weighted sum established}}.

**Pitfalls:** two retrievers cost two of everything; short lists drop candidates; min-max is relative; a tuned weight belongs to one pair of retrievers ([What goes wrong in production](#hybrid-retrieval-in-production)).

## Understand and rewrite queries

**When:** you are adding spelling correction, unit rules, synonyms, query classes or LLM rewrites.

1. Correct only words the catalog lexicon doesn't hold, checking each word's neighbours ([Correct spelling against the catalog's words](#correct-spelling-from-catalog)).
2. Write unit rules from the catalog's text and apply them to products and queries together ([Normalize units the way the catalog writes them](#normalize-units)).
3. Write each synonym from a failing query and keep it only on a held-out gain ([Expand queries with a synonym list](#expand-with-synonyms)).
4. Check a query classifier against judged exact products' classes ([Train a query classifier on product text](#classify-queries-from-products)), and boost the predicted class instead of filtering ([Boost the predicted class instead of filtering](#boost-predicted-class)).
5. Test an LLM rewrite or HyDE on held-out queries before using it, reading losses for dropped requirements; with HyDE, always average the query's own embedding in ([Rewrite queries with an LLM](#rewrite-queries-with-llm), [Search with a hypothetical product (HyDE)](#hyde-hypothetical-products)).
6. Never search a follow-up alone ([Turn a follow-up into a standalone query](#standalone-follow-ups)).

**Check:** run [the example](examples/catalog-spelling-corrector.py) and compare its `same as lib.queries.correct_query` line: {{out:catalog-spelling-corrector:same as lib.queries.correct_query}}.

**Pitfalls:** each rewrite is a model call on the query's path; the lexicon and classifier must follow the catalog; rewrites surface unjudged products ([What goes wrong in production](#understanding-queries-in-production)).

## Learn to rank and rerank

**When:** you are training a learned ranker or adding a cross-encoder or LLM reranker.

1. Compare each feature by grade on train queries, and compute features with one code path ([Describe each candidate with features](#ranking-features)).
2. Train with a gradient weighted by your metric, as LambdaMART does ([Learn a ranking from pairs of products](#learn-from-pairs)). In LightGBM, pass `group`, set `label_gain`, stop early on held-out train queries, fix seed and threads, and order candidates by ID ([Train LambdaMART with LightGBM](#train-lambdamart)).
3. Adopt it only on an established test gain, after reading its gain importance ([Report LambdaMART on the test queries](#lambdamart-on-test-queries)).
4. Rerank your strongest retrieval at a depth fixed in advance ([Rerank semantic retrieval with a cross-encoder](#rerank-semantic-with-cross-encoder)).
5. For LLM rerankers, fix the tie-break ([Score each product with an LLM](#llm-pointwise-reranking)), check each listwise answer for missing or repeated numbers ([Order the 20 products with one LLM prompt](#llm-listwise-reranking)), and run each query in two orders ([Reverse the input to measure positional bias](#llm-position-bias)).
6. Set the depth on train queries from calls and characters per query ([Set the rerank depth from a budget](#rerank-depth-from-budget)); distil only a reranker with an established gain ([Distill the cross-encoder into a small ranker](#distill-the-cross-encoder)).

**Check:** run [the example](examples/listwise-position-bias.py) and compare its `tau between the two answers, mean` line: {{out:listwise-position-bias:tau between the two answers, mean}}.

**Pitfalls:** features drift; row order leaks into training; LLM judges favour LLM rerankers; give each stage a time limit and a fallback ([What goes wrong in production](#ranking-in-production)).

## Evaluate retrieval for RAG answers and agents

**When:** retrieval feeds an LLM's answer or an agent's search tool.

1. Put a few retrieved items in the prompt, choosing how many on train queries ([Put retrieved products in the prompt](#put-retrieved-products-in-the-prompt)).
2. Score answers by their cited items' human grades, count answers that recommend nothing, and keep a random context as a floor ([Score each answer by the products it cites](#retrieval-bounds-the-answer)).
3. Check every cited ID is in the context before a user sees it ([Check every citation against the context](#check-citations-against-context)); calibrate any support check on a human sample ([Check what the cited product supports](#check-citation-support)).
4. Report the context's precision@k from your judgments, with recall@k beside it ([Measure the context, then the answer](#measure-the-context)).
5. Index short items whole; chunk long ones with the title in front ([Choose the retrievable unit](#choose-the-retrievable-unit)).
6. Read an agent's traces and add its searches to your test queries ([Read an agent's searches from its trace](#read-agent-searches)); test its tool's retriever on its searches and picks ([Compare the retrievers behind the agent's tool](#retriever-behind-agent-tool)).

**Check:** run [the example](examples/citations-inside-the-context.py) and compare its `answers that cite an ID outside their context` line: {{out:citations-inside-the-context:answers that cite an ID outside their context}}.

**Pitfalls:** model, prompt and context are one system; abstentions move with retrieval; product text reaches the prompt; agent traffic is a different workload ([What goes wrong in production](#retrieval-for-llm-applications-in-production)).

## Test a change before shipping it

**When:** a candidate ranker is ready and you must decide whether to ship it.

1. Diff the candidate against the baseline per query on a regression set nothing is tuned on, and read every loss ([Diff two runs query by query](#diff-runs-per-query)).
2. Report URisk with α = 1 beside the mean change; don't ship when it is negative ([Weigh losses more than wins with URisk](#weigh-losses-with-urisk)).
3. Rescore both runs with each side's unjudged products as exact; if the leader changes, judge them ([Judge the products a new ranker surfaces](#judge-new-results)).
4. Choose every setting on train queries and log each look at the test queries ([Keep the test queries out of every choice](#keep-test-queries-held-out)).
5. Before an A/B test, fix the metric and guardrails, run an A/A test, check the sample ratio, and read once at the planned size ([Run an A/B test](#run-an-ab-test)), sized from power ([Size the test with power](#size-tests-with-power)).
6. To learn which ranking users prefer, use team-draft interleaving ([Interleave two rankings with team draft](#team-draft-interleaving)); on real traffic, fix the number of searches in advance and decide with a sign test, and never treat simulated verdicts as evidence ([Rehearse interleaving with simulated shoppers](#simulate-interleaving-with-clicks)).
7. When offline and online disagree, find the cause ([Find the cause when offline and online disagree](#offline-online-disagree)). Ship only when every checklist item passes ([Decide whether to ship](#decide-whether-to-ship)).

**Check:** run [the example](examples/aa-tests-with-peeking.py) and compare its `significant on some day, checked daily` line, {{out:aa-tests-with-peeking:significant on some day, checked daily}}, with `significant at the planned end`: {{out:aa-tests-with-peeking:significant at the planned end}}.

**Pitfalls:** the dashboard invites peeking; groups don't match the plan; the regression set wears out; ties move results ([What goes wrong in production](#testing-changes-in-production)).
