---
chapter_id: semantic-retrieval
---

# Semantic retrieval

By the end of this chapter you can build semantic retrieval for a catalog from stored embeddings: compare vectors by cosine similarity, find a query's nearest products exactly with NumPy, switch to an approximate index when a catalog outgrows exact search and measure what the index misses, and find the queries where embeddings go wrong. You can also choose a vector size by testing it on your own judgments, and compare models the same way.

[Semantic retrieval](#search-by-meaning) is the ranker to beat in the store's search, but so far the book has used it as one class, `lib.semantic.SemanticSearch`, without opening it. [Chapter 1](#search-end-to-end) built it for [the store](#store-data), whose catalog runs from sofas to shower curtains, over vectors that one model computed once, ahead of time. On the [test queries](#store-data) it scored a mean [nDCG@10](#score-one-ranking) of {{out:compare-lexical-semantic:semantic ndcg@10}}. Its lead over [BM25](#rank-with-bm25), which scored {{out:compare-lexical-semantic:bm25 ndcg@10}}, is one that [chapter 3 establishes](#compare-runs-in-a-harness). Yet it returns hoods of other widths for `48 inch kitchen hood`, and round rugs for `rug that is not round`. Before the store depends on it, you need to know what those vectors hold, what searching them would cost for a far larger catalog, and which queries they get wrong. This chapter builds exact search by hand, measures an approximate index against it, and tests both on the store's queries and judgments.

On the test queries, the chapter plans {{out:hnsw-test-recall:planned test comparisons}} comparisons, both in [the section on testing the index](#hnsw-on-test-queries). Each sets exact search, which compares a query with every product, against HNSW, an approximate index that compares it with only a fraction of them: once at faiss's default setting, and once at the setting chosen on the train queries. Under [the book's rule](#when-a-difference-is-established), a difference is established when its paired permutation p-value is below {{out:hnsw-test-recall:p-value threshold}}. Every other look at a test query in this chapter is an exploratory diagnosis.

## Turn text into embeddings {#turn-text-into-embeddings}

An **embedding** is a list of numbers, a vector, that a trained model computes for a piece of text. The model is an **encoder**, trained so that texts with similar meanings get vectors that point in similar directions, whether or not they share words. Every text gets the same count of numbers, the embedding's **dimension**, whether it's a three-word query or a product's whole description. One number means nothing on its own; only comparisons between whole vectors do.

Search uses the encoder as a **bi-encoder**: it encodes each product by itself, once, ahead of time, and each query by itself when the shopper types it. Comparing a query with a product is then arithmetic on two vectors rather than a run of the model, which is what makes it cheap enough to compare one query with a whole catalog. [Chapter 1](#search-by-meaning) gives the size of the difference from the Sentence-BERT paper: about 65 hours for BERT to read every pair of 10,000 sentences, against about 5 seconds to embed each sentence once and compare. A model that reads a query and a product together is a cross-encoder, and [chapter 1](#rerank-with-cross-encoder) uses one to rerank a shortlist.

The store's encoder is `Snowflake/snowflake-arctic-embed-m-v1.5`, an open model with 109 million parameters under the Apache-2.0 licence ([model card](https://huggingface.co/Snowflake/snowflake-arctic-embed-m-v1.5)). [Merrick et al. (2024)](https://arxiv.org/abs/2405.05374) report that its family's training kept only documents identified as mainly English. We chose it because anyone may use it commercially, it's small, and it was trained so that its vectors can be shortened, which [the section on dimensions](#truncate-dimensions-test-loss) tests. Examples don't run the model. They read a **recording**: vectors that the script `recordings/embeddings.py` computed once, at a pinned revision, and saved at unit length, so every example sees the same numbers. The model card asks for a prefix on queries only, so the script adds it to each query and embeds products as they are. Examples read the vectors through `lib.recorded.embeddings()`:

[Example: inspect-recorded-embeddings](../examples/inspect-recorded-embeddings.py)

Each product is a vector of {{out:inspect-recorded-embeddings:dimensions}} numbers; product 1's begins {{out:inspect-recorded-embeddings:product 1, first four numbers}}, and none of those numbers means anything alone. The recording stores each number as a `{{out:inspect-recorded-embeddings:stored as}}`, in 2 bytes, and `lib.recorded` widens them to 32-bit floats when it loads them. That rounding moves the stored lengths slightly off 1, so `lib.recorded` scales each vector back to length 1. The model reads text as **tokens**, word pieces from its own vocabulary, not the terms of [chapter 5's tokenizer](#split-text-into-terms). The longest product text is {{out:inspect-recorded-embeddings:longest product text, tokens}} tokens, inside the model's limit of {{out:inspect-recorded-embeddings:model's input limit, tokens}} tokens, so the model read every product in full.

Notice what the vectors capture that a string match misses. The median similarity between two of the store's queries is {{out:inspect-recorded-embeddings:median similarity of two queries}}, on a scale where 1 means two vectors point the same way ([cosine similarity](#compare-with-cosine-similarity)). Yet `whimsical solar lights` finds `{{out:inspect-recorded-embeddings:whimsical solar lights, nearest other query}}` at {{out:inspect-recorded-embeddings:whimsical solar lights, similarity}}, and `wall design shelf` finds `{{out:inspect-recorded-embeddings:wall design shelf, nearest other query}}` at {{out:inspect-recorded-embeddings:wall design shelf, similarity}}, with {{out:inspect-recorded-embeddings:whimsical solar lights, words in common}} and {{out:inspect-recorded-embeddings:wall design shelf, words in common}} words in common: `lights` and `light` are different strings, and so are `shelves` and `shelf`. The baseline BM25 doesn't stem words, so it sees nothing shared in either pair. [Chapter 5's stemming](#stem-word-forms) would match `lights` with `light`, but not `shelves` with `shelf`.

**Working rule:** Embed products ahead of time and queries at search time, with the same model at the same revision, and follow the model card on prompts: here, a prefix on queries only. Record the model, revision and prompts next to the vectors, as each `manifest.json` in `examples/data/recorded/` does. Check that your longest product fits the model's input limit before you trust its embedding.

## Compare embeddings with cosine similarity {#compare-with-cosine-similarity}

To rank products for a query, you need one number per pair that says how alike two vectors are. **Cosine similarity** is the cosine of the angle between them: 1 when they point the same way, 0 at right angles and −1 when they point in opposite directions. It ignores how long the vectors are. The **dot product** multiplies two vectors number by number and adds up the results, so it equals the cosine times both lengths. Scaling every vector to length 1, called **normalizing**, makes the two the same number, as [Sentence Transformers](https://sbert.net/docs/sentence_transformer/usage/semantic_textual_similarity.html) states. The dot product is then the cheaper choice, because it skips computing the lengths.

```text
dot(q, d)    = q1 × d1 + q2 × d2 + ... + qn × dn
length(q)    = √(q1² + q2² + ... + qn²)
cosine(q, d) = dot(q, d) / (length(q) × length(d))
```

Every query is compared with every product this way. To see the formula at work, take the test query `clear shower curtain` and its most similar product: compute the dot product and the cosine by hand, and check the cosine with scikit-learn {{version:scikit-learn}}. Then break the one assumption that makes them agree. Keep the first 256 of each vector's {{out:inspect-recorded-embeddings:dimensions}} numbers, a shortcut that [the section on dimensions](#truncate-dimensions-test-loss) tests, but skip the rescaling, and rank every query's products both ways:

[Example: cosine-by-hand](../examples/cosine-by-hand.py)

On the store's unit-length vectors, the dot product is {{out:cosine-by-hand:dot product}}, the cosine by hand is {{out:cosine-by-hand:cosine by hand}}, and scikit-learn's `cosine_similarity` gives {{out:cosine-by-hand:cosine from scikit-learn}}, so the three agree.

Shortening breaks that. The shortened vectors have lengths from {{out:cosine-by-hand:shortened product lengths}}, and for {{out:cosine-by-hand:queries whose top 10 changes}} of the {{out:cosine-by-hand:queries}} queries, ranking by dot product gives a different top 10 from ranking by cosine. A query's own length scales all of its scores by the same factor, so it can't change their order. A product's length can: it multiplies that product's score for every query, a boost the shopper never asked for.

Some libraries report a distance instead of a similarity. faiss's L2 indexes return the squared Euclidean distance, and its inner product equals cosine similarity only when the vectors are normalized, according to the [faiss wiki](https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances). For unit vectors, the squared distance is 2 − 2 × cosine, so it ranks products in the same order as cosine, smallest distance first.

**Working rule:** Normalize every vector once, when you store it, and rank by dot product. Normalize again after anything that changes a vector, such as shortening it or averaging several. If a model's documentation says it was trained for the dot product of unnormalized vectors, follow the documentation instead, and never mix the two in one index.

## Find the exact nearest neighbours with NumPy {#exact-nearest-neighbours-numpy}

Retrieval by embeddings is a **nearest-neighbour search**: find the k products whose vectors are most similar to the query's. **Exact** search compares the query with every product and keeps the best k, so it never misses one. With [unit-length vectors](#compare-with-cosine-similarity), comparing one query with every product is one matrix–vector product, and comparing many queries is one matrix product, with a row per query and a column per product. What's left is to find the k highest scores in each row. Sorting every row does more work than needed. NumPy's `argpartition` moves the k highest scores to one side without ordering either side, and [NumPy's documentation](https://numpy.org/doc/stable/reference/generated/numpy.argpartition.html) leaves that order undefined, so you sort only those k.

[Chapter 1's semantic search](#search-by-meaning) took its rankings from `lib.semantic.SemanticSearch`. Build its core by hand for the {{out:load-the-store:test queries}} test queries with NumPy {{version:numpy}}, then check it twice: against `lib.semantic`, and against faiss's `IndexFlatIP`, which computes the inner product with every vector. The [faiss guidelines](https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index) name two indexes that guarantee exact results, `IndexFlatIP` and `IndexFlatL2`:

[Example: exact-search-numpy](../examples/exact-search-numpy.py)

The hand-built search agrees with both checks: it returns the same top 10 as `lib.semantic` for {{out:exact-search-numpy:same top 10 as lib.semantic}} test queries, and as faiss for {{out:exact-search-numpy:same top 10 as faiss IndexFlatIP}}, so its run scores the same mean nDCG@10, {{out:exact-search-numpy:ndcg@10 of the exact run}}. One matrix product computed every similarity: {{out:exact-search-numpy:similarities computed}} of them, {{out:exact-search-numpy:similarities per query}} per query. The book's shared module, `lib.vectors`, keeps this `top_k` for later examples and wraps it in `exact_search`.

Whether exact search is enough comes down to its cost. Its work grows in step with the catalog: one similarity per product, for every query. Its memory is the vectors themselves:

[Example: exact-search-memory](../examples/exact-search-memory.py)

Each product takes {{out:exact-search-memory:float32 bytes per product}} bytes as 32-bit floats, so the store's catalog takes {{out:exact-search-memory:product vectors in memory, MB}} MB, and a million products would take {{out:exact-search-memory:the same for 1,000,000 products, GB}} GB. [Douze et al. (2024)](https://arxiv.org/abs/2401.08281) advise deciding first whether an index is needed at all, because in some cases a brute-force search is the best option. We judge the store's catalog to be such a case.

**Working rule:** We recommend exact search until it no longer fits in memory or meets your latency budget, measured with your own catalog on your own hardware. It has no settings to tune and never misses a neighbour, so it is also the ground truth for any approximate index. Store the normalized vectors as one 32-bit matrix, take the top k with `argpartition`, and sort only those.

## Trade recall for speed with HNSW {#hnsw-recall-for-speed}

When [exact search](#exact-nearest-neighbours-numpy) costs too much, an **approximate nearest-neighbour** (ANN) index skips most of its comparisons, and in return may miss some of the true nearest neighbours. Its quality is its **recall@k against exact search**: the share of exact search's top k that it also returns. That isn't the [recall@k of chapter 3](#count-relevant-in-top-k), which compares a ranking with judgments. An index with perfect recall against exact search returns what exact search returns, good or bad.

**HNSW**, short for hierarchical navigable small world, is a graph index. Each product is a node linked to some of its nearest neighbours, in a stack of layers: the bottom layer holds every product, and each layer above holds fewer. A search starts at the top, moves to whichever linked node is nearest the query, steps down a layer when it gets no closer, and at the bottom explores a list of candidates before it returns the best k ([Malkov and Yashunin, 2018](https://arxiv.org/abs/1603.09320)). In faiss, `M` sets the links per node. More links find more true neighbours, but use more memory ([faiss wiki](https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index)). Two more settings follow. `efConstruction` sets how many candidates the build considers, and `efSearch` how many a search considers. With faiss {{version:faiss-cpu}}, their defaults are 40 and 16, and its graph build is deterministic, because the only build `add()` uses is a deterministic one ([`HNSW.h`](https://raw.githubusercontent.com/facebookresearch/faiss/v1.15.1/faiss/impl/HNSW.h); [changelog](https://raw.githubusercontent.com/facebookresearch/faiss/v1.15.1/CHANGELOG.md)). So the outputs here repeat exactly. Timings differ from machine to machine, so the examples count work instead: faiss counts the distances it computes in each search (`HNSW.h`), which for this index are similarities. The count leaves out following links and reading memory, so it is a guide to speed, not a measure.

The store doesn't need an index, but a store with millions of products would, and a catalog small enough to check every answer is the place to learn the trade-off. Build HNSW over the store's {{out:inspect-recorded-embeddings:products embedded}} products with `hnsw_index(product_vectors, links=32)`, `lib.vectors`'s name for `M = 32`. Then sweep `efSearch` from 16 to 256 on the [train queries](#store-data):

[Example: hnsw-efsearch-sweep](../examples/hnsw-efsearch-sweep.py)

The graph thins fast: the bottom layer holds all {{out:hnsw-efsearch-sweep:products in layer 1}} products, the next {{out:hnsw-efsearch-sweep:products in layer 2}} and the top {{out:hnsw-efsearch-sweep:products in layer 3}}.

The mean hides a failure. At the default `efSearch` of 16, a query computes {{out:hnsw-efsearch-sweep:efSearch 16, similarities computed per query}} similarities instead of {{out:exact-search-numpy:similarities per query}}, and mean recall@10 against exact search is {{out:hnsw-efsearch-sweep:efSearch 16, recall@10 against exact}}. Yet {{out:hnsw-efsearch-sweep:efSearch 16, train queries with recall@10 of 0}} train queries get none of their true top 10.

Those queries are {{out:hnsw-efsearch-sweep:efSearch 16, those queries}}. Look at `foutains with brick look`, one of them: HNSW returns a cushion, a wall decor set and an ivy branch, while exact search's top 3 are brick-look tiles. The search ended in a part of the graph far from its true neighbours.

The smallest `efSearch` in the sweep at which no train query is at 0 is {{out:hnsw-efsearch-sweep:smallest efSearch with no query at 0}}, but the sweep's table shows that its worst query still misses most of its true top 10. So choose by a stricter rule: the smallest `efSearch` at which every train query finds at least 9 of its true top 10. That is {{out:hnsw-efsearch-sweep:smallest efSearch with every query at 0.9 or more}}, with a mean recall of {{out:hnsw-efsearch-sweep:chosen efSearch, recall@10 against exact}} and {{out:hnsw-efsearch-sweep:chosen efSearch, similarities computed per query}} similarities per query.

**Working rule:** Use an ANN index only when exact search is too slow or too large. Tune it on train queries by its recall against exact search, at the k your pipeline uses, and choose by the worst query, not the mean. We recommend the smallest `efSearch` at which every query keeps at least 9 of its true top 10, with `M = 32`. This chapter tunes only `efSearch`; `M` and `efConstruction` change the graph itself, and deserve the same sweep on a large catalog. [Aumüller et al. (2018)](https://arxiv.org/abs/1807.05614) find that very different ANN methods reach comparable quality-performance trade-offs, so measure on your own data, not by reputation.

## Check the index on the test queries {#hnsw-on-test-queries}

An index tuned on train queries still has to be checked on queries it hasn't seen, by its [recall against exact search](#hnsw-recall-for-speed) and by its effect on the judged scores. These are the chapter's {{out:hnsw-test-recall:planned test comparisons}} planned test comparisons: the HNSW index at faiss's default `efSearch` of 16, and at the setting chosen on the train queries, each against exact search. A difference is established when its [paired permutation test](#permutation-test-on-queries) gives p below {{out:hnsw-test-recall:p-value threshold}}, with a [t-test and an interval](#when-a-difference-is-established) at the same level beside it. The search also asks for `k = 100`, because [hybrid retrieval](#fuse-with-rrf) fuses each retriever's top 100:

[Example: hnsw-test-recall](../examples/hnsw-test-recall.py)

The setting chosen on train queries holds up. Its recall@10 against exact search is {{out:hnsw-test-recall:chosen, recall@10 against exact}}, its mean nDCG@10 is {{out:hnsw-test-recall:chosen, hnsw ndcg@10}}, the same as exact search's {{out:hnsw-test-recall:exact ndcg@10}}, and {{out:hnsw-test-recall:chosen, queries whose ndcg@10 changes}} queries change score.

At faiss's default setting, what's lost is recall, not measured quality. {{out:hnsw-test-recall:default, queries with recall@10 of 0}} test queries get none of their true top 10 ({{out:hnsw-test-recall:default, those queries}}), and the mean nDCG@10 falls to {{out:hnsw-test-recall:default, hnsw ndcg@10}}, but only {{out:hnsw-test-recall:default, queries whose ndcg@10 changes}} queries change score, and neither test's p-value falls below that threshold, so the fall isn't established.

The 97.5% interval, from {{out:hnsw-test-recall:default, 97.5% interval}}, excludes 0 and disagrees with the tests, as happens when so few queries carry a difference.

Asking for more results doesn't make up for a small `efSearch`: at the default, recall@100 is {{out:hnsw-test-recall:default, recall@100 against exact}}, against {{out:hnsw-test-recall:chosen, recall@100 against exact}} at the chosen setting.

The graph also costs memory, on top of the vectors:

[Example: hnsw-index-memory](../examples/hnsw-index-memory.py)

Each product has {{out:hnsw-index-memory:links per product, upper layers}} links on the upper layers and {{out:hnsw-index-memory:links per product, bottom layer}} on the bottom one, which is twice `M`. faiss estimates 4d + 8M bytes per vector ([faiss wiki](https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index)): 4 bytes for each of the d numbers, and 4 bytes for each of the 2M links on the bottom layer, {{out:hnsw-index-memory:faiss's estimate, 4d + 8M bytes per product}} here. The saved index comes close, at {{out:hnsw-index-memory:bytes per product, hnsw index}} bytes per product, against {{out:hnsw-index-memory:bytes per product, exact index}} for the exact index.

**Working rule:** Check a tuned index on held-out queries by its recall against exact search, and read the queries that lose their whole top k before the mean score. Treat a fall in nDCG@10 that rests on a handful of queries as unproven until a paired test establishes it. Budget memory for the links as well as the vectors.

## Check queries that contain digits {#embeddings-match-digits}

[An index](#hnsw-on-test-queries) can only return what the embeddings put near a query, so the next question is which queries the embeddings themselves get wrong. An encoder squeezes a whole query into one vector, so a short token that changes the meaning, such as a size, a model number or a negation, may move that vector very little. [Deng et al. (2026)](https://aclanthology.org/2026.findings-eacl.289/) measured this for numbers, in a study of 13 embedding models on synthetic financial text. Their accuracy at telling texts apart by a number was 0.54, only slightly above the 0.5 of random guessing, and four-digit numbers were at the level of guessing. Products aren't financial reports, so check the store's queries.

Few of the store's queries hold a digit, so compare BM25 and semantic retrieval on each of those that do, from both splits. Nothing is tuned on these queries, and a handful can't establish anything:

[Example: digits-in-queries](../examples/digits-in-queries.py)

Neither ranker owns the queries with digits. Semantic retrieval scores higher on {{out:digits-in-queries:semantic better}} of them and BM25 on {{out:digits-in-queries:bm25 better}}, and {{out:digits-in-queries:gaps under 0.03}} of the gaps are under 0.03. BM25's clearest loss is `7qt slow cooker`: the word `7qt` appears in {{out:digits-in-queries:products whose text has the word 7qt}} products, and {{out:digits-in-queries:products whose text has 7 qt}} write `7 qt`.

On the test query `48 inch kitchen hood`, BM25 scores higher, so here is semantic retrieval's top 10:

[Example: hood-query-digits](../examples/hood-query-digits.py)

For this query, the embedding matched the digits as text, not the width they stand for. Of semantic retrieval's top 10, {{out:hood-query-digits:semantic, width 48 in the top 10}} hoods have a width of 48. The others are a 36-inch hood from a series named `ws-48` and a 30-inch hood rated at 480 cfm, cubic feet per minute, the airflow of its fan. Both contain the digits 48, neither is 48 wide, and the judges marked both [partial](#store-data). BM25's top 10 holds {{out:hood-query-digits:bm25, width 48 in the top 10}} hoods of width 48.

Each hood's width comes from the `attributes` column, which holds curated WANDS product features. `product_text`, the text the model embedded, leaves that column out, so the embeddings never saw it, and a size the shopper types can be checked against it exactly.

To see how the model handles sizes in general, use **probe queries**: pairs of queries written for the book that differ only in a size, a code or a negation, which `recordings/probes.py` embeds with the same model. One of them repeats the test query `48 inch kitchen hood` word for word. Probes have no judgments, so the example counts their top 10s and never scores them. The right size comes from the width in `attributes` for the hoods, and from the name for the rest:

[Example: probe-pair-overlap](../examples/probe-pair-overlap.py)

For these probes, the model mostly finds the right size. The two hood probes share {{out:probe-pair-overlap:hood, products in both top 10s}} products, yet each top 10 holds hoods of the right width, {{out:probe-pair-overlap:36 inch kitchen hood, right size in the top 10}} for 36 inches and {{out:probe-pair-overlap:48 inch kitchen hood, right size in the top 10}} for 48; the slow-cooker probes find {{out:probe-pair-overlap:4 qt slow cooker, right size in the top 10}} and {{out:probe-pair-overlap:7 qt slow cooker, right size in the top 10}} of the right capacity, and the bed probes {{out:probe-pair-overlap:twin bed, right size in the top 10}} and {{out:probe-pair-overlap:king bed, right size in the top 10}} of the right size.

No product name gives a 2- or 4-seat count ({{out:probe-pair-overlap:catalog names that give a seat count}} of them), so the top 10 for `2 seat sofa` holds {{out:probe-pair-overlap:2 seat sofa, same digit but a piece count or 2 in 1}} products that share only the digit, two-piece sets and "2 in 1" kids' sofas, and {{out:probe-pair-overlap:2 seat sofa, right size in the top 10}} of the catalog's {{out:probe-pair-overlap:catalog products named loveseat}} loveseats.

### Expect a code to carry little weight {#codes-carry-no-weight}

A product code should match one product exactly. To an encoder, it's a rare string of letters and digits. Two pairs of the [probe queries](#embeddings-match-digits) test codes: one built on a code no product holds, taken from the train query `ines shag rug w003096809`, and one built on a screen resolution that exactly one product's text holds:

[Example: code-probes](../examples/code-probes.py)

Changing the last digits of the unknown code leaves the two queries at a similarity of {{out:code-probes:code, similarity of the two queries}}, and their top 10s share {{out:code-probes:code, products in both top 10s}} products. A code that matches nothing changes nothing, and BM25 has nothing to match either.

A code that one product holds decides BM25's ranking and only nudges the embedding's. The resolution `1920x1080p` appears only in the text of the {{out:code-probes:that product}}. BM25 ranks it {{out:code-probes:1920x1080p projector, its bm25 rank}} for `1920x1080p projector`; for `2560x1440p projector` it is {{out:code-probes:2560x1440p projector, its bm25 rank}}. [Sciavolino et al. (2021)](https://arxiv.org/abs/2109.08535) found that dense retrievers do drastically worse than sparse methods on questions about rare entities, while [Karpukhin et al. (2020)](https://arxiv.org/abs/2004.04906) describe BM25 as responsive to highly selective keywords and phrases.

The embedding ranks it {{out:code-probes:1920x1080p projector, its semantic rank}} with its own code and {{out:code-probes:2560x1440p projector, its semantic rank}} with the other, because the store has only {{out:code-probes:products in the Projectors class}} projectors. The code shows only in the margin over the next product: {{out:code-probes:1920x1080p projector, its similarity}} against {{out:code-probes:1920x1080p projector, similarity of the next product}} with it, and {{out:code-probes:2560x1440p projector, its similarity}} against {{out:code-probes:2560x1440p projector, similarity of the next product}} without.

**Working rule:** List the queries in your log that hold sizes, quantities or codes, and compare both rankers on them before you trust either. When a code matters, match it lexically. When an exact size matters, match it against a product field such as the width in `attributes`, or lexically once the tokenizer keeps digits and units are written one way ([chapter 5](#split-text-into-terms), [chapter 8](#normalize-units)). Leave the rest of the query to semantic retrieval. [Chapter 7](#when-hybrid-helps) tests combining the two.

## Test embeddings on negation {#embeddings-on-negation}

Negation reverses what a shopper wants with one short word: a rug that is *not* round. An embedding of the whole query can stay close to the same query without that word. NevIR, a benchmark of document pairs that differ by a negation, found that no bi-encoder architecture scores above 12% paired accuracy, where ranking at random scores 25% ([Weller et al., 2024](https://arxiv.org/abs/2305.07614)). In a 2025 reproduction, [van den Elsen et al.](https://arxiv.org/abs/2502.13506) found that one 7-billion-parameter bi-encoder, GritLM-7B, reached 39.0%, that other bi-encoders improved little, and that only cross-encoders and listwise LLM rerankers handled both negation tasks reasonably well.

{{out:negation-probes:store queries with no, not, without or non}} of the store's queries contain "no", "not", "without" or "non", so the judgments can't test negation. {{out:negation-probes:negation probe pairs}} pairs of [probe queries](#embeddings-match-digits) stand in instead, each a query and its negation, and we count what each top 10 holds.

The catalog could satisfy every negated probe: it has {{out:negation-probes:products named armless}} products named armless, {{out:negation-probes:coffee tables that never mention glass}} coffee tables that never mention glass, and {{out:negation-probes:area rugs not named round}} area rugs not named round.

Before the run, guess how many of semantic retrieval's top 10 for `rug that is not round` are named round.

[Example: negation-probes](../examples/negation-probes.py)

If you guessed few, you read "not" the way a shopper does, and the embedding didn't. In the rug and coffee-table probes it kept the attribute and lost the "not". Semantic retrieval's top 10 for `rug that is not round` holds {{out:negation-probes:rug that is not round, semantic top 10 with round}} products named round, and its top 10 for `round rug` holds {{out:negation-probes:round rug, semantic top 10 with round}}. For `coffee table not glass`, {{out:negation-probes:coffee table not glass, semantic top 10 with glass}} of the top 10 mention glass, against {{out:negation-probes:coffee table with glass top, semantic top 10 with glass}} for `coffee table with glass top`.

`sofa without arms` goes the other way, and {{out:negation-probes:sofa without arms, semantic top 10 with armless}} of its top 10 are named armless, against {{out:negation-probes:sofa with arms, semantic top 10 with armless}} for `sofa with arms`. The catalog has one word for that negation, armless, and the model places "without arms" near it.

BM25 does no better on these probes. Its tokenizer drops "not", "that" and "is" as [stop words](#split-text-into-terms), so it searches `rug that is not round` as `{{out:negation-probes:bm25 terms of rug that is not round}}`. Its top 10s hold {{out:negation-probes:rug that is not round, bm25 top 10 with round}} round rugs and {{out:negation-probes:sofa without arms, bm25 top 10 with armless}} armless sofas.

These probes are a few pairs, not a measurement, and the sofa pair goes the other way. But the rug and coffee-table pairs agree with NevIR: the query vector stays near the attribute the query names.

**Working rule:** Don't rely on a first-stage retriever, lexical or semantic, to handle negation. When shoppers exclude an attribute your catalog stores, such as shape, material or arms, detect the exclusion and filter on the attribute. Otherwise, rerank, since in van den Elsen et al.'s reproduction cross-encoders and LLM rerankers handled negation far better. Chapter 9 reranks the store's semantic retrieval with [a cross-encoder](#rerank-semantic-with-cross-encoder) and [LLMs](#llm-listwise-reranking), and none of its rerankers beats semantic retrieval on the store's judged queries, which can't test negation. Keep probes like these as a check to rerun whenever you change models or add a reranker.

## Choose an embedding model {#choose-embedding-model}

The model decides what the vectors can know, and changing it later means embedding the whole catalog again. Public benchmarks narrow the field. The store's own judgments decide.

### Read benchmarks as a shortlist {#benchmarks-as-a-shortlist}

MTEB, the Massive Text Embedding Benchmark, scores models on many tasks. Its authors found that no single method dominates across all of them ([Muennighoff et al., 2023](https://aclanthology.org/2023.eacl-main.148/)). Its authors also found that its averages lean toward tasks with many datasets, notably retrieval, classification and clustering, so read the scores for the task you have.

Public scores can also overstate performance on data a model hasn't seen. The makers of RTEB, a retrieval benchmark, describe a gap between models' reported scores on public benchmarks and their actual performance on new, unseen data ([Liu et al., 2025](https://huggingface.co/blog/rteb)). RTEB keeps 28 of its 48 datasets private. [Verma et al. (2026)](https://neurips.cc/virtual/2026/poster/139103) find that RTEB's rankings of models correlate only at ρ = 0.38 with those on MTEB and on BEIR ([Thakur et al., 2021](https://arxiv.org/abs/2104.08663)), an earlier benchmark across many domains. Here ρ is a rank correlation, where 1 means the same order. Size isn't quality either. [Enevoldsen et al. (2025)](https://arxiv.org/abs/2502.13595) report that in MMTEB, MTEB's multilingual successor, the best-performing publicly available model is multilingual-e5-large-instruct, which has only 560 million parameters.

The store considered four open models, and none of their cards reports results on WANDS. They are [the store's encoder](#turn-text-into-embeddings) and `Qwen/Qwen3-Embedding-0.6B` ([card](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)). The other two are `BAAI/bge-base-en-v1.5` ([card](https://huggingface.co/BAAI/bge-base-en-v1.5)) and `ibm-granite/granite-embedding-english-r2` ([card](https://huggingface.co/ibm-granite/granite-embedding-english-r2)). Product-search benchmarks such as [ESCI](#gains-are-a-choice), which holds about 130,000 queries and 2.6 million judgments ([Reddy et al., 2022](https://arxiv.org/abs/2206.06588)), are closer to the store's task, but they still aren't its catalog. So the store's shortlist rests on checks a leaderboard can't make: a licence that allows commercial use, English, a size that runs without special hardware, and documented support for shorter vectors. The card of the store's choice reports 55.14 nDCG@10 on MTEB's retrieval tasks, a vendor's measurement on general text, not on products ([model card](https://huggingface.co/Snowflake/snowflake-arctic-embed-m-v1.5)). To choose between candidates, record each on the same product text, score each on the train queries, and compare them with [the book's paired tests](#when-a-difference-is-established), as [the next subsection](#truncate-dimensions-test-loss) does.

### Truncate dimensions, then test the loss {#truncate-dimensions-test-loss}

Every dimension costs 4 bytes per product as a 32-bit float, and every similarity multiplies all of them, so a shorter vector is cheaper to store and to search. **Matryoshka** training teaches a model to put the most useful information in its first dimensions, so that the start of a vector works as a smaller embedding on its own. [Kusupati et al. (2022)](https://arxiv.org/abs/2205.13147) report embeddings up to 14 times smaller for ImageNet-1K classification at the same level of accuracy. Shortening saves storage and search, not encoding: a Matryoshka model is no faster to run, as [Sentence Transformers' documentation](https://sbert.net/examples/sentence_transformer/training/matryoshka/README.html) notes. Only models trained this way are built to survive it, and a shortened vector isn't unit length, so [normalize it again](#compare-with-cosine-similarity), as the model card's own example does.

The [store's encoder](#turn-text-into-embeddings) has a [model card](https://huggingface.co/Snowflake/snowflake-arctic-embed-m-v1.5) that documents shortening to 256 dimensions. `lib.recorded.embeddings(dims=...)` keeps the first `dims` numbers of each vector and normalizes again. Score each length on the train queries:

[Example: truncate-embeddings](../examples/truncate-embeddings.py)

Shortening to 256 dimensions cuts a product from {{out:truncate-embeddings:768 dimensions, float32 bytes per product}} bytes to {{out:truncate-embeddings:256 dimensions, float32 bytes per product}}, and moves mean nDCG@10 from {{out:truncate-embeddings:768 dimensions, ndcg@10}} to {{out:truncate-embeddings:256 dimensions, ndcg@10}}.

At 128 dimensions the mean is {{out:truncate-embeddings:128 dimensions, ndcg@10}}, with [judged@10](#score-every-test-query) at {{out:truncate-embeddings:128 dimensions, judged@10}} against {{out:truncate-embeddings:768 dimensions, judged@10}}, so part of that fall could come from products nobody judged.

Is the change real, or noise among {{out:load-the-store:train queries}} queries? Test it with `lib.significance.compare`, which runs [chapter 3's permutation test](#permutation-test-on-queries) and its [paired t-test](#paired-t-test-on-queries). It also reports a [bootstrap interval](#report-the-size-of-a-gap). These comparisons use train queries, so they tune the vector size; they aren't evidence about the test queries:

[Example: truncation-paired-test](../examples/truncation-paired-test.py)

At 128 dimensions the loss is clear. The mean difference is {{out:truncation-paired-test:128 vs 768, mean difference in ndcg@10}}, and {{out:truncation-paired-test:128 vs 768, queries worse}} train queries get worse. The permutation test gives p {{out:truncation-paired-test:128 vs 768, permutation test p}}, and the 95% interval, from {{out:truncation-paired-test:128 vs 768, 95% bootstrap interval}}, excludes 0.

At 256 dimensions the evidence is borderline, because the tests and the interval disagree. The mean difference is {{out:truncation-paired-test:256 vs 768, mean difference in ndcg@10}}; the permutation test gives p = {{out:truncation-paired-test:256 vs 768, permutation test p}} and the t-test p = {{out:truncation-paired-test:256 vs 768, paired t-test p}}, yet the interval, from {{out:truncation-paired-test:256 vs 768, 95% bootstrap interval}}, excludes 0 by a hair. The [effect size](#report-the-size-of-a-gap), {{out:truncation-paired-test:256 vs 768, effect size}} at 256 dimensions against {{out:truncation-paired-test:128 vs 768, effect size}} at 128, says any loss at 256 is the smaller of the two.

For the store, the decision follows from its size, not from the test. Its full vectors take [{{out:exact-search-memory:product vectors in memory, MB}} MB](#exact-nearest-neighbours-numpy), so there is little to save; it keeps all {{out:inspect-recorded-embeddings:dimensions}} dimensions, and its test score doesn't change. A catalog of hundreds of millions of products would weigh a cut from {{out:truncate-embeddings:768 dimensions, float32 bytes per product}} to {{out:truncate-embeddings:256 dimensions, float32 bytes per product}} bytes per product against a loss that this many train queries can't pin down, and would judge more queries before it decided.

**Working rule:** Shortlist models on licence, language, size and domain, and treat benchmark ranks as a hint. Decide between models, or between vector sizes, on your own train queries with a paired test, and read the interval as well as the p-value. Shorten vectors only for a model trained for it, normalize after shortening, and only when memory or search time requires it.

## What goes wrong in production {#semantic-retrieval-in-production}

**A similarity is not a confidence.** It's tempting to show nothing when the best similarity is low. But similarities aren't on one scale across queries:

[Example: similarity-scale](../examples/similarity-scale.py)

The two extremes among the test queries show it. `{{out:similarity-scale:first query, lowest best similarity}}` has the weakest best match, at {{out:similarity-scale:its best similarity}}, yet its top 10 holds {{out:similarity-scale:exact products in its top 10}} exact products, while `{{out:similarity-scale:second query, highest 10th similarity}}` has the most similar 10th match, at {{out:similarity-scale:its 10th similarity}}.

A cutoff of 0.5 would keep {{out:similarity-scale:products above 0.5 for the first query}} products for the first query and {{out:similarity-scale:products above 0.5 for the second query}} for the second, and nothing for {{out:similarity-scale:test queries with no product above 0.5}} test queries. Nor does semantic search ever come back empty: a [code that no product holds](#codes-carry-no-weight) still gets a full top 10. Decide that a query has no match some other way, for example when lexical retrieval finds none of its rare terms.

**Mismatched vectors fail silently.** Query and product vectors must come from the same model, revision and prompts. Vectors from two models are numbers in different spaces, and when the two have the same dimension, comparing them raises no error. Sentence Transformers' `encode_query` adds the model's predefined query prompt when one exists ([API reference](https://sbert.net/docs/package_reference/sentence_transformer/SentenceTransformer.html)). Make sure the code that embeds live queries does what the indexing code did.

**An index doesn't follow the catalog by itself.** faiss's HNSW does not support removing vectors, according to [the faiss wiki](https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index), so a discontinued product stays in the graph until you filter it out of results or rebuild. [Douze et al. (2024)](https://arxiv.org/abs/2401.08281) say that faiss is not a database: it has no concurrent writes, sharding or transactions. Recall against exact search belongs to one catalog and one set of settings, so measure it again after every rebuild.

**Long texts are cut short.** The model reads at most {{out:inspect-recorded-embeddings:model's input limit, tokens}} tokens, and [Sentence Transformers' API reference](https://sbert.net/docs/package_reference/sentence_transformer/SentenceTransformer.html) says that longer inputs are truncated. The store's longest product is {{out:inspect-recorded-embeddings:longest product text, tokens}} tokens, but in a catalog with long descriptions the model never reads what falls past the limit. Count tokens before you embed, as `recordings/embeddings.py` does.

**Typos move vectors.** [Zhuang and Zuccon (2021)](https://arxiv.org/abs/2108.12139) found that for BERT-based retrievers not trained on misspellings, typos can cause a significant drop in retrieval and ranking effectiveness. {{out:spelling-probes:spelling probes}} of the store's queries contain a typo, and each one has a correctly spelled probe. A corrected probe asks for the same thing as its query, so that query's judgments can score it:

[Example: spelling-probes](../examples/spelling-probes.py)

One typo can change a query's whole semantic top 10. For the train query `foutains with brick look`, the typed and corrected spellings share {{out:spelling-probes:foutains, products in both semantic top 10s}} products, and nDCG@10 goes from {{out:spelling-probes:foutains, semantic ndcg@10 as typed}} to {{out:spelling-probes:foutains, semantic ndcg@10 corrected}}; BM25 moves too, from {{out:spelling-probes:foutains, bm25 ndcg@10 as typed}} to {{out:spelling-probes:foutains, bm25 ndcg@10 corrected}}.

For the other probes the scores move less: the one test query among them, `refrigerator with ice an water in door`, goes from {{out:spelling-probes:ice-an, semantic ndcg@10 as typed}} to {{out:spelling-probes:ice-an, semantic ndcg@10 corrected}}. None of this establishes a pattern: with {{out:spelling-probes:spelling probes}} probes, the smallest two-sided permutation p-value possible is {{out:spelling-probes:smallest two-sided permutation p-value possible with 4 queries}}.

**Working rule:** Don't filter on a fixed similarity. Record the model, revision and prompts with every vector, and check them where queries are embedded. Rebuild and re-measure the index as the catalog changes, count tokens before you embed, and correct spelling before you embed a query when your log shows typos, with a corrector that checks each word's neighbours ([chapter 8](#correct-spelling-from-catalog)).

## Decisions this chapter made {#semantic-retrieval-decisions}

Each choice below is a default, measured on the train queries where it could be.

- **Model:** `Snowflake/snowflake-arctic-embed-m-v1.5`, which `recordings/embeddings.py` records once at a pinned revision. It is open, under Apache-2.0, small, and trained for shorter vectors. Queries get the prefix its card asks for; products don't.
- **Similarity:** the dot product of unit-length vectors, which is cosine similarity. Vectors are normalized once, and again after any change.
- **Search:** exact, with NumPy, through `lib.vectors.exact_search`, because the store's vectors take {{out:exact-search-memory:product vectors in memory, MB}} MB. It is also the ground truth for any index.
- **If the catalog outgrows exact search:** faiss HNSW with `M = 32` and an `efSearch` of {{out:hnsw-test-recall:efSearch chosen on train queries}}, the smallest setting in the train sweep at which every query keeps at least 9 of its true top 10. On the test queries, its recall@10 against exact search is {{out:hnsw-test-recall:chosen, recall@10 against exact}}.
- **An index's recall is measured against exact search,** on train queries, at the k the pipeline uses, reading the worst queries as well as the mean.
- **Dimensions:** all {{out:inspect-recorded-embeddings:dimensions}}. Shortening to 256 would cut each product from {{out:truncate-embeddings:768 dimensions, float32 bytes per product}} to {{out:truncate-embeddings:256 dimensions, float32 bytes per product}} bytes, at a cost the train queries can't pin down, and the store has little to save.
- **Probe queries test what the judgments can't,** such as negation, codes and spelling. They are compared by their top 10s, apart from the judged queries; only a corrected spelling, which asks the same thing, is scored with its query's judgments.
- **Known weak spots go to other stages:** codes go to lexical retrieval, and exact sizes go to attribute fields, or to lexical retrieval once the tokenizer keeps digits ([chapter 5](#split-text-into-terms), [chapter 8](#normalize-units)). For both, a filter to test rather than a blend ([chapter 7](#when-hybrid-helps)). Negation goes to attribute filters and reranking.
- **What to carry away:** An index is only as good as its measured recall against exact search, on the catalog and settings you ship. The probes found the embeddings handling most sizes, but not a code, and not the negations in the rug and coffee-table pairs; a few pairs prove nothing, but they mark the queries to route elsewhere.
