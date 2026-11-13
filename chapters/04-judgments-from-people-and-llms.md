---
chapter_id: judgments-from-people-and-llms
---

# Judgments from people and LLMs

By the end of this chapter you can write judging guidelines, measure how well two judges agree with kappa and weighted kappa, read an LLM judge's errors from a confusion matrix, and decide where LLM judgments can stand in for people's and where they can't.

Every score in this book rests on the judgments of people who graded search results from [the store](#store-data), an online furniture and home-goods retailer. Those judgments are WANDS's labels, each the majority vote of up to three trained annotators, and they have holes: on the [test queries](#store-data), BM25's top 10s hold {{out:fill-unjudged-pairs:unjudged products in the top 10s, bm25}} products nobody judged ([Count unjudged products](#unjudged-results)). They also take people's time, and the agreement among WANDS's annotators rose over four months of calibration ([Expect judges to disagree](#judges-disagree-on-some-pairs)). A **large language model** (LLM), a model trained on large amounts of text to generate text in response to a prompt, can grade any query and product it is shown. Before the store lets one judge, it needs to know how often the LLM agrees with the store's judges, and whether its grades would lead to the same decisions about the rankers.

The LLM's grades come from a **recording**: `recordings/llm_judgments.py` ran the open model `{{out:print-judging-guidelines:model}}` once, locally, at temperature 0 with a fixed seed, and saved its answers, so the examples read them through `lib.recorded.llm_judgments()` without calling the model. We chose the model because it is open, under the Apache 2.0 licence, and runs locally, so anyone can rerun the recording. A JSON schema limits each answer to one of the three labels. The recording graded each pair with two prompts, `short` and `guidelines` ([Write judging guidelines](#write-judging-guidelines)), and took its pairs from two sets, which this chapter calls pools. The **train pool**, for measuring agreement, holds pairs people judged for train queries, drawn evenly by label: up to 4 of each label per query, {{out:kappa-by-hand:pairs}} pairs in all. The **test pool** holds the pairs an LLM would grade to score the store's rankers: every product in the top 10 of [chapter 1's four rankers](#search-end-to-end-decisions) for each test query, judged by people or not, {{out:fill-unjudged-pairs:pairs in the test pool}} pairs. The four are BM25, semantic retrieval on embeddings from `Snowflake/snowflake-arctic-embed-m-v1.5`, the hybrid of the two, and the hybrid reranked by the cross-encoder `Alibaba-NLP/gte-reranker-modernbert-base`. Each measurement of the LLM judge holds for the pool it is taken on, so this chapter names the pool every time.

The scores that decide between rankers stay on human grades. The chapter's planned comparisons on the test queries are [chapter 3's three](#when-a-difference-is-established), rescored with each prompt's LLM grades at chapter 3's threshold of {{out:planned-comparisons-llm-grades:p-value threshold}}: semantic retrieval against BM25, the hybrid against semantic retrieval, and the reranked hybrid against the hybrid.

## Write judging guidelines {#write-judging-guidelines}

A **judgment** is a judge's label for one pair of a query and a product, and a **grade** is the number that stands for it. [Chapter 1](#store-data) defines a judgment as a person's label; this chapter widens it to an LLM's label too. Judgments depend on the instructions behind them. **Judging guidelines** say what each grade means, how to choose between grades, and what to do with the cases that fall between them.

WANDS published the guidelines its annotators used: eight pages, under the MIT licence ([annotation guidelines](https://github.com/wayfair/WANDS/raw/3b74dcf4ba29ab8ff3e6a50b5b09fc627cb882b5/Product%20Search%20Relevance%20Annotation%20Guidelines.pdf)). For each pair they list what a judge sees: the query, and the product's ID, name, image and web page. They define the three grades and add two notes, one asking the judge to understand the query first, and one saying to ignore misspellings and annotate the pair as written. Then they teach by example: a wire basket without dividers is partial for `wire basket with dividers`, one of the store's own train queries, and a salon anti-fatigue mat is irrelevant to `salon chair`, because the customer wants a chair and not a mat. Guidelines also change as judges use them. WANDS's authors say their annotators' agreement rose over four months, and among the reasons they give are a daily routine for discussing conflicting annotations and refining the guidelines with the annotators' input ([Chen et al., 2022, section 5](https://easychair.org/publications/preprint/wm4R)). One web search engine's rater guidelines do the same, at 182 pages in their version of September 11, 2025 ([Search Quality Rater Guidelines](https://guidelines.raterhub.com/searchqualityevaluatorguidelines.pdf)).

The `short` prompt is WANDS's three definitions alone. `guidelines` keeps them and adds rules, and it is the prompt this chapter argues about, so print it as the recording sent it:

[Example: print-judging-guidelines](../examples/print-judging-guidelines.py)

The `guidelines` prompt is the longer of the two, at {{out:print-judging-guidelines:words in the guidelines prompt}} words against {{out:print-judging-guidelines:words in the short prompt}} for `short`. Its first two rules spell out WANDS's definition of partial: find the entity, then the modifiers. The accessory rule settles a case that WANDS's own labels split: for `clear shower curtain`, WANDS lists one set of shower curtain hooks twice, once as partial and once as irrelevant ([Expect judges to disagree](#judges-disagree-on-some-pairs)). The misspelling rule follows WANDS's note. The last rule has no counterpart in WANDS's guidelines, because the store's data holds no images, so the recording shows the LLM text only. Neither prompt has worked examples, which WANDS's guidelines rely on.

**Working rule:** Write guidelines as definitions, a procedure, a worked example with a reason for each grade, and a rule for each case your judges have disagreed on. Log disagreements and revise the guidelines when one kind keeps coming back. Give an LLM judge the same guidelines as people, plus a rule for what it can't see, and then measure what it does, because a rule you write isn't a rule the judge follows.

## Measure agreement beyond chance with kappa {#agreement-beyond-chance}

Before you trust a second judge, measure how often it agrees with the first. The share of pairs on which two judges give the same grade, the **observed agreement**, is the obvious number, and it flatters. Two judges who guessed independently would agree on some pairs by chance, and on more of them when both use one label often. **Cohen's kappa** corrects for that by comparing the observed agreement with the agreement expected if each judge kept its own label shares but chose independently of the other. A kappa of 1 means complete agreement, and 0 means "no agreement beyond what would be expected by chance" ([scikit-learn documentation](https://scikit-learn.org/1.9/modules/generated/sklearn.metrics.cohen_kappa_score.html)):

```text
kappa    = (observed - expected) / (1 - expected)
expected = sum over grades g of  (share of pairs judge A gives g) × (share of pairs judge B gives g)
```

The question is how often the LLM of [the recording](#judgments-from-people-and-llms) grades a pair the way the store's judges did. Answer it on the train pool, with the [`short` prompt](#write-judging-guidelines) (WANDS's three definitions alone). Build the 3×3 table of counts by hand, compute kappa from it, and check the result against scikit-learn's `cohen_kappa_score`:

[Example: kappa-by-hand](../examples/kappa-by-hand.py)

The LLM and the store's judges give the same grade to {{out:kappa-by-hand:observed agreement, short prompt}} of the train pairs, and chance alone would produce {{out:kappa-by-hand:agreement expected by chance, short prompt}}. Kappa removes that chance share, which leaves {{out:kappa-by-hand:kappa by hand, short prompt}} for `short`, the same value scikit-learn computes.

With `guidelines`, the two sets of grades agree on {{out:kappa-by-hand:observed agreement, guidelines prompt}} of the pairs, a little more than `short` does, and kappa is {{out:kappa-by-hand:kappa from scikit-learn, guidelines prompt}}.

Is that good? Scales that call a kappa "moderate" or "substantial" are conventions from particular fields. Artstein and Poesio's survey concludes that choosing an adequate level of agreement for a specific purpose is still little more than a black art ([Artstein and Poesio, 2008](https://aclanthology.org/J08-4004/)).

Published figures give context, not a bar. WANDS's annotators agreed with each other at a kappa of 0.467 in their first two months and 0.826 in their fourth ([Chen et al., 2022, Table 2](https://easychair.org/publications/preprint/wm4R)).

On a stratified sample of pairs from TREC, the Text REtrieval Conference's shared evaluations, Thomas et al. found that GPT-4's kappa against the assessors ran from as low as 0.20 up to 0.64, depending on the prompt. Two groups of trained human assessors reached 0.52 on TREC ad hoc documents that they judged a second time ([Cormack et al., 1998, as reported by Thomas et al., 2024](https://arxiv.org/abs/2309.10621)). Each figure comes from a different sample, and [kappa depends on the sample's label mix](#kappa-depends-on-label-mix).

**Working rule:** Report kappa, with observed agreement beside it, never observed agreement alone. Measure it on pairs people judged for train queries, and label any threshold you use as a rule of thumb. With more than two judges or missing labels, use Krippendorff's alpha, which compares observed with chance disagreement and handles any number of observers and incomplete data ([Krippendorff, 2011](https://www.asc.upenn.edu/sites/default/files/2021-03/Computing%20Krippendorff%27s%20Alpha-Reliability.pdf)).

## Give near misses partial credit with weighted kappa {#weighted-kappa-for-grades}

[Kappa](#agreement-beyond-chance) counts every disagreement the same. The store's grades are ordered, exact above partial above irrelevant, and calling an exact product partial is a smaller error than calling it irrelevant. In nDCG@10 ([Score one ranking](#score-one-ranking)) it costs half as much gain: one grade instead of two. **Weighted kappa** gives each disagreement a weight by its distance and compares the weighted disagreement with the weighted disagreement expected by chance ([Artstein and Poesio, 2008, section 2.6](https://aclanthology.org/J08-4004/)). With **linear** weights, a disagreement one grade apart weighs 0.5 and one two grades apart weighs 1:

```text
weighted kappa = 1 - (sum of weight × observed count) / (sum of weight × count expected by chance)
```

With every disagreement weighted 1, the formula gives plain kappa. scikit-learn's `cohen_kappa_score` computes all three kinds this way: weight 1 by default, the distance between grades with `weights="linear"`, and its square with `weights="quadratic"` ([scikit-learn source](https://raw.githubusercontent.com/scikit-learn/scikit-learn/1.9.1/sklearn/metrics/_classification.py)). Scaling the weights doesn't change the result.

For both [prompts](#write-judging-guidelines), the example builds the linear weights and weighted sums by hand on the evenly drawn train pairs of [Measure agreement beyond chance](#agreement-beyond-chance), then checks each result against scikit-learn:

[Example: weighted-kappa-by-hand](../examples/weighted-kappa-by-hand.py)

Weighting raises both kappas, because the disagreements sit one grade apart more often than chance would put them there. For `short`, {{out:weighted-kappa-by-hand:disagreements one grade apart, share, short prompt}} of the disagreements are one grade apart, against {{out:weighted-kappa-by-hand:one grade apart, share expected by chance, short prompt}} expected by chance. Weighted kappa is {{out:weighted-kappa-by-hand:linear weighted kappa by hand, short prompt}}, against {{out:weighted-kappa-by-hand:kappa, short prompt}} plain, and it matches scikit-learn's.

For `guidelines`, {{out:weighted-kappa-by-hand:disagreements one grade apart, share, guidelines prompt}} of the disagreements are one grade apart, against {{out:weighted-kappa-by-hand:one grade apart, share expected by chance, guidelines prompt}} expected by chance. Weighted kappa is {{out:weighted-kappa-by-hand:linear weighted kappa by hand, guidelines prompt}}, against {{out:weighted-kappa-by-hand:kappa, guidelines prompt}} plain, and both weighted values match scikit-learn's.

The prompts also differ in their far misses, the pairs two grades apart: {{out:weighted-kappa-by-hand:pairs two grades apart, short prompt}} for `short` and {{out:weighted-kappa-by-hand:pairs two grades apart, guidelines prompt}} for `guidelines`. Whether either prompt agrees better is a question for [an interval](#prompt-sensitivity).

**Working rule:** For ordered grades, report linear weighted kappa next to plain kappa, and count the far misses separately. Weight each disagreement by the gap in gain it costs. With the book's linear gains, that's linear weights. With exponential gains of 0, 1 and 3 ([Choose a metric](#choose-a-ranking-metric)), the gaps are 1, 2 and 3, which neither of scikit-learn's options gives, so build that matrix as the example builds the linear one.

## Read kappa against the label mix {#kappa-depends-on-label-mix}

[Kappa's](#agreement-beyond-chance) correction depends on how often each judge uses each label, so the same judge gets a different kappa on a sample with a different mix of labels. When one label dominates, observed agreement can be high while kappa stays low. Artstein and Poesio call this the prevalence problem: "coders may agree on a high proportion of items while producing annotations that are indeed correct to a high degree, yet the reliability coefficients remain low" ([Artstein and Poesio, 2008](https://aclanthology.org/J08-4004/)).

[The recording](#judgments-from-people-and-llms) drew the train pool evenly, up to 4 pairs of each label per train query, so the pool holds more exact and irrelevant pairs than the store's judgments do. To see what that does to kappa, weight each pair so the pool has the label mix of all the train queries' judgments, and measure again:

[Example: kappa-under-two-label-mixes](../examples/kappa-under-two-label-mixes.py)

Partial makes up {{out:kappa-under-two-label-mixes:partial share, as drawn}} of the pool as drawn and {{out:kappa-under-two-label-mixes:partial share, in the train judgments}} of the train judgments, so the reweighting counts partial pairs more heavily and exact and irrelevant pairs less.

For `short`, kappa falls from {{out:kappa-under-two-label-mixes:kappa, short prompt, as drawn}} as drawn to {{out:kappa-under-two-label-mixes:kappa, short prompt, train judgments' mix}} in the judgments' mix. For `guidelines` it goes from {{out:kappa-under-two-label-mixes:kappa, guidelines prompt, as drawn}} to {{out:kappa-under-two-label-mixes:kappa, guidelines prompt, train judgments' mix}}.

The prompt with the higher kappa is `{{out:kappa-under-two-label-mixes:higher kappa, as drawn}}` as drawn, and `{{out:kappa-under-two-label-mixes:higher kappa, train judgments' mix}}` in the judgments' mix.

Published numbers carry their mix with them. The one study of LLM judges on WANDS we found drew 5,000 query-product pairs evenly across the three labels. It reports accuracy from 0.726 for the best open model, with 70 billion parameters, to 0.837 for the best closed one, each in its best setting, with 16 worked examples retrieved for every pair ([Sachdev et al., 2025, Table 3](https://arxiv.org/html/2502.15990)). The store's prompts have no worked examples. That accuracy is observed agreement on a balanced sample, as the train pool's is, so neither number transfers to the label mix of the store's judgments.

The pairs the store will ask an LLM to judge are the products its rankers put in their top 10s, the test pool. Of its judged pairs, {{out:kappa-under-two-label-mixes:exact share, judged test-pool pairs}} are exact, {{out:kappa-under-two-label-mixes:partial share, judged test-pool pairs}} partial and {{out:kappa-under-two-label-mixes:irrelevant share, judged test-pool pairs}} irrelevant. That is a third mix, and a different kind of pair, which [Read the direction of the LLM's errors](#read-llm-error-direction) measures.

**Working rule:** Report the label mix with every kappa, and never compare kappas from samples with different mixes, your own or published ones. We recommend drawing the agreement sample the way you'll use the judge: if it will grade your rankers' top 10s, sample pairs from top 10s on train queries.

## Read the direction of the LLM's errors {#read-llm-error-direction}

[Kappa](#agreement-beyond-chance) says how much two judges agree, not how they differ. A **confusion matrix** does: a table of counts with one row for each of the store's grades and one column for each of the LLM's, both in the order exact, partial, irrelevant. Above the diagonal, the LLM gave a lower grade than the store's judges; below it, a higher one. A judge whose errors fall mostly below is **lenient**, and mostly above, **strict**.

Several studies find LLM judges lenient. At TREC 2024's RAG track, which evaluates retrieval-augmented generation (answers written from retrieved text), human assessors appear stricter overall than UMBRELA, an open-source LLM judge ([Upadhyay et al., 2024a](https://arxiv.org/abs/2411.08275)). Arabzadeh and Clarke report that UMBRELA, built on GPT-4o ([Upadhyay et al., 2024b](https://arxiv.org/abs/2406.06519)), labelled over 26% more documents as perfectly relevant than people did across TREC's Deep Learning tracks of 2019 to 2021, 3,063 against 2,429 ([Arabzadeh and Clarke, 2025a](https://arxiv.org/abs/2504.12558)).

Yu et al. report that LLM judges consistently give inflated relevance scores ([Yu et al., 2026](https://arxiv.org/abs/2602.17170)). Not every study agrees. With one of their 32 prompts, on a sample stratified by label, Thomas et al. found the LLM more likely than TREC's assessors to call a pair not relevant, 44% against 33% ([Thomas et al., 2024](https://arxiv.org/abs/2309.10621)).

Build the matrix for both prompts on both [pools](#judgments-from-people-and-llms): the train pool, drawn evenly by label, and the judged pairs of the test pool, the top 10s of [chapter 1's four rankers](#search-end-to-end-decisions) that an LLM would grade to score them. The example prints one matrix per prompt and pool, and each shows which way the LLM errs. The test-pool matrices report on test queries; they don't choose a prompt:

[Example: llm-confusion-matrix](../examples/llm-confusion-matrix.py)

On the train pool, the LLM is stricter than the store's judges under both prompts: it grades lower than people on more pairs than it grades higher. With `short` it grades {{out:llm-confusion-matrix:llm grade lower than human, train pool, short prompt}} pairs lower and {{out:llm-confusion-matrix:llm grade higher than human, train pool, short prompt}} higher, and with `guidelines`, {{out:llm-confusion-matrix:llm grade lower than human, train pool, guidelines prompt}} lower and {{out:llm-confusion-matrix:llm grade higher than human, train pool, guidelines prompt}} higher.

The largest error on the train pool is {{out:llm-confusion-matrix:largest error, train pool, short prompt}} with `short` and {{out:llm-confusion-matrix:largest error, train pool, guidelines prompt}} with `guidelines`, and kappa is {{out:llm-confusion-matrix:kappa, train pool, short prompt}} and {{out:llm-confusion-matrix:kappa, train pool, guidelines prompt}}.

On the test pool the LLM isn't stricter under either prompt, grading higher than people on slightly more pairs. Its mean grade is {{out:llm-confusion-matrix:mean llm grade, test pool, short prompt}} with `short` and {{out:llm-confusion-matrix:mean llm grade, test pool, guidelines prompt}} with `guidelines`, against {{out:llm-confusion-matrix:mean human grade, test pool}} for the judges.

The largest errors on the test pool are {{out:llm-confusion-matrix:largest error, test pool, short prompt}} and {{out:llm-confusion-matrix:largest error, test pool, guidelines prompt}}, and kappa is {{out:llm-confusion-matrix:kappa, test pool, short prompt}} and {{out:llm-confusion-matrix:kappa, test pool, guidelines prompt}}. The direction, the largest error and kappa all change with the pool, as [the label mix](#kappa-depends-on-label-mix) warns. [Compare what the LLM saw](#what-the-llm-saw) finds one pattern that fits both pools.

Nor does the direction of the errors say what happens to a ranker's score. A lenient judge counts more products as relevant, which raises precision@10 ([Count the relevant products](#count-relevant-in-top-k)). nDCG@10 divides by an ideal top 10 built from the same grades ([Score one ranking](#score-one-ranking)), so raising grades can raise the ideal too, and the net effect can go either way and differ by ranker. [The check of the rankers' order](#system-order-human-versus-llm) measures it on the test pool.

**Working rule:** Read the confusion matrix before you read kappa, on pairs drawn the way you'll use the judge, and name the direction and the largest error for that pool. Don't carry a direction from one pool to another, and don't predict a ranker's nDCG@10 from it. Compare rankers with each other under the same grades, and never compare a score under LLM grades with a score under human grades.

## Compare what the LLM saw with what the judges saw {#what-the-llm-saw}

A judge can grade only the evidence it is shown. WANDS's guidelines told the judges to understand the query first: "Strongly recommend google it first" ([annotation guidelines](https://github.com/wayfair/WANDS/raw/3b74dcf4ba29ab8ff3e6a50b5b09fc627cb882b5/Product%20Search%20Relevance%20Annotation%20Guidelines.pdf)). The recording showed the LLM each product's name, class, category, description and attributes, as `lib.agreement.as_the_llm_saw` writes them out. The store's sample cuts descriptions to about 400 characters ([data card](../examples/data/wands/README.md)) and holds no images, so the recording sent the LLM text only. Some disagreement may come from this gap rather than from how well the LLM judges; that is our inference. One consequence can be tested: if the LLM grades what is written, it should give a lower grade to exact products whose text doesn't say what the query asks for, and a higher grade to partial products whose text does.

This check takes one train pair, then the test query `clear shower curtain` and every judged exact and partial pair in both pools ([the recording](#judgments-from-people-and-llms)). It looks for each query word in what the LLM saw, split into lowercase words the way [chapter 2's check](#relevance-is-a-judgment) splits them:

[Example: llm-grades-follow-written-words](../examples/llm-grades-follow-written-words.py)

For `{{out:llm-grades-follow-written-words:query}}`, a train query, the store's judges called `{{out:llm-grades-follow-written-words:Name}}` exact. Its grades from people, `short` and `guidelines`, in that order, are {{out:llm-grades-follow-written-words:grades from people, short, guidelines}}, and nothing the LLM saw says the stand holds two monitors.

For `clear shower curtain`, of the {{out:llm-grades-follow-written-words:clear shower curtain products that never say clear}} test-pool products that never say `clear` anywhere the LLM could read, the store's judges called {{out:llm-grades-follow-written-words:products that never say clear, graded exact by people}} exact, `short` {{out:llm-grades-follow-written-words:products that never say clear, graded exact by the short prompt}} and `guidelines` {{out:llm-grades-follow-written-words:products that never say clear, graded exact by the guidelines prompt}}.

Of the {{out:llm-grades-follow-written-words:clear shower curtain products that say clear}} that say it, the judges called {{out:llm-grades-follow-written-words:products that say clear, graded exact by people}} exact, `short` {{out:llm-grades-follow-written-words:products that say clear, graded exact by the short prompt}} and `guidelines` {{out:llm-grades-follow-written-words:products that say clear, graded exact by the guidelines prompt}}. For this query, the LLM's exact grades follow the word.

With `short`, exact pairs whose text misses a query word get a lower grade from the LLM at a share of {{out:llm-grades-follow-written-words:share graded lower, short prompt, exact train pairs missing a query word}} on the train pool and {{out:llm-grades-follow-written-words:share graded lower, short prompt, exact test pairs missing a query word}} on the test pool, against {{out:llm-grades-follow-written-words:share graded lower, short prompt, exact train pairs with every query word}} and {{out:llm-grades-follow-written-words:share graded lower, short prompt, exact test pairs with every query word}} for exact pairs whose text has every word.

On the test pool, partial pairs whose text has every query word get a higher grade at a share of {{out:llm-grades-follow-written-words:share graded higher, short prompt, partial test pairs with every query word}}, against {{out:llm-grades-follow-written-words:share graded higher, short prompt, partial test pairs missing a query word}} for those that miss a word.

On the test pool, `guidelines` splits the same way: {{out:llm-grades-follow-written-words:share graded lower, guidelines prompt, exact test pairs missing a query word}} against {{out:llm-grades-follow-written-words:share graded lower, guidelines prompt, exact test pairs with every query word}} for exact pairs, and {{out:llm-grades-follow-written-words:share graded higher, guidelines prompt, partial test pairs with every query word}} against {{out:llm-grades-follow-written-words:share graded higher, guidelines prompt, partial test pairs missing a query word}} for partial ones.

The pools differ in which case is common. The train pool holds {{out:llm-grades-follow-written-words:exact train pairs missing a query word}} exact pairs that miss a word and {{out:llm-grades-follow-written-words:partial train pairs with every query word}} partial pairs with every word; the test pool holds {{out:llm-grades-follow-written-words:exact test pairs missing a query word}} and {{out:llm-grades-follow-written-words:partial test pairs with every query word}}. By our reading, that is why [the error direction](#read-llm-error-direction) differs between the pools. The check is rough: a plural or a misspelling counts as a missing word too. The `guidelines` prompt asks for the demotions: "If a modifier isn't mentioned, decide from what is written, and don't assume it is met." The `short` prompt has no such rule, and demotes as well.

**Working rule:** Give an LLM judge every field your judges relied on that you can turn into text, such as attributes and the full description, and write down what it can't see. Before you count a disagreement against the LLM, check whether the evidence was in what it saw. When your judges' evidence is visual, measure whether the LLM grades exact products lower, as this one does on both pools, or give it the images if your model accepts them.

## Measure how much the prompt changes the labels {#prompt-sensitivity}

An LLM judge is a model and a prompt together. Across 42 paraphrases of one prompt, Thomas et al. found mean kappas spread from 0.50, which they call moderate agreement, up to 0.72. They warn that one prompt's result is a single sample from a wider range of potential performance ([Thomas et al., 2024](https://arxiv.org/abs/2309.10621)).

Across three LLMs and 72 prompts on TREC's Deep Learning collections, Arabzadeh and Clarke found that binary and pairwise methods were the least sensitive to the wording of the prompt, while graded relevance judgments, the store's kind, were highly sensitive to it ([Arabzadeh and Clarke, 2025b](https://arxiv.org/abs/2504.12408)). No single prompt was best for every LLM, though UMBRELA did consistently well on graded assessments.

The two prompts share their definitions, and `guidelines` adds rules on top ([Write judging guidelines](#write-judging-guidelines)). Count where they disagree on each pool, using [the recording](#judgments-from-people-and-llms), and then ask whether the train pool can tell which prompt agrees better with the store's judges. Pairs from the same query aren't independent, so the interval resamples whole train queries, as [chapter 3's bootstrap](#report-the-size-of-a-gap) resamples queries:

[Example: prompt-disagreement](../examples/prompt-disagreement.py)

A second prompt is a second judge. The two prompts disagree on {{out:prompt-disagreement:train pairs where the prompts disagree}} train pairs and on {{out:prompt-disagreement:test pairs where the prompts disagree}} test pairs, and their kappa with each other is {{out:prompt-disagreement:kappa between the prompts, train pool}} on the train pool and {{out:prompt-disagreement:kappa between the prompts, test pool}} on the test pool.

The largest disagreement differs by pool. On the train pool it is {{out:prompt-disagreement:largest disagreement, train pool}}; on the test pool, the rankers' top 10s, it is {{out:prompt-disagreement:largest disagreement, test pool}}.

Of the train pairs that `short` calls partial and `guidelines` calls irrelevant, the store's judges called {{out:prompt-disagreement:moved train pairs judged irrelevant by people}} irrelevant, siding with `guidelines`, {{out:prompt-disagreement:moved train pairs judged partial by people}} partial, siding with `short`, and {{out:prompt-disagreement:moved train pairs judged exact by people}} exact. Among the moved pairs the example lists are a sofa bed for `toddler couch fold out` and window curtains for `70s inspired furniture`. The recording keeps labels, not reasons, so from names like these we infer that the move comes from the `guidelines` rule that calls the wrong kind of product irrelevant.

On kappa with the store's judges, `guidelines` leads `short` by {{out:prompt-disagreement:kappa with people, guidelines minus short}}, and the 95% interval over resampled train queries runs from {{out:prompt-disagreement:95% interval over resampled train queries}}. That interval includes 0, so the train pool can't tell the prompts apart, and the store keeps both.

**Working rule:** Treat the prompt as part of the judge. Record every prompt word for word with the model version, the settings and the answer schema, as this recording does in its manifest and script. Treat a change of prompt as a different judge whose agreement you measure again. Run at least two prompts, and trust a conclusion from LLM grades only when both give it. Choose one prompt over another on train pairs, and only when the interval for their difference excludes 0.

## Fill unjudged pairs with LLM grades {#fill-unjudged-with-llm-grades}

We consider filling the holes in the human judgments the least risky use of an LLM judge. Keep every human grade, and use the LLM only for pairs nobody judged, which a metric would otherwise count as grade 0 ([Count unjudged products](#unjudged-results)). The LLM then decides only what was a guess before.

The test pool holds every product in the top 10 of [chapter 1's four rankers](#search-end-to-end-decisions) for each test query. Of its {{out:fill-unjudged-pairs:pairs in the test pool}} pairs, {{out:fill-unjudged-pairs:of them, unjudged}} are unjudged, and BM25's top 10s hold {{out:fill-unjudged-pairs:unjudged products in the top 10s, bm25}} of them.

Semantic retrieval's top 10s hold {{out:fill-unjudged-pairs:unjudged products in the top 10s, semantic}} unjudged products, and the hybrid's hold {{out:fill-unjudged-pairs:unjudged products in the top 10s, hybrid}}.

The example fills them under each [prompt](#write-judging-guidelines) and rescores the four rankers on nDCG@10 ([Score one ranking](#score-one-ranking)). It then reruns [chapter 3's planned comparisons](#when-a-difference-is-established) on the filled grades:

[Example: fill-unjudged-pairs](../examples/fill-unjudged-pairs.py)

The prompts disagree about the unjudged pairs. Counted as exact, partial and irrelevant, `short` grades them {{out:fill-unjudged-pairs:unjudged pairs graded exact, partial, irrelevant, short prompt}}, and `guidelines` {{out:fill-unjudged-pairs:unjudged pairs graded exact, partial, irrelevant, guidelines prompt}}. They part on the unjudged products for `outdoor light fixtures` in BM25's top 10 ([Score every test query](#score-every-test-query)): of those {{out:fill-unjudged-pairs:unjudged products for outdoor light fixtures}}, {{out:fill-unjudged-pairs:of them, with bulb in the name}} have `bulb` in their names.

`guidelines` calls {{out:fill-unjudged-pairs:unjudged bulbs graded irrelevant, guidelines prompt}} of the bulbs irrelevant, and `short` calls {{out:fill-unjudged-pairs:unjudged bulbs graded irrelevant, short prompt}}. Chapter 1 argued that a bulb isn't a light fixture. Which reading the store's judges would take, nobody knows until they judge the bulbs.

With the holes filled, BM25 scores {{out:fill-unjudged-pairs:bm25 ndcg@10, filled by the short prompt}} with `short` and {{out:fill-unjudged-pairs:bm25 ndcg@10, filled by the guidelines prompt}} with `guidelines`, against {{out:kendall-tau-four-rankers:bm25 ndcg@10 under people}} on human grades alone.

Semantic retrieval scores {{out:fill-unjudged-pairs:semantic ndcg@10, filled by the short prompt}} and {{out:fill-unjudged-pairs:semantic ndcg@10, filled by the guidelines prompt}}, against {{out:kendall-tau-four-rankers:semantic ndcg@10 under people}}.

The order of the four rankers on human grades alone is `{{out:fill-unjudged-pairs:order, people's grades only}}`, and filling keeps that order under both prompts. The rankers that change place when `short` fills the holes are {{out:fill-unjudged-pairs:rankers that change place when filled, short prompt}}, and when `guidelines` does, {{out:fill-unjudged-pairs:rankers that change place when filled, guidelines prompt}}.

The verdicts of the planned comparisons that change are {{out:fill-unjudged-pairs:verdicts that change when filled, short prompt}} with `short` and {{out:fill-unjudged-pairs:verdicts that change when filled, guidelines prompt}} with `guidelines`.

Under human grades, the rankers' mean judged@10, the share of each top 10 that people judged, runs from {{out:compare-chapter-one-rankers:bm25 judged@10}} for BM25 to {{out:compare-chapter-one-rankers:semantic judged@10}} for semantic retrieval ([Compare runs in a harness](#compare-runs-in-a-harness)). Filled, every product in a top 10 has a grade.

How far can the filled grades be trusted? The closest evidence is the judged pairs of the same pool, where the LLM's kappa with the store's judges is {{out:llm-confusion-matrix:kappa, test pool, short prompt}} for `short` and {{out:llm-confusion-matrix:kappa, test pool, guidelines prompt}} for `guidelines`. On those judged pairs the LLM grades higher than people on slightly more pairs than lower, under both prompts ([Read the direction of the LLM's errors](#read-llm-error-direction)).

**Working rule:** Fill the unjudged pairs in your rankers' top 10s with LLM grades as a diagnostic, under every prompt you run, and report the filled score beside the human-only one with the number of pairs filled. If filling changes the order of two rankers or a verdict, have people judge those pairs. Never let a filled grade replace a human one.

## Check whether LLM grades order the rankers the same way {#system-order-human-versus-llm}

Two sets of judgments can disagree on many labels and still rank systems alike. Voorhees scored TREC runs with different sets of relevance judgments and found that "Very high correlations were found among the rankings of systems produced using different relevance judgment sets" ([Voorhees, 1998](https://people.eng.unimelb.edu.au/ammoffat/sigir98/abstracts/voorhees.html)). LLM judges show the same split. Across 42 sets of LLM labels for TREC's 2023 Deep Learning track, kappa ran from about 0.06 to 0.29 while the correlation of system rankings stayed between about 0.83 and 0.95 ([Rahmani et al., 2025, Table 3](https://arxiv.org/abs/2502.13908)).

That correlation is **Kendall's tau**. For every pair of systems, ask whether two orderings put them the same way round: **concordant** if they do, **discordant** if not. Tau is the difference between the two counts divided by the number of pairs, from 1 for the same order to −1 for the reverse. SciPy's `kendalltau` computes τ_b, a form that also handles ties ([SciPy documentation](https://docs.scipy.org/doc/scipy-1.18.0/reference/generated/scipy.stats.kendalltau.html)).

Yilmaz et al. note that rankings with a tau of 0.9 or more are often treated as effectively equivalent, at least empirically ([Yilmaz et al., 2008](https://www.ccs.neu.edu/home/jaa/tmp/RankCorrelation.pdf)). Sanderson and Soboroff found that basing decisions on such thresholds is less reliable than has been assumed ([Sanderson and Soboroff, 2007](https://eprints.whiterose.ac.uk/id/eprint/4589/)).

Score [chapter 1's four rankers](#search-end-to-end-decisions) on nDCG@10 under human grades and under each [prompt's](#write-judging-guidelines) LLM grades. The LLM judged only the test pool, the products in those rankers' top 10s, so a control also scores them with human grades for the pool's pairs only. With four rankers there are {{out:kendall-tau-four-rankers:pairs of rankers}} pairs, so one swapped pair moves tau by {{out:kendall-tau-four-rankers:change in tau from one swapped pair}}, and tau can take only seven values. That is crude. Make a guess first: will LLM grades put the four rankers in the same order that human grades do?

[Example: kendall-tau-four-rankers](../examples/kendall-tau-four-rankers.py)

Human grades give `{{out:kendall-tau-four-rankers:order under people}}`. The control, people's grades on the LLM's pairs, keeps that order, with a tau of {{out:kendall-tau-four-rankers:tau by hand, people against people on the LLM's pairs}}, so limiting the grades to the pool moves no ranker. It does raise every score, BM25's from {{out:kendall-tau-four-rankers:bm25 ndcg@10 under people}} to {{out:kendall-tau-four-rankers:bm25 ndcg@10 under people on the LLM's pairs}}, because the ideal top 10 can only hold pooled products.

The order under `short` is `{{out:kendall-tau-four-rankers:order under the short prompt}}`, and under `guidelines` it is `{{out:kendall-tau-four-rankers:order under the guidelines prompt}}`. Both prompts swap the same two pairs: `{{out:kendall-tau-four-rankers:swapped under the short prompt}}` under `short`, and `{{out:kendall-tau-four-rankers:swapped under the guidelines prompt}}` under `guidelines`.

Tau for `short` is {{out:kendall-tau-four-rankers:tau by hand, people against the short prompt}} by hand and {{out:kendall-tau-four-rankers:tau from scipy, people against the short prompt}} from SciPy, and for `guidelines`, {{out:kendall-tau-four-rankers:tau from scipy, people against the guidelines prompt}}.

Against the control, LLM grades don't move the four rankers by the same amount, or in the same direction. With `short`, BM25 changes by {{out:kendall-tau-four-rankers:bm25, the short prompt minus people on the LLM's pairs}}, semantic retrieval by {{out:kendall-tau-four-rankers:semantic, the short prompt minus people on the LLM's pairs}}, the hybrid by {{out:kendall-tau-four-rankers:hybrid, the short prompt minus people on the LLM's pairs}} and the reranked hybrid by {{out:kendall-tau-four-rankers:reranked, the short prompt minus people on the LLM's pairs}}.

With `guidelines`, the changes are {{out:kendall-tau-four-rankers:bm25, the guidelines prompt minus people on the LLM's pairs}}, {{out:kendall-tau-four-rankers:semantic, the guidelines prompt minus people on the LLM's pairs}}, {{out:kendall-tau-four-rankers:hybrid, the guidelines prompt minus people on the LLM's pairs}} and {{out:kendall-tau-four-rankers:reranked, the guidelines prompt minus people on the LLM's pairs}}. The judge raises some rankers and lowers others, and that moves them past each other.

Under `short`, semantic retrieval, the hybrid and the reranked hybrid nearly tie, at {{out:kendall-tau-four-rankers:semantic ndcg@10 under the short prompt}}, {{out:kendall-tau-four-rankers:hybrid ndcg@10 under the short prompt}} and {{out:kendall-tau-four-rankers:reranked ndcg@10 under the short prompt}}. Tau ignores how far apart two rankers score, so a swap between near-ties counts as much as one between the best ranker and the worst.

Tau also weighs a swap at the top the same as one at the bottom. Yilmaz et al. say that Kendall's tau penalizes errors at any part of a list equally ([Yilmaz et al., 2008](https://www.ccs.neu.edu/home/jaa/tmp/RankCorrelation.pdf)). In another comparison of TREC runs, they note that the systems performed so similarly that the tau statistic itself may be affected by them ([Yilmaz et al., 2012](https://static.aminer.org/pdf/20170130/pdfs/sigir/jcmbhoqtt46bhqr5xegf9zy7snars82a.pdf)).

These rankers keep their place under every set of grades: `{{out:kendall-tau-four-rankers:rankers in the same place under every set of grades}}`. If you predicted the same order, notice where the orders differ. Among the three rankers that human grades score {{out:kendall-tau-four-rankers:semantic ndcg@10 under people}}, {{out:kendall-tau-four-rankers:hybrid ndcg@10 under people}} and {{out:kendall-tau-four-rankers:reranked ndcg@10 under people}}, the gaps are ones that no planned comparison establishes ([Rerun the planned comparisons](#llm-graded-verdicts)).

**Working rule:** Report tau with the list of swapped pairs, never alone. With a handful of rankers, where one swap moves tau by {{out:kendall-tau-four-rankers:change in tau from one swapped pair}}, don't read it against 0.9. Then test whether the swapped pairs differ at all, with [the planned comparisons under each set of grades](#llm-graded-verdicts).

## Rerun the planned comparisons with LLM grades {#llm-graded-verdicts}

Decisions come from verdicts, not from the [orderings that tau compares](#system-order-human-versus-llm). A difference is established when its paired [permutation test](#permutation-test-on-queries) p-value falls below the chapter's Bonferroni threshold, with the t-test and a bootstrap interval beside it ([Decide in advance what counts as established](#when-a-difference-is-established)). Would LLM grades have given the store the same verdicts? Expect trouble near the top. Otero et al. found that LLM-based judgements are unfair at ranking the top-performing systems, with a very high rate of false positives on statistical differences ([Otero et al., 2025](https://arxiv.org/abs/2411.13212)).

The planned comparisons are chapter 3's three, at a p-value threshold of {{out:planned-comparisons-llm-grades:p-value threshold}}, with intervals at the {{out:planned-comparisons-llm-grades:interval level}} level. Run each under human grades and under each [prompt's](#write-judging-guidelines) grades on the test pool, and add chapter 3's exploratory comparison of the reranked hybrid with semantic retrieval:

[Example: planned-comparisons-llm-grades](../examples/planned-comparisons-llm-grades.py)

| Measure | Grades | Semantic minus BM25 | Hybrid minus semantic | Reranked minus hybrid |
| --- | --- | --- | --- | --- |
| Mean difference | Human | {{out:planned-comparisons-llm-grades:semantic minus bm25, people}} | {{out:planned-comparisons-llm-grades:hybrid minus semantic, people}} | {{out:planned-comparisons-llm-grades:reranked minus hybrid, people}} |
| Mean difference | `short` | {{out:planned-comparisons-llm-grades:semantic minus bm25, short prompt}} | {{out:planned-comparisons-llm-grades:hybrid minus semantic, short prompt}} | {{out:planned-comparisons-llm-grades:reranked minus hybrid, short prompt}} |
| Mean difference | `guidelines` | {{out:planned-comparisons-llm-grades:semantic minus bm25, guidelines prompt}} | {{out:planned-comparisons-llm-grades:hybrid minus semantic, guidelines prompt}} | {{out:planned-comparisons-llm-grades:reranked minus hybrid, guidelines prompt}} |
| Interval | Human | {{out:planned-comparisons-llm-grades:semantic minus bm25, people, interval}} | {{out:planned-comparisons-llm-grades:hybrid minus semantic, people, interval}} | {{out:planned-comparisons-llm-grades:reranked minus hybrid, people, interval}} |
| Interval | `short` | {{out:planned-comparisons-llm-grades:semantic minus bm25, short prompt, interval}} | {{out:planned-comparisons-llm-grades:hybrid minus semantic, short prompt, interval}} | {{out:planned-comparisons-llm-grades:reranked minus hybrid, short prompt, interval}} |
| Interval | `guidelines` | {{out:planned-comparisons-llm-grades:semantic minus bm25, guidelines prompt, interval}} | {{out:planned-comparisons-llm-grades:hybrid minus semantic, guidelines prompt, interval}} | {{out:planned-comparisons-llm-grades:reranked minus hybrid, guidelines prompt, interval}} |
| Permutation p | Human | {{out:planned-comparisons-llm-grades:semantic minus bm25, people, permutation p}} | {{out:planned-comparisons-llm-grades:hybrid minus semantic, people, permutation p}} | {{out:planned-comparisons-llm-grades:reranked minus hybrid, people, permutation p}} |
| Permutation p | `short` | {{out:planned-comparisons-llm-grades:semantic minus bm25, short prompt, permutation p}} | {{out:planned-comparisons-llm-grades:hybrid minus semantic, short prompt, permutation p}} | {{out:planned-comparisons-llm-grades:reranked minus hybrid, short prompt, permutation p}} |
| Permutation p | `guidelines` | {{out:planned-comparisons-llm-grades:semantic minus bm25, guidelines prompt, permutation p}} | {{out:planned-comparisons-llm-grades:hybrid minus semantic, guidelines prompt, permutation p}} | {{out:planned-comparisons-llm-grades:reranked minus hybrid, guidelines prompt, permutation p}} |
| t-test p | Human | {{out:planned-comparisons-llm-grades:semantic minus bm25, people, t-test p}} | {{out:planned-comparisons-llm-grades:hybrid minus semantic, people, t-test p}} | {{out:planned-comparisons-llm-grades:reranked minus hybrid, people, t-test p}} |
| t-test p | `short` | {{out:planned-comparisons-llm-grades:semantic minus bm25, short prompt, t-test p}} | {{out:planned-comparisons-llm-grades:hybrid minus semantic, short prompt, t-test p}} | {{out:planned-comparisons-llm-grades:reranked minus hybrid, short prompt, t-test p}} |
| t-test p | `guidelines` | {{out:planned-comparisons-llm-grades:semantic minus bm25, guidelines prompt, t-test p}} | {{out:planned-comparisons-llm-grades:hybrid minus semantic, guidelines prompt, t-test p}} | {{out:planned-comparisons-llm-grades:reranked minus hybrid, guidelines prompt, t-test p}} |
| Established | Human | {{out:planned-comparisons-llm-grades:semantic minus bm25, people, established}} | {{out:planned-comparisons-llm-grades:hybrid minus semantic, people, established}} | {{out:planned-comparisons-llm-grades:reranked minus hybrid, people, established}} |
| Established | `short` | {{out:planned-comparisons-llm-grades:semantic minus bm25, short prompt, established}} | {{out:planned-comparisons-llm-grades:hybrid minus semantic, short prompt, established}} | {{out:planned-comparisons-llm-grades:reranked minus hybrid, short prompt, established}} |
| Established | `guidelines` | {{out:planned-comparisons-llm-grades:semantic minus bm25, guidelines prompt, established}} | {{out:planned-comparisons-llm-grades:hybrid minus semantic, guidelines prompt, established}} | {{out:planned-comparisons-llm-grades:reranked minus hybrid, guidelines prompt, established}} |

Under `short`, the verdicts that differ from the human ones are {{out:planned-comparisons-llm-grades:verdicts that differ from people's, short prompt}}. Under `guidelines`, the verdict that differs is {{out:planned-comparisons-llm-grades:verdicts that differ from people's, guidelines prompt}}, the one result chapter 3 established.

The planned comparisons where the t-test or the interval disagrees with the permutation test number {{out:planned-comparisons-llm-grades:planned comparisons where the t-test or interval disagrees, people}} under human grades, {{out:planned-comparisons-llm-grades:planned comparisons where the t-test or interval disagrees, short prompt}} under `short` and {{out:planned-comparisons-llm-grades:planned comparisons where the t-test or interval disagrees, guidelines prompt}} under `guidelines`. For semantic retrieval minus BM25 under `guidelines`, the check that disagrees is the {{out:planned-comparisons-llm-grades:semantic minus bm25, guidelines prompt, disagrees with the permutation test}}, so whether the store's one established result survives LLM grades depends on the prompt.

[Comparing the rankers' order](#system-order-human-versus-llm) found two swapped pairs. One, the reranked hybrid against the hybrid, is a planned comparison, and the table shows it established under no set of grades. The other, the reranked hybrid against semantic retrieval, is the exploratory comparison. Under human grades, the reranked hybrid minus semantic retrieval is {{out:planned-comparisons-llm-grades:reranked minus semantic, people}} (permutation p {{out:planned-comparisons-llm-grades:reranked minus semantic, people, permutation p}}). Under LLM grades it is {{out:planned-comparisons-llm-grades:reranked minus semantic, short prompt}} with `short` and {{out:planned-comparisons-llm-grades:reranked minus semantic, guidelines prompt}} with `guidelines`. Chapter 3 chose it after seeing the test means, so it establishes nothing either way.

LLM judges can also favour rankers that use LLMs: Balog et al. report "the first empirical evidence of LLM judges exhibiting significant bias towards LLM-based rankers" ([Balog et al., 2025](https://arxiv.org/abs/2503.19092)). Chapter 1's rankers use no LLM, so this chapter can't measure that bias. The book scores the LLM-based rankers of [chapter 8](#rewrite-queries-with-llm) and [chapter 9](#llm-pointwise-reranking) on human grades.

**Working rule:** Never ship a change on a verdict from LLM grades alone. Use LLM grades to decide which candidates are worth judging, and decide on human grades. When two prompts give different verdicts, the LLM can't settle the comparison, and people must.

## Correct LLM scores with a human sample {#calibrate-on-a-human-sample}

A middle path uses both kinds of grades: grade every query with the LLM, have people judge a random sample of queries, and use the sample to correct the LLM's error. **Prediction-powered inference** makes this rigorous, with confidence intervals that hold "without making any assumptions on the machine-learning algorithm that supplies the predictions" ([Angelopoulos et al., 2023](https://arxiv.org/abs/2301.09633)). A simple form of the idea, for a mean, takes the LLM's mean over every query and adds the human sample's average of human minus LLM:

```text
corrected = mean over all queries of the LLM score + mean over the human sample of (human score − LLM score)
```

In search evaluation, Oosterhuis et al. build metric intervals this way, warning that "generated relevance annotations are not immune to (systematic) errors" ([Oosterhuis et al., 2024](https://arxiv.org/abs/2407.02464)). Takehi et al. let the LLM pick which pairs people judge under a budget, then calibrate it on them ([Takehi et al., 2025](https://arxiv.org/abs/2411.06877)).

Suppose the store could afford human judgments for {{out:correct-llm-scores-with-humans:queries judged by people in each draw}} test queries. The simulation below draws {{out:correct-llm-scores-with-humans:draws}} random sets of that size and, for each, corrects the LLM's estimate, under each [prompt](#write-judging-guidelines), of two gaps between [chapter 1's rankers](#search-end-to-end-decisions): semantic retrieval minus BM25, and the hybrid minus semantic retrieval. **Spread** is an estimate's standard deviation over the draws. The simulation chooses nothing:

[Example: correct-llm-scores-with-humans](../examples/correct-llm-scores-with-humans.py)

For semantic retrieval minus BM25, the LLM alone estimates {{out:correct-llm-scores-with-humans:semantic minus bm25, short prompt, LLM only}} with `short`, against {{out:correct-llm-scores-with-humans:semantic minus bm25, people, all queries}} on human grades. Corrected, the estimate averages {{out:correct-llm-scores-with-humans:semantic minus bm25, short prompt, corrected, mean over draws}} over the draws, so the correction removes the LLM's bias.

Its spread is {{out:correct-llm-scores-with-humans:semantic minus bm25, short prompt, corrected, spread}}, against {{out:correct-llm-scores-with-humans:semantic minus bm25, human sample alone, spread}} for the human sample alone: the LLM's grades for the other queries add information, because its per-query gaps correlate with the human ones at {{out:correct-llm-scores-with-humans:semantic minus bm25, short prompt, correlation of per-query gaps with people's}}.

For the hybrid minus semantic retrieval, the corrected spread is {{out:correct-llm-scores-with-humans:hybrid minus semantic, short prompt, corrected, spread}}, against {{out:correct-llm-scores-with-humans:hybrid minus semantic, human sample alone, spread}} for the human sample alone, with a correlation of {{out:correct-llm-scores-with-humans:hybrid minus semantic, short prompt, correlation of per-query gaps with people's}}. Is the corrected spread smaller than the human sample's? For this gap, no, under either prompt. For semantic retrieval minus BM25 it is smaller under both prompts.

The correction fixes the level of a score, but narrows the uncertainty only where the LLM's per-query scores track people's, and here they track them least, on the closer of the two gaps.

**Working rule:** Whenever you report a score from LLM grades, have people judge a random sample of queries and report the corrected estimate. On that sample, check how well the LLM's per-query gaps correlate with the human ones. When they correlate weakly, spend the budget on judging more queries.

## What goes wrong in production {#judgments-from-people-and-llms-in-production}

**The judge changes without notice.** A store that judges with an LLM runs it again and again, so a different model version, quantization (the precision at which the model's weights are stored) or reworded prompt is a different judge. The recording's `manifest.json` pins the model's digest and both prompts word for word. The answer schema and the context length live in `recordings/llm.py` and `recordings/llm_judgments.py`, so record those too. Rerun the agreement check on the same human sample whenever any of them changes.

**Systems tuned to the judge score well on it.** Clarke and Dietz built a run that exploits an LLM judge, and it ranked "significantly higher under automatic assessment (5th) than under manual assessment (28th)" ([Clarke and Dietz, 2024](https://arxiv.org/abs/2412.17156)). A ranker tuned on LLM grades, or built on the same LLM, needs its headline score on human grades.

**Product text can steer the judge.** Alaofi et al. found that LLM judges can be misled by query words planted in an irrelevant text, and by instructions written into it ([Alaofi et al., 2024](https://arxiv.org/abs/2501.17969)). Product descriptions may come from outside your team. Keep a human sample in every evaluation, so that a judge led astray shows up as a falling agreement.

**Calibrating on test is tuning on test.** Rewording a prompt until it matches the human grades on test queries makes agreement there meaningless. Choose prompts on train queries, and report agreement on test.

**Human grades aren't the truth either.** On the TREC Podcast track, where an LLM and a TREC assessor disagreed most, Mansour et al. found that experts re-judging the pairs more often sided with the LLM ([Mansour et al., 2026](https://arxiv.org/abs/2601.05603)). Treat human grades as the reference you chose, and read disagreements before you blame either side.

**Working rule:** Pin the judge, keep a human sample in every evaluation, rerun agreement when anything changes, and never score a ranker with the judge you tuned it on.

## Decisions this chapter made {#judgments-from-people-and-llms-decisions}

Each choice below is a default. Later chapters keep it unless they state a reason.

- **Human grades for every headline score.** LLM grades are a diagnostic: filling holes, screening candidates and checking verdicts. The LLM-based rankers of [chapter 8](#rewrite-queries-with-llm) and [chapter 9](#llm-pointwise-reranking) are scored on human grades.
- **Every finding about the judge names its pool.** Kappa, weighted kappa, the far misses and the label mix are measured on train pairs people judged, and reported for the test pool's judged pairs too. The direction and the largest error are read on each pool, and only the rankers' top 10s speak to ranker scores.
- **Two prompts, recorded word for word.** The train pool can't tell them apart, so the store keeps both and trusts a conclusion only when both prompts give it.
- **Unjudged pairs filled with LLM grades as a diagnostic,** reported beside the human-only score. On the test queries, filling changes no ranker's place and no verdict.
- **Compare scores only under the same grades.** On the same pool, LLM grades don't move every ranker's nDCG@10 alike, so a score under LLM grades never meets one under human grades.
- **Tau with its swapped pairs, then the planned comparisons at chapter 3's threshold.** With four rankers tau is crude, and the verdicts decide.
- **A score from LLM grades comes with a human sample** of queries that corrects it. The sample shows whether the correction narrows the uncertainty: it did for semantic retrieval against BM25, and not for the hybrid against semantic retrieval.
