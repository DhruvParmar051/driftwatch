# driftwatch: the whole project, explained in depth

**Did the model change, or did the users? Label-free monitoring of a deployed language model.**

Dhruv Parmar · Pronnati Tapaswi · Mahak Khurdia · Aditya Jana
Dhirubhai Ambani University, Gandhinagar
28 October 2026

---

## Contents

1. The project in one page
2. The problem, from the beginning
3. The core difficulty: users change too
4. The idea that solves it: re-ask the old model
5. The four monitors, piece by piece
6. The served-answer test and its proof
7. How we set thresholds without fooling ourselves
8. How we kept the whole study honest
9. The data, the models and the changes we tested
10. Experiment by experiment: what we did, what came out, what it means
11. What it all means
12. What it cannot do
13. How the project unfolded
14. How to run it, and what it cost
15. Presentation script for sir
16. Questions sir may ask, with answers
17. Glossary

---

# 1. The project in one page

**The problem.** Many products now have a language model answering users behind the scenes. Often the model is rented from a provider through an API. The provider can replace it with a new version under the same name, or the team can switch to a cheaper model, a lower-precision build, an edited prompt or a different sampling setting. When that happens nothing crashes: response times, error rates and the fluency of the text all look normal. Yet the answers have changed, sometimes for the worse. Checking quality normally needs the right answers, and nobody labels live traffic.

**The question.** Can a program notice that the model changed, using only the questions and answers it sees, with no labels, at a false-alarm rate we fix in advance?

**The difficulty.** Users change too. When the mix of questions shifts (more maths one week, more trivia the next), every statistic of the answers moves even though the model is exactly the same. Our first monitor, which compared live answers with a fixed healthy reference, alarmed in 98–100% of streams where only the topics changed.

**The idea.** Keep the old model available and ask it the same questions users are asking now. Then compare answer with answer, question by question. A change of topic affects both sides equally and cancels out; a change of model does not.

**Our test.** The served-answer test does this with only the single answer each user actually received. We proved that its expected score is exactly zero for every question when nothing changed, which is the formal reason topic shifts cannot move it.

**What we found.**

- It catches our planted faults in 99–100% of streams, DeepSeek's real update in 95%, the same update on real chatbot traffic in 94%, and Qwen's real update in 96.5%.
- It raised no false alarm in 87,716 healthy windows and alarmed on 0–1% of harmless topic shifts.
- Hosted models drift a little from day to day, so calibration has to use several days of healthy answers.
- An alarm means the model changed, not that it got worse: two of the changes we caught made the model better. A small graded check after each alarm told better from worse with no wrong verdict in 1,833 alarms.
- In a blind test on five changes it had never seen, run frozen, it caught four in 98–100% of streams with no false alarm, and missed the fifth, answers cut at 256 tokens, which lies beyond the 1,000 characters it compares.

**How we kept it honest.** Every study was pre-registered in a committed file before its data existed. Every threshold was frozen before any fault answer was read. Every later change is a dated amendment. Every study was replayed five times. Total paid spend: $26.11.

---

# 2. The problem, from the beginning

## 2.1 What "deployed" means here

A deployed language model is one that real users talk to through an application. Think of a support chatbot, a homework helper, a tool that turns emails into structured data, or a search assistant. The application sends each user's question to the model and shows the user the answer. Thousands of questions a day may pass through it.

The model itself is often not owned by the company. It is rented from a provider (OpenAI, DeepSeek, Alibaba's Qwen, and so on) through an API, or run on the company's own servers from downloaded weights.

## 2.2 How a model changes without anyone noticing

There are many ordinary ways the model behind an application changes:

- **A silent provider update.** The provider ships a new version under the same name. This happened during our project: on 17 October 2026 DeepSeek replaced V4 Flash with V4.1 Flash behind the same API name, without announcing it.
- **A cheaper model.** To save money, someone switches to a smaller model.
- **Lower precision.** A model can be stored with fewer bits per number (quantisation). It is cheaper to run and usually almost as good, but not identical.
- **An edited prompt.** The system prompt (the hidden instructions the model always receives) gets a small edit, perhaps by someone who did not realise its effect.
- **A changed setting.** A configuration change drops a sampling parameter such as temperature, which controls how varied the answers are.

## 2.3 Why the usual monitoring misses it

Production systems are monitored for latency, error rates, uptime and cost. All of these stay normal when the model changes, because the model still answers, quickly and fluently. The thing that changed is the *content* of the answers.

The obvious way to check content is to grade it: compare each answer with the correct answer. But live traffic has no correct answers attached. Nobody labels thousands of user questions a day. A common substitute is to re-run a fixed, labelled benchmark on a schedule. That helps, but it is slow, it costs grading, and it measures a fixed question set that may look nothing like what users actually ask.

## 2.4 The precise question

So we set ourselves a precise goal:

> Build a monitor that sees only the questions and answers of live traffic, never a label, and alarms soon after the model changes, while keeping false alarms at or below a rate fixed in advance: one per thousand windows of traffic.

A **window** is a block of 200 consecutive questions. A new window closes every 50 questions, so neighbouring windows overlap. The monitor makes one decision per window: alarm or no alarm.

---

# 3. The core difficulty: users change too

## 3.1 The obvious design

The obvious design is a statistical alarm. Before deployment, record what healthy answers look like: how long they are, how consistent the model is when asked the same thing several times, how varied the wording is, what the answers are about. Then, for every live window, measure the same things and alarm if they have moved too far from normal.

We built exactly this (we call it the **pooled monitor**, v1). It worked in one sense: it caught every planted fault.

## 3.2 The flaw

It also alarmed when nothing about the model had changed. We tested it on **topic shifts**: the same healthy answers, replayed under a different mix of questions. In one shift, more of the window came from long-form tasks (explanations, summaries); in another, more came from short-form tasks (trivia, maths answers). The model was identical. The monitor alarmed in 98% of streams for the first shift and 100% for the second.

The reason is simple once you see it. The monitor's statistics mostly measure *what kinds of questions a window contains*. A week of maths homework produces short answers; a week of "explain like I'm five" produces long ones. Longer answers could mean a different model, or just different questions. A monitor that compares with a fixed reference has no way to tell.

An alarm that rings every time the users change topic is useless. An operator would switch it off within a week.

## 3.3 A partial fix that needs labels

Our next attempt (the **per-family monitor**, v3) computed each statistic within one task type at a time: maths answers compared only with healthy maths answers, trivia with trivia. More maths in a window then no longer looks like shorter answers. It cut topic-shift alarms to 23% on the harder shift, but it still fired on about one shift in four, and it needs to know the task type of every question. Real user traffic does not come labelled with task types.

So the real job became clear: **separate a change of model from a change of users**, without labels.

---

# 4. The idea that solves it: re-ask the old model

## 4.1 An everyday analogy

Suppose a restaurant hires a new cook and you want to know whether the food changed. Comparing today's dishes with last month's is unfair: today people ordered different dishes. The fair test is to ask the old cook to make the same dishes people ordered today, and compare plate by plate.

## 4.2 The same idea for a model

Keep the old model available: a saved copy of the weights, or a pinned, dated API version. For every question a user asks, also send the same question to the old model (the **reference**). Then compare the live answer with the reference's answers to the same question.

- If only the topics changed, both sides see the same new topics, both change in the same way, and the comparison stays balanced.
- If the model changed, the live answers stop looking like the reference's answers to the same questions, and the comparison moves.

This turns the impossible question ("is today's traffic different because of the model or the users?") into a fair one ("for this very question, does the live answer look like what the old model would say?").

---

# 5. The four monitors, piece by piece

We studied four monitors that differ in one thing: what a window of live answers is compared with.

| Monitor | Compares a window with | Needs task labels | Re-asks a reference | Production answers per question |
|---|---|---|---|---|
| Pooled (v1) | a fixed healthy reference | no | no | 4 |
| Per-family (v3) | a fixed reference, one task type at a time | yes | no | 4 |
| Paired | the reference re-asked the same questions | no | yes | 4 |
| Served answer | the reference re-asked the same questions | no | yes | 1 |

## 5.1 Pooled (v1)

Five statistics, each comparing the window with healthy answers. None of them needs a label.

- **Answer length.** A Kolmogorov–Smirnov statistic: the largest gap between the distribution of answer lengths in the window and in healthy answers. Smaller or edited models often write shorter or longer answers.
- **Agreement among samples.** Each question is answered four times with different random seeds. A healthy model tends to give similar answers to the same question; a confused or weaker model disagrees with itself more.
- **Lexical diversity.** How varied the vocabulary is, measured with a length-normalised type–token ratio (MATTR), so it does not simply restate length.
- **Label entropy.** On the intent-classification task, how spread out the predicted labels are. A collapsing model concentrates on a few labels.
- **Embedding shift.** Each answer is turned into a vector of numbers that captures its meaning (an embedding). This statistic measures how far the window's answers have moved in meaning from healthy ones.

The original design had seven signals. Two of them, refusal rate and JSON validity, never moved on healthy data (the model almost never refused and its JSON always parsed). A threshold on a constant is meaningless, so calibration set them aside.

## 5.2 Per-family (v3)

The same statistics, each computed within one task type against that type's own healthy answers, then averaged with each type's weight capped at its usual share of traffic. It needs task labels. (v2 was the same without the cap; it failed a test it had not seen, which is why v3 exists.)

## 5.3 Paired

The reference is asked each live question four times. Three statistics compare the four live answers with the four reference answers to the same question:

- the length distributions (Kolmogorov–Smirnov);
- the gap in agreement between samples;
- one minus the cosine similarity between the average embeddings of the two sides.

Lexical diversity was dropped here because it drifts between runs of the same model on different days (section 10.3).

## 5.4 Served answer

Production gives one answer per request. Asking it for three more costs money, and those extra answers are not what users actually saw. So the last monitor uses only the served answer. Section 6 explains it in full.

---

# 6. The served-answer test and its proof

## 6.1 The score

For one question, let *c* be the answer the user received, and *r₁, r₂, r₃, r₄* the reference model's four answers to the same question. Let *K(a, b)* measure how similar two answers are: the share of matching characters in the first 1,000 characters (the same kernel Model Equality Testing uses).

```
s = (average similarity between pairs of reference answers)
  − (average similarity between c and each reference answer)
```

In symbols, with k = 4 reference answers:

```
s = [ sum over i ≠ j of K(r_i, r_j) ] / (k(k−1))  −  [ sum over j of K(c, r_j) ] / k
```

The window statistic is the mean of *s* over the window's 200 questions. Because there is only one statistic, it gets the whole 0.1% false-alarm budget.

**Intuition.** If the served answer comes from the same model as the reference, it is just one more draw from that model, so on average it is as similar to the reference answers as they are to each other, and *s* averages zero. If the served model is different, its answer is less like the reference's answers than they are like each other, and *s* becomes positive.

## 6.2 What it measures: the proof

For a question *q*, write:

- *K_rr*: the expected similarity between two independent reference answers;
- *K_cc*: the expected similarity between two independent answers of the served model;
- *K_cr*: the expected similarity between one of each.

The squared maximum mean discrepancy between the two models' answers to *q* is

```
MMD²_q = K_cc + K_rr − 2 K_cr
```

**Proposition.**

```
E[s | q] = K_rr − K_cr = ½ · MMD²_q + ½ · (K_rr − K_cc)
```

**Proof.** The first term of *s* averages *K(r_i, r_j)* over pairs of distinct, independent reference answers, so its expectation is *K_rr*. The second term averages *K(c, r_j)*, whose expectation is *K_cr*. So *E[s | q] = K_rr − K_cr*. Rearranging the definition of MMD² gives *K_cr = ½ (K_cc + K_rr − MMD²_q)*; substituting gives the decomposition. ∎

## 6.3 Three consequences

1. **No change means zero, question by question.** If the served model is the reference model, *K_cc = K_rr = K_cr*, so *E[s | q] = 0* for every question. The window mean is an average of such terms, so its expectation is zero whatever questions the window holds. **No mix of questions can move it.** This is the mechanism behind the whole project, now a stated guarantee rather than an observation.
2. **A change shows up in two parts.** The score rises by half of the per-question MMD² (exactly what Model Equality Testing estimates), plus half of how much less alike the new model's answers are to each other than the reference's. Answers that vary more (for example, a higher sampling temperature) are easier to see.
3. **There is a blind spot.** If a change makes the served model's answers *more* alike than the reference's (more repetitive, more deterministic), the second term is negative and cancels part of the first. Model Equality Testing has no such blind spot, but it needs several production answers per question.

## 6.4 The proof checked against the data

Using the four production and four reference answers we have for every fault question, we estimated each term and compared the prediction with the measured mean of *s* (which uses the served answer alone). They agreed on all 14 conditions. Two rows explain the test's behaviour (values × 10⁻³):

| Condition | Measured score | ½ · MMD² | ½ · (K_rr − K_cc) |
|---|---|---|---|
| Temperature 1.0 | +5.8 | +1.4 | +6.5 |
| DeepSeek update, real traffic | +29.7 | +42.6 | −13.1 |
| Prompt edit (F3) | +44.2 | +159.4 | −115.1 |

- At temperature 1.0, almost all the signal is the self-similarity term, which MET does not see, so our test beats MET there.
- On real traffic, and for the prompt edit, the new answers are more alike than the old ones, the second term is negative, and part of the signal is cancelled. That explains why our test is a few points behind MET on real traffic.

---

# 7. How we set thresholds without fooling ourselves

The false-alarm budget is **0.1% per window**, counted *family-wise*: the chance that *any* of a monitor's statistics fires on a healthy window. Setting a threshold that honours this sounds easy and is not. Three traps would each inflate every number we publish.

## 7.1 Trap 1: several statistics share one budget

If each of five statistics were allowed 0.1%, the monitor as a whole would fire about five times too often. We split the budget with a **Šidák correction**:

```
α' = 1 − (1 − 0.001)^(1/m)
```

where *m* is the number of statistics. Our statistics are correlated, which makes this split conservative (slightly safer than needed), not wrong.

## 7.2 Trap 2: too few real windows

The 2,500 calibration questions give only about 46 distinct windows. You cannot read a 1-in-1,000 threshold from 46 values. So we draw **20,000 bootstrap windows**: random windows built from calibration questions under "no change".

## 7.3 Trap 3: the tail is still thin

At *α'*, only a handful of the 20,000 draws lie beyond the threshold. Reading the threshold directly from so few points is noisy. We fit a **generalised Pareto distribution** to the far tail (peaks over threshold) and read the threshold from the fit. Once, a maximum-likelihood fit came out degenerate on tied values (it produced a length threshold of 2 × 10¹⁶). Such fits now fall back to probability-weighted moments, and that record was withdrawn and refitted before any evaluation used it.

## 7.4 Calibrating across days

For the re-asking monitors, "no change" means two healthy answer sets for the same questions. We generate three healthy rounds on different days (called healthy, R and R2), calibrate on each of the three pairs, and keep the largest threshold. Section 10.3 shows why one pair is not enough.

---

# 8. How we kept the whole study honest

A monitoring study can lie without anyone intending it: tune a threshold after seeing faults, drop an awkward condition, or report the one lucky run. We built in five safeguards.

1. **Labels never reach the monitor.** A ground-truth grader (the **oracle**) scores accuracy only after detection is fixed. A test walks the code's import graph and fails the build if any monitoring code imports the grader, even indirectly.
2. **Frozen thresholds.** Each set of thresholds is saved in a record with the git commit, a content hash of the exact calibration rows, the model ids and digests, and the seeds. Every evaluation re-checks the record and refuses to run if one row changed.
3. **Pre-registration.** Every study's conditions, analysis and pass/fail criteria were written in a config file and committed to git before the answers that judge it existed.
4. **Dated amendments.** When something had to change later, we wrote an amendment saying what changed, why, and what had been seen by then. Some are awkward: a design changed after an earlier one failed; a round was paused and resumed; a planned control turned out to be impossible; the server software updated itself in the middle of a study. All are recorded.
5. **Known onsets and multiple replays.** Every change is injected at a known window, so we know exactly when an alarm is right or early. Every study was replayed five times with different random streams, and the spread is reported.

---

# 9. The data, the models and the changes we tested

## 9.1 The questions

5,000 questions over seven task types:

| Task type | Source | Graded? |
|---|---|---|
| Maths word problems | GSM8K | yes, exact answer |
| Trivia | TriviaQA | yes, fuzzy match |
| Reading comprehension | SQuAD 2.0 | yes, fuzzy match |
| Intent classification | Banking77 | yes, label accuracy |
| JSON extraction | jsonx (built by us) | yes, strict schema |
| One-sentence summaries | XSum | no |
| Long explanations | ELI5 | no |

They are split into calibration (2,500), held-out (1,000, used only to count false alarms) and fault (1,500) questions.

## 9.2 The healthy model

`gemma4:26b-a4b`: 25.8 billion parameters, a mixture-of-experts model with about 4 billion active per token, at 4-bit precision, run on one laptop through Ollama and pinned by its file digest. Every question is answered four times with distinct seeds, at temperature 0.7 and up to 512 tokens.

## 9.3 The changes

**Local faults** (all on the same 1,500 fault questions):

| | What changes | Mean accuracy change |
|---|---|---|
| F1 | a much smaller sibling, gemma4 e2b (4.6B) | −11.3 points |
| F2 | a smaller sibling, gemma4 e4b (8.0B) | −5.7 points |
| F3 | two lines added to the system prompt | −0.5 points |
| F5 | an older generation, gemma3 12B | −5.3 points |

**Harmless changes**, where the right answer is silence:

- **F4 and F6, topic shifts:** healthy answers replayed under a different question mix, towards long-form (F4) or short-form tasks (F6).
- **F7, a rerun:** the healthy model again with fresh seeds.
- **Q1:** DeepSeek's old model served at 4-bit by a different host.

**Real updates:**

- **DeepSeek V4 Flash**, 0731 → 1017, both versions pinned to one host (DeepInfra via OpenRouter) at the same precision. Accuracy went up (+0.2 to +8.6 points across tasks).
- **Qwen3.5 Plus**, 02-15 → 04-20, both served by Alibaba. Accuracy went down (strict JSON −26.7 points, trivia −6.9).

**Real traffic:** 3,000 first turns from WildChat (a public collection of real chatbot conversations), answered by both DeepSeek versions. No task types are known.

**The ladder:** the healthy model served differently, with the model and prompt unchanged: 8-bit weights (L1), or temperature 1.0, 1.3 and 1.6 (L2–L4).

## 9.4 Replay

Each condition gets 200 streams of 40 windows. A stream runs healthy answers until an onset at window 10 or later, then switches to the condition's answers to the same questions. We report:

- **detection:** the share of streams that alarm after the onset, with Wilson 95% intervals;
- **latency:** how many windows after the onset the first alarm comes;
- **false alarms:** alarms per healthy window before the onset;
- **held-out alarms:** alarms on real held-out windows.

When two monitors are compared, they see identical streams, and the comparison is an exact McNemar test.

---

# 10. Experiment by experiment

## 10.1 A change of topic looks like a change of model

**What we did.** Ran the pooled and per-family monitors on the local faults and the topic shifts.

**What came out.** The pooled monitor caught every fault (96–100% of streams within about a window), and also 98% of F4 streams and 100% of F6. The per-family monitor cut F6 to 23% and F4 to 4%, with false alarms of 0.053%.

**What it means.** Fixed-reference monitors mostly measure which kinds of questions a window holds. Task labels reduce the problem but do not remove it, and real traffic has no labels. This is the problem the rest of the project solves.

## 10.2 Re-asking removes the topic confusion

**What we did.** Ran the paired and served monitors on the same streams, and on real traffic with topics found by clustering (section 10.7).

**What came out.** Share of harmless-shift streams alarmed:

| Monitor | Local (F6) | DeepSeek (F6) | Real traffic (short topics) |
|---|---|---|---|
| Pooled (v1) | 100 | 100 | 80 |
| Per-family (v3) | 23 | 31 | needs labels |
| Paired | 0.5 | 0.5 | 0.0 |
| Served answer | 0.0 | 0.0 | 0.0 |

**What it means.** Comparing answer with answer cancels the change of topic, exactly as the proof predicts, including on real traffic where no task labels exist.

## 10.3 Calibration must span days

**What we did.** First calibrated the paired monitor on one pair of healthy answer sets, then on three pairs from different days.

**What came out.** With one pair it caught every fault but raised 0.21–0.59% false alarms, two to six times the budget. Two healthy rounds generated on different days sat about 10% further apart in embedding distance than the pair used for calibration, though each was just as varied on its own. With three pairs, keeping the largest threshold:

| Study | Faults / update caught | Short-form topic shift | False alarms |
|---|---|---|---|
| Local | 100% | 0.5% | 0.102% |
| DeepSeek, benchmark | 98.5% | 0.5% | 0.052% |
| DeepSeek, real traffic | 100% | n/a | 0.000% |
| Qwen | 98.5% | 0.5% | 0.000% |

**What it means.** A hosted model answers a little differently from one day to the next, and one pair of healthy runs cannot show how big that noise is. One honest caveat: the local study sits *on* its budget (0.102%, and 0.092–0.136% across five replays). The served test, calibrated the same way, raised no false alarm anywhere.

## 10.4 One served answer is enough

**What we did.** Pre-registered the served test, two kernels (characters as primary, embedding cosine as secondary) and three criteria before computing it on any data: S1, false alarms at or below 0.1%; S2, detection within 5 points of Model Equality Testing; S3, topic-shift alarms no higher than v3's.

**What came out (character kernel):**

| Study | Caught | Short-form topic shift | False alarms | MET caught |
|---|---|---|---|---|
| Local faults | 99–100% | 0.0% | 0.000% | 100% |
| DeepSeek update | 95.0% | 0.0% | 0.000% | 100% |
| DeepSeek on real traffic | 94.0% | n/a | 0.000% | 100% |
| Qwen update | 96.5% | 1.0% | 0.000% | 99.0% |

No false alarm in 87,716 healthy windows of the primary replay; one in all 25 replays together. S1 and S3 hold everywhere; S2 fails once, on real traffic, by one point (6.0 behind MET). The cosine kernel failed S1 on DeepSeek (0.234%), through the same day-to-day drift of embeddings.

**What it means.** With a quarter of the production answers, the served test does nearly everything the paired monitor does, and it is the steadiest monitor we have. MET, with four production answers, catches 2.5–6 points more and alarms a window or two sooner; section 6.4 explains exactly where and why.

## 10.5 Real updates, in both directions

| Change | Accuracy | v1 | v3 | Paired | Served | MET |
|---|---|---|---|---|---|---|
| DeepSeek update | +0.2 to +8.6 | 93.0 | 94.5 | 98.5 | 95.0 | 100 |
| … on real traffic | n/a | 90.0 | n/a | 100 | 94.0 | 100 |
| Qwen update | −0.3 to −26.7 | 96.0 | 96.5 | 98.5 | 96.5 | 99.0 |
| DeepSeek at 4-bit | within ±1.5 | 1.5 | 3.5 | 1.0 | 0.0 | 0.0 |

**What it means.**

- Both updates are caught within about one window, whichever way accuracy moved.
- The same old model at 4-bit from another host answers almost exactly as before, and no monitor flags it: a correct silence.
- v3 broke on Qwen: it alarmed on a plain rerun of the healthy model in 25% of streams and on 73.5% of topic shifts, at twice its false-alarm budget. Lexical diversity raised 47 of its 50 rerun alarms, the same day-to-day drift. The re-asking monitors held on the same data.

## 10.6 The same model, served differently

| Monitor | 8-bit | T 1.0 | T 1.3 | T 1.6 |
|---|---|---|---|---|
| v3 (fixed reference) | 1.0 | 50.5 | 90.5 | 99.0 |
| Paired | 96.0 | 63.5 | 92.5 | 99.5 |
| Served answer | 93.0 | 43.5 | 91.5 | 99.5 |
| MET | 96.5 | 15.5 | 89.5 | 98.5 |
| Accuracy change | +2.3 | −0.1 | +0.1 | +0.4 |

**What it means.**

- **Only re-asking sees the precision change.** The 8-bit answers have the same length and mix, so a fixed-reference monitor sees almost nothing (1%). Compared question by question with the 4-bit reference, they differ, and re-asking sees 93–96.5%. This row is the project's argument in miniature.
- **Higher temperature is caught as the answers scramble.** At T 1.0 the served test beats MET (43.5% against 15.5%), as the proof predicts.
- **None of it was harmful.** Accuracy did not fall anywhere, and the 8-bit model was 2.3 points better. Over all eight local faults, detection ranks only loosely with the accuracy drop (Spearman ρ = 0.55, p = 0.15). The monitors answer "did the model change?", not "did it get worse?".

## 10.7 Topics on real traffic, found without labels

**What we did.** Embedded the 3,000 WildChat prompts, clustered them into six topics (k-means, fitted on calibration prompts only), ranked the topics by answer length, and built two harmless shifts putting 70% of a window into the long-answer or short-answer topics.

| Monitor | Long topics | Short topics | False alarms |
|---|---|---|---|
| Pooled | 33.5% | 80.0% | 0.204% |
| Paired | 1.5% | 0.0% | 0.010% |
| Served answer | 0.0% | 0.0% | 0.000% |
| MET | 2.0% | 0.5% | 0.041% |

**What it means.** The topic problem is not an artefact of our benchmark. Real users cause it, and re-asking removes it.

## 10.8 What re-asking costs

**What we did.** Re-ran the served test with fewer reference answers per question (2, 3 or 4) and on only part of each window's questions (all, half or a quarter), recalibrating every setting.

| Reference calls per served question | 4 | 2 | 2 (half the questions) | 1 (a quarter) |
|---|---|---|---|---|
| Local faults (weakest) | 99.0% | 95.0% | 97.0% | 95.0% |
| DeepSeek update | 95.0% | 95.0% | 93.5% | 71.5% |
| … on real traffic | 94.0% | 94.0% | 85.0% | 34.0% |
| Qwen update | 96.5% | 96.0% | 94.5% | 94.0% |
| 8-bit | 93.0% | 52.0% | 82.0% | 65.0% |

**What it means.** Big changes are cheap to catch: one reference call per served question still finds the local faults and the Qwen update in 94–97.5% of streams. Subtle changes need the full budget. One setting broke the false-alarm budget (Qwen, three references on a quarter of the questions, 0.219%), and it is reported.

## 10.9 Slow faults: a model that stays wrong on part of the traffic

**What we did.** Replayed streams where a smaller model answers only part of the traffic after the onset (5%, 10%, 25%, 50% or 100%, or a 30-window ramp). Added two sequential tests, CUSUM and Page–Hinkley, which add up small shifts over many windows, calibrated on healthy streams to the same 0.1% budget.

| Bad share of traffic | Threshold | CUSUM | Page–Hinkley |
|---|---|---|---|
| F1 at 10% | 81.0% | 100% | 99.0% |
| F2 at 10% | 47.5% | 98.0% | 92.5% |
| F1 at 5% | 10.0% | 22.5% | 50.5% |
| F2 at 5% | 1.5% | 7.0% | 25.0% |

No detector raised a single false alarm before the onset. The same CUSUM on the per-family monitor broke its budget three to six times over.

**What it means.** A model that is wrong on a tenth of traffic is caught within about 14–19 windows. Because the served score is centred on zero for healthy answers on any questions (section 6.3), there is no offset for a sequential test to pile up.

## 10.10 After the alarm: was it harm?

**What we did.** After every alarm, ran two pre-designed follow-ups: a label-free readout (how answer length and agreement between samples moved) and a graded check (grade 50 or 100 pairs of old and new answers to the same questions; an exact McNemar test returns harm, improvement or no clear change). Applied to all 1,833 alarms of the served test.

**What came out.**

- **No verdict pointed the wrong way.**
- After the smallest model's alarms, the check confirmed harm in 199 of 200 at 100 pairs.
- After DeepSeek's update and the 8-bit build, it reported improvement.
- After every temperature rung, it reported "no clear change", which the oracle confirms.
- The label-free hint alone would have misled on temperature: agreement falls with temperature although accuracy does not.

**What it means.** An alarm should be read as "go and look", and a small graded check tells what was found.

## 10.11 Trying to close the blind spot: the hybrid test

**What we did.** The blind spot is the self-similarity term, which needs a second production answer to estimate. We pre-registered a hybrid: on 10% of questions, ask production for one extra answer and use it to estimate the missing term, giving a full MMD estimate at 1.1 production answers per question.

**What came out.**

| | Served | Hybrid (10%) | Hybrid (50%) | MET |
|---|---|---|---|---|
| DeepSeek, real traffic | 94.0 | 93.5 | 96.0 | 100 |
| DeepSeek update | 95.0 | 95.0 | 95.5 | 100 |
| Qwen update | 96.5 | 96.0 | 96.0 | 99.0 |
| T 1.0 | 43.5 | 40.5 | 40.5 | 15.5 |

It kept the false-alarm budget but failed two of its three criteria: it did not beat the served test, and it did not close the gap to MET on real traffic.

**What it means.** One extra answer on a tenth of the questions is too noisy an estimate, and splitting the budget across two statistics costs as much as it gains. The blind spot is real and not cheap to close; MET's advantage comes from several production answers on every question. We report this as a pre-registered negative result.

## 10.12 Baselines

On the same local streams:

| Monitor | F1 | F2 | F3 | F5 | F6 (harmless) | False alarms |
|---|---|---|---|---|---|---|
| Per-family (v3) | 97.0 | 97.5 | 100 | 97.0 | 23.0 | 0.053% |
| Paired | 100 | 100 | 100 | 100 | 0.5 | 0.102% |
| Served answer | 100 | 99.5 | 99.0 | 99.0 | 0.0 | 0.000% |
| Model Equality Testing | 100 | 100 | 100 | 100 | 1.0 | 0.054% |
| Embedding MMD | 93.5 | 86.5 | 91.5 | 85.0 | 94.0 | 0.587% |
| LLM judge, Qwen 2.5 14B | 89.5 | 49.0 | 93.5 | 3.0 | 11.5 | 0.535% |
| LLM judge, Qwen 2.5 72B | 95.5 | 88.5 | 92.0 | 57.0 | 8.5 | 0.047% |

- **Embedding MMD**, a standard drift test, alarms on 94% of the harmless shift at six times the budget: it answers "did the traffic change?".
- **An LLM judge** that grades each answer from 1 to 10 grades style more than correctness. It hardly sees the older-generation model (3% at 14B, 57% at 72B), whose answers read well but are about 5 points less accurate.

## 10.13 How much the numbers move between replays

**What happened.** When the 8-bit round joined the ladder, the temperature numbers moved by up to 7 points although their answers were unchanged. Each condition's streams had been seeded by its position in a list, so adding one reshuffled the others.

**What we did.** Pre-registered a fix: streams are seeded by the condition's name, and every study is replayed five times with thresholds fixed. Replay 0 is primary and gives every number in this document; replays 1–4 give the spread.

**What came out.** The served test is the steadiest (one false alarm in 25 replays); the local paired monitor the least (0.092–0.136% false alarms); where no monitor is sure (T 1.0), detection moves by up to 10 points.

**What it means.** One replay carries more noise than its binomial interval suggests, so the spread is reported beside the headline rates. The rerun also uncovered a real weakness, the local paired monitor's budget, that a lucky single replay had hidden.

## 10.14 Checking the server version

**What happened.** The Ollama server software that runs the local model updated itself between rounds (0.35.0 for the early local rounds, 0.40.3 for the reference rounds and ladder, 0.40.9 for part of a later round). A server update could itself change the answers slightly.

**What we did.** Compared two healthy rounds made under different server versions (0.35.0 against 0.40.3) with two made under the same version (0.40.3 against 0.40.3), on the calibration questions.

| No-change pair | Mean served score (×10⁻³) | MMD² (×10⁻³) |
|---|---|---|
| healthy (0.35.0) → R (0.40.3) | −1.06 ± 0.75 | −0.30 ± 0.58 |
| healthy (0.35.0) → R2 (0.40.3) | +0.28 ± 0.74 | −0.40 ± 0.57 |
| R (0.40.3) → R2 (0.40.3) | −0.39 ± 0.73 | −0.38 ± 0.55 |

**What it means.** Across versions looks exactly like within one version: centred on zero with the same spread. The server change did not drive any result. For the blind confirmatory study we reinstalled 0.40.3 so its new rounds come from one version.

## 10.15 The blind confirmatory study

**Why.** Several of our designs came after an earlier one failed. Each was frozen before the data that judged it, but a sceptic can still ask whether the final pipeline is tuned to our own data. The cleanest answer is to test it, frozen, on changes it has never seen, with no amendments allowed after scoring.

**What it tests.** Five unseen changes, scored by every monitor exactly as frozen:

- C1, C2, C3: the deployment silently swapped to Qwen 2.5 14B, Llama 3.1 8B and Phi-4 14B;
- C4: a new prompt edit ("write for a general audience, prefer plain words, add a short example");
- C5: the answer limit cut from 512 to 256 tokens.

**Criteria (fixed in advance).** False alarms at or below 0.1%; C1–C3 each detected in at least 90% of streams; C4 and C5 reported without a criterion.

**What came out.**

| Change | Accuracy | v3 | Paired | Served | MET |
|---|---|---|---|---|---|
| C1, Qwen 2.5 14B | −4.5 | 97.5 | 100 | 100 | 100 |
| C2, Llama 3.1 8B | −8.6 | 99.5 | 100 | 99.0 | 100 |
| C3, Phi-4 14B | +0.3 | 98.5 | 100 | 100 | 100 |
| C4, prompt edit | −0.6 | 90.0 | 98.0 | 98.0 | 100 |
| C5, 256-token limit | −6.3 (maths −32) | 94.5 | 96.5 | **0.0** | 69.5 |
| False alarms | | 0.038% | 0.078% | 0.000% | 0.053% |

Both criteria hold: no false alarm in 24,420 healthy windows, and each vendor swap caught in 99–100% of streams. The new prompt edit was caught in 98%.

**The miss.** The served test did not see C5 at all, and C5 was harmful: maths accuracy fell from 95.6% to 63.7%, because the final answer sits on the last line and was cut off. The cause is mechanical. With the same seeds, a cut answer is identical to the healthy answer up to the cut, and the character kernel compares only the first 1,000 characters, about 250 tokens, so the cut lies at the edge of what it reads. The length statistics of v3 and the paired monitor caught it in 94.5–96.5% of streams.

**What it means.** The frozen pipeline is not fitted to our own data: it passed a blind test on changes it had never seen. It also has a hard edge we had not stated before: the served test cannot see what happens after the first 1,000 characters. Keeping a cheap answer-length check beside it, which needs no extra calls, would have caught C5. That pairing is our recommendation, not something the blind test tested.

---

# 11. What it all means

## 11.1 For someone running a model in production

1. A monitor that compares live traffic with a fixed healthy reference will alarm on ordinary changes in what users ask. Task labels reduce this but do not remove it, and real traffic has none.
2. Keep the old model available and re-ask it the live questions. Comparing answer with answer separates a changed model from changed users, on unlabelled traffic too.
3. You need no extra production answers. One served answer per question catches real updates in 94–96.5% of streams with no false alarm in about 88,000 windows.
4. Re-asking can be cheap: one reference call per served question is enough for big changes; subtle ones need four. A CUSUM on the served score catches a bad model serving a tenth of traffic.
5. Calibrate on at least three healthy rounds from different days, and keep a cheap answer-length check beside the served test.
6. Read an alarm as "something changed, go and look", then grade a few dozen answer pairs. Two of the changes we caught made the model better.

## 11.2 For research

- A harmless-shift control should be standard in drift-detection evaluations. Without one, "100% detection" says nothing.
- Pre-registration with dated amendments is practical for ML systems work. It kept us honest when designs failed, which they did several times.
- Replay-to-replay variation is real and worth reporting next to any detection rate.
- The served statistic's decomposition turns a heuristic into a method with a stated guarantee and a stated blind spot.

## 11.3 How it connects to the course themes

- **Systems degrade without crashing and can stay wrong for a long time:** we notice silent changes, and catch a model that is wrong on a tenth of traffic.
- **Evaluation can lie:** frozen thresholds, pre-registration, harmless-shift controls, and reporting every failure.
- **Reproducibility:** frozen records, five replays, a server-version check, one script per analysis.

---

# 12. What it cannot do

- **Two real updates from two vendors**, one of them also on real traffic. More vendors would strengthen the claim.
- **Local models are 4.6–26B open-weight models on one machine**, from one family.
- **Streams are replays of real answers**, not a live deployment.
- **The re-asking monitors need a reference that stays available** (a kept checkpoint or a pinned API version) and reference calls: four per question at full power, fewer for large changes.
- **Several designs came after an earlier one failed** (v3, the three-round calibration, the served test). Each was frozen before the data that judged it, and every revision is recorded; the blind study addresses this directly.
- **No answer-only monitor sees a fault that leaves the answers looking the same.** Faults confined to intent labels or JSON were caught in at most 3.5% of streams, because a wrong label looks like a right one. The served test is also weakest when a change makes answers more alike.
- **The served test reads only the first 1,000 characters** of each answer, so it missed answers cut at 256 tokens in the blind test.
- **The local paired monitor sits on its budget,** and rates move between replays, by up to 10 points where detection is uncertain.
- **Versions and precisions of hosted models are the hosts' own labels.**

---

# 13. How the project unfolded

| Stage | What happened |
|---|---|
| September 2026 | Project brief: a label-free monitor with frozen thresholds and planted faults |
| 10–12 October | Healthy and fault rounds generated locally; v1 frozen; v1 alarms on topic shift F4 |
| 13–19 October | v2, then v3 (per-family, capped); transfer to three other model families; seed and leakage checks |
| 20–21 October | Hosted DeepSeek study; the real update on 17 October caught; real WildChat traffic; 72B judge |
| 21–22 October | Paired monitor; one-day calibration over budget; three-round calibration fixes it |
| 22 October | Served-answer test pre-registered and run; Qwen real update; Model Equality Testing baseline |
| 22–23 October | Same-model ladder; replay seeding fixed; five replays of every study |
| 23 October | Proof of what the served score measures; cost against power; slow faults; triage of alarms |
| 23–28 October | Hybrid test (pre-registered negative); blind confirmatory study pre-registered; server version checked |
| 28 October | Blind study scored frozen: both criteria hold; answers cut short missed |

---

# 14. How to run it, and what it cost

## 14.1 Running it

```
uv sync
uv run pytest -m gate                           # architectural gates
scripts/generate_all.sh                         # local rounds and embeddings
scripts/analyse_all.sh                          # fixed-reference analyses
scripts/replay_all.sh                           # five replays of every study
uv run python scripts/served_extensions.py x1 local   # theory check
uv run python scripts/hybrid_test.py local            # hybrid test
python3 scripts/port_papers.py                  # ICML and NeurIPS texts from ACL
```

Every evaluation verifies its frozen record and refuses to run on a mismatch. Interrupted generation resumes from its checkpoint.

## 14.2 Cost

| Item | Cost |
|---|---|
| DeepSeek study | $1.30 |
| Real traffic | $2.08 |
| 72B judge | $6.81 |
| Reference rounds for those two studies | $1.76 |
| Qwen study | $8.26 |
| Qwen reference rounds | $5.90 |
| **Total hosted** | **$26.11** |

Local generation (well over 100,000 answers) ran free on one laptop (Apple M5 Pro, 48 GB). Five replays of every study take about 40 minutes.

## 14.3 The papers

- **ACL version:** "Did the Model Change, or Did the Users? Label-Free Monitoring of Deployed Language Models", main text 6 of 8 pages, for ACL Rolling Review (deadline 4 January 2027).
- **ICML and NeurIPS versions:** built from the ACL text by one script.

---

# 15. Presentation script for sir

About 15 minutes for 13 slides, with time for questions. Speakers are a suggestion.

**Slide 1, title (Dhruv).** "Good morning sir. Our project is driftwatch. A language model behind an API can change without anyone being told, and nothing crashes when it does. We asked whether we can notice that change from the answers alone, with no labels, and without being fooled when users simply ask different things. Hence the title: did the model change, or did the users?"

**Slide 2, the problem (Dhruv).** "This happened during our project: on 17 October DeepSeek replaced its model under the same name. Teams also swap in cheaper models, lower precision, or edit a prompt. Dashboards stay green. Checking quality needs right answers, and nobody labels live traffic. So our constraint: no labels, and a false-alarm rate fixed in advance, one per thousand windows."

**Slide 3, why it is hard (Pronnati).** "Our first monitor compared live answers with a fixed healthy reference. It caught every fault, but when only the mix of questions changed it alarmed in 98–100% of streams. Comparing each task type with itself helped, to about one shift in four, but needs labels real traffic doesn't have."

**Slide 4, the idea (Mahak).** "So we keep the old model and ask it the same live questions, then compare answer with answer. A topic shift moves both sides equally, so it cancels. The paired monitor uses four production answers per question; our served test only the one the user got."

**Slide 5, the served test (Mahak).** "For each question: is the served answer as similar to the reference's four answers as those are to each other? We proved its expected score is half the MMD plus a self-similarity term, exactly zero for every question when nothing changed. The same formula shows its blind spot: answers becoming more alike are harder to see."

**Slide 6, honesty (Aditya).** "The budget is set first: 20,000 bootstrap windows and a fitted tail because there are only 46 real windows. Thresholds are frozen with a hash. Every study was pre-registered in git, and every change is a dated amendment. Every change starts at a known moment, replayed five times. Labels only score accuracy afterwards."

**Slide 7, experiments (Aditya).** "Planted faults: smaller models, an older model, a prompt edit. Two real updates: DeepSeek better, Qwen worse. Real chatbot prompts. The same model served differently. And harmless changes where silence is correct."

**Slide 8, topic shifts (Pronnati).** "Fixed references alarmed on 80–100% of harmless shifts, the per-family monitor 23–31%, re-asking 0–0.5%. On real traffic, topics found by clustering with no labels: the served test alarmed on none."

**Slide 9, real updates (Dhruv).** "Both updates caught within about one window: 95% DeepSeek, 94% on real traffic, 96.5% Qwen, 99–100% our faults. Zero false alarms in nearly 88,000 windows. MET is a few points stronger but needs four production answers. v3 broke its budget on Qwen."

**Slide 10, the ladder (Mahak).** "With 8-bit weights the answers look the same in aggregate: the fixed reference sees 1%. Question by question they differ: re-asking sees 93%. None of these changes hurt accuracy; 8-bit was 2.3 points better. Monitors answer 'did it change?', not 'did it get worse?'."

**Slide 11, practical (Aditya).** "Calibrate across days: one day gave two to five times too many false alarms. Re-asking can be cheap: one reference call per question catches big changes. And a model wrong on 10% of traffic is caught in 98–100% of streams with CUSUM, with no false alarms."

**Slide 12, alarm to action (Pronnati).** "An alarm means 'go and look'. Grading 50–100 answer pairs after each alarm gave no wrong verdict in 1,833 alarms. It can't see faults that leave answers looking the same, or changes that make answers more alike; our cheap fix for that didn't work, and we report it."

**Slide 13, takeaways (Dhruv).** "A fixed reference can't tell a changed model from changed users; re-asking can, without labels; one served answer is enough, with zero false alarms and a proof of why. Calibrate across days; treat an alarm as a reason to look. In a blind test on five unseen changes, run frozen, it caught four with no false alarm and missed answers cut short, which we explain. We target ACL Rolling Review on 4 January. Thank you."

---

# 16. Questions sir may ask, with answers

**Why not just run a labelled benchmark every day?**
It is slow, costs grading, and tests a fixed question set rather than what users ask. Our monitor runs on live traffic with no labels; a small graded check is the follow-up after an alarm.

**Isn't this just drift detection?**
Drift detection asks whether the inputs changed. We treat an input-only change as a false alarm and ask whether the model changed. Embedding MMD, a standard drift test, alarmed on 94% of harmless shifts.

**What is new compared with Model Equality Testing?**
MET needs several production answers per prompt and prompts the auditor chooses. Our test uses the one served answer on real traffic, and we proved what it measures (half of MET's distance plus a self-similarity term).

**Why can't a topic shift fool the served test?**
It compares answers to the same question, and with no model change its expected score is zero for every question, so any mix of questions averages to zero.

**Is keeping the old model available realistic?**
For a self-hosted model, keep the checkpoint; many APIs serve dated versions (we used DeepSeek and Qwen this way). If a version is retired, store reference answers in advance. It is listed as a limitation.

**Doesn't re-asking cost a lot?**
Four reference calls per question at full power; one call per served question is enough for big changes, and you can monitor a sample of traffic.

**How did you set the threshold?**
On healthy calibration answers only: 20,000 bootstrap windows, a generalised Pareto tail fit, a Šidák split, and thresholds frozen with a hash before any fault was seen.

**Why three days of calibration?**
Hosted models vary from day to day; one pair gave two to five times too many false alarms; three pairs fixed it.

**Your designs changed after failures. Isn't that tuning?**
Each revision was frozen before the data that judged it and recorded as a dated amendment. The blind confirmatory study then tested the frozen pipeline on five unseen changes with no amendments allowed: both criteria held, and it missed one change, answers cut short.

**The blind test missed the 256-token cut, which hurt accuracy. Isn't that a failure?**
Yes, for the served test alone, and we report it. It compares only the first 1,000 characters, and a cut answer equals the healthy one up to the cut. Length statistics caught it in 94.5–96.5%, so we recommend a free length check beside it.

**Why replays and not live traffic?**
Replays of real answers give an exact onset, so latency and early false alarms can be measured. A live deployment is future work.

**You caught changes that made the model better. Is that a failure?**
No: the monitor answers "did it change?". The graded check answers "was it worse?" and never pointed the wrong way in 1,833 alarms.

**What can it not detect?**
Faults that leave answers looking the same, and changes that make answers more alike. Our cheap fix for the second did not work.

**Why is the served test behind MET on real traffic?**
The new model's answers are more alike than the old one's, so the self-similarity term is negative; MET sees it with several production answers.

**The local paired monitor is at 0.102%. Why report it?**
Because it is what happened. It is one reason we recommend the served test, which raised no false alarm anywhere.

**Did the server version change affect results?**
We checked: across versions looks exactly like within one version. For the blind study we reinstalled the original version.

**Does it generalise beyond two vendors and 26B models?**
Two real updates in opposite directions, three other model families as healthy models, and real chatbot traffic. More vendors and larger models are future work.

**What would you do with more time?**
A live deployment, more vendors' updates, larger models, and a better fix for the blind spot.

**Where will you submit?**
ACL Rolling Review (4 January 2027); ICML and NeurIPS versions are ready too.

---

# 17. Glossary

| Term | Meaning |
|---|---|
| Window | 200 consecutive questions; every statistic is computed over one |
| False-alarm budget | the share of healthy windows allowed to alarm, here 0.1% |
| Calibration | learning from healthy answers what "normal" looks like, and setting thresholds |
| Frozen record | thresholds saved with a hash of their data, so nobody can change them quietly |
| Pre-registration | committing the plan and pass/fail criteria before the data exists |
| Amendment | a dated, recorded change to a pre-registered plan |
| Re-asking | sending the live questions to a reference model to compare answer with answer |
| Served answer | the one answer a user actually received |
| Embedding | a list of numbers that captures the meaning of a text |
| Kolmogorov–Smirnov statistic | the largest gap between two distributions |
| MMD | maximum mean discrepancy: a distance between two sets of samples |
| MET | Model Equality Testing (ICLR 2025): an MMD test that an API serves a given model |
| Šidák split | dividing the false-alarm budget across several statistics |
| Generalised Pareto tail | a fit to the far end of a distribution, to read a rare threshold reliably |
| Bootstrap | building many random windows from the calibration questions |
| CUSUM / Page–Hinkley | sequential tests that add up small shifts over windows |
| Replay | a stream built from real recorded answers, with a known moment of change |
| Wilson interval | a confidence interval for a proportion |
| McNemar test | a test comparing two monitors on the same streams |
| Topic shift | a harmless change: same model, different mix of questions |
| Oracle | the ground-truth grader; used only afterwards, never to monitor |
