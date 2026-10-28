# ARR Responsible NLP Research checklist: answers

For the ACL/ARR submission (`paper/acl/driftwatch_acl.tex`). Section and appendix
references are to that version as built on 23 October 2026. ARR's rule: answer Yes
with the section, or No with a justification. A No is not grounds for rejection.

Every answer is settled. D (human participants) is No, and E1 (AI assistance) is
Yes, disclosed in the paper's Ethics Statement. Before pasting into OpenReview,
check the section and appendix letters against the final PDF.

---

## A. For every submission

**A1. Did you describe the limitations of your work?**
Yes. See the *Limitations* section, items (i)–(viii):

- two real updates from two vendors;
- local models of 4.6–26B from one machine and one family;
- replayed rather than live streams;
- a reference that must stay available;
- designs revised after earlier ones failed, each frozen before its data;
- faults invisible in the answers;
- hosts' own version labels;
- rates that move between replays, and a local paired monitor at its budget.

The results report where the monitors fail:

- §6.1, fixed-reference monitors alarming on topic shifts;
- §6.2 and Appendix C, the per-family monitor over budget on Qwen;
- §6.3, the local paired monitor at its budget;
- §6.4, the served-answer test missing one criterion on real traffic;
- §6.8 and Appendix L.4.3, faults confined to classification or extraction.
- Appendix F, the served-answer test's blind spot when a change makes answers
  more alike;
- §6.6, the blind test: the served-answer test missed answers cut at 256 tokens,
  beyond the 1,000 characters it compares;
- Appendix H, the hybrid test that did not close the blind spot;
- §6.4, how detection of subtle changes falls when re-asking is cut.

Appendix M states the assumptions behind the false-alarm guarantee and where they
hold only approximately.

**A2. Did you discuss any potential risks of your work?**
Yes. The *Ethics Statement*, "Use and misuse", covers the main risk of misreading
the monitor:

- An alarm signals change, not harm: the DeepSeek update (§6.2) and the 8-bit
  build (§6.5) both improved accuracy and were caught.
- An alarm calls for a person to look, not for an automatic decision (§7); a
  small graded check after each alarm tells harm from improvement (Appendices S
  and J).
- Silence is not evidence that a model is safe or correct (Appendix L.4.3).

Other risks:

- **Privacy:** the real-traffic data is filtered and stripped of metadata
  (*Ethics Statement*, "Data"; Appendix T).
- **Environmental cost:** small. One laptop and $26.11 of hosted inference in total
  (Appendix V).
- **Fairness:** the work studies no model bias, so this does not apply.

## B. Did you use or create scientific artifacts?

Yes.

**B1. Did you cite the creators of artifacts you used?**
Yes.

- **Datasets:** every benchmark dataset is cited in Appendix L.2, and WildChat in
  §5 and Appendix T. Appendix U gives the Hugging Face id and split of each. The
  exact pinned revisions are in the supplementary code (`src/corpus/build.py`,
  `src/config/chat_study.yaml`).
- **Models:** named in §5 (local models, DeepSeek and Qwen), §6.7 (judges) and
  Appendix P (transfer models), with sizes in Appendix V. Their build digests are in
  the code.
- **Prior method:** Model Equality Testing is cited in §2 and §4 and run as a
  baseline in §6.
- **Embedding model and software:** cited in §4; versions in Appendix V.

**B2. Did you discuss the license or terms for use and/or distribution of any artifacts?**
Yes. Appendix U (Table "Datasets and licences") lists the licence each dataset card
declares:

| Dataset | Licence |
|---|---|
| GSM8K | MIT |
| SQuAD 2.0 | CC BY-SA 4.0 |
| Banking77 | CC BY 4.0 |
| WildChat-1M | ODC-BY |
| TriviaQA, XSum, ELI5 | none declared |

The *Ethics Statement* says how the undeclared ones are used: non-commercial
research only, with no source text redistributed.

**B3. Did you discuss if your use of existing artifacts was consistent with their intended use?**
Yes. The *Ethics Statement*, "Data", and Appendix U:

- **Benchmark datasets:** used for research evaluation, as released.
- **WildChat:** used as a research release under ODC-BY.
- **Hosted models:** used through their public APIs under the router's terms, paid
  for by the authors (*Ethics Statement*, "Models and services").
- **Our own artifacts:** we release code and frozen threshold records only, no
  question or answer text, so nothing derived from the datasets is redistributed.

**B4. Did you discuss the steps taken to check whether the data contains personally identifying information or offensive content?**
Yes. The *Ethics Statement*, "Data", and Appendix T.

- **WildChat**, the only source of real user text: we keep a first user turn only
  if no flag marks the conversation or the turn as toxic or redacted (the dataset's
  PII flag), and the logged moderation result did not flag it.
- **Metadata:** only the turn's text is read. Country, state, IP hash, request
  headers, timestamps and model fields are never read into the corpus.
- **Benchmark datasets:** standard public research sets; we did not screen them
  further.
- **Sending text out:** no personal information was collected, and only the
  benchmark questions and these filtered prompts were sent to the hosted APIs.

**B5. Did you provide documentation of the artifacts?**
Yes.

- **Tasks and quotas:** Appendix L.2 lists the seven task families, their quota per
  window and the oracle used.
- **Sources and fields:** Appendix U gives sources, splits and the fields read.
- **Real traffic:** Appendix T describes the WildChat sample, filters and split;
  Appendix D the topics found by clustering.
- **Language and demographics:** all data is English. The demographics of WildChat
  users are not known to us and are not reported.
- **jsonx:** our synthesised task is described in Appendix L.2, and its generator
  and JSON schema are in the supplementary code.

**B6. Did you report relevant statistics like the number of examples, details of train/test/dev splits?**
Yes. There is no training. The splits are:

| Study | Where | Questions | Calibration | Held-out | Fault |
|---|---|---|---|---|---|
| Local, DeepSeek, Qwen | §5 | 5,000 | 2,500 | 1,000 | 1,500 |
| Real traffic | §5, Appendix T | 3,000 | 1,500 | 600 | 900 |

- **Answer counts:**
  - local: 64,000 main answers, 60,000 for the transfer deployments (Appendix P),
    and 48,000 for the reference rounds and the ladder (Appendix V);
  - hosted: 38,000 for DeepSeek (Appendix L.5), 19,200 for real traffic
    (Appendix T), and 56,000 for Qwen with its reference rounds (Appendix C).
- **Per-slice detail:** per-slice counts and per-window quotas are in Appendix L.2.

## C. Did you run computational experiments?

Yes.

**C1. Did you report the number of parameters in the models used, the total computational budget, and computing infrastructure used?**
Yes. Appendix V (Compute and software) gives:

- the parameters of every local model, judge and the embedding model;
- the hardware: one Apple M5 Pro laptop with 48 GB, running Ollama 0.40.3;
- the time:
  - about 31 hours of generation for the main answers;
  - at most 162 hours for the transfer rounds, including interruptions;
  - about 9 hours for the ladder;
  - about 4.5 hours for the 14B judge;
  - about 40 minutes for five replays of every study;
- the hosted work: $26.11 in total, itemised by study.

**C2. Did you discuss the experimental setup, including hyperparameter search and best-found hyperparameter values?**
Yes. §3, §5 and Appendix V.

- **No hyperparameter search:** decoding (k = 4, temperature 0.7, 512 tokens),
  window (W = 200, S = 50), budget (0.1%) and bootstrap (20,000 windows, GPD tail)
  were fixed before calibration.
- **Calibration:** thresholds are calibrated on the calibration split only and
  frozen before evaluation. The re-asking monitors calibrate on three healthy
  rounds and keep the largest threshold (§3, §6.3).
- **Pre-registration:** every study's design and criteria were committed before
  its data existed; every amendment is listed in Appendix A.
- **Sensitivity:** calibration size and W are varied after the fact, as
  sensitivity analyses (Appendix X).

**C3. Did you report descriptive statistics about your results?**
Yes.

- **Intervals and streams:** every detection and false-alarm rate comes from 200
  replayed streams per condition, with Wilson 95% intervals (§5).
- **Paired tests:** comparisons between monitors on identical streams use exact
  McNemar tests (§5, §6).
- **Replays:** every study was replayed five times with thresholds fixed; the main
  text reports the primary replay, and Appendix E gives the range of each rate
  over all five.
- **Blind test:** a pre-registered confirmatory study on five unseen changes,
  scored frozen with no amendment allowed (§6.6).
- **Server versions:** a check that the local server's version did not change
  results (Appendix K).
- **Seeds:** ranges over ten calibration seeds are in Appendix N, and intervals from
  50 question half-samples in Appendix W.
- **Splits:** the calibration, held-out and fault splits are separate. Held-out
  false alarms are reported for every monitor.

**C4. If you used existing packages, did you report the implementation, model, and parameter settings used?**
Yes. Appendix V:

- **Software:** Ollama 0.40.3; Python 3.11 with NumPy 2.4.6, SciPy 1.17.1,
  pandas 3.0.5, PyArrow 25.0.1 and Hugging Face Datasets 5.0.1. The supplementary
  lock file pins every dependency.
- **Models:** local models are Q4_K_M builds pinned by digest, plus one Q8_0 build
  for the ladder; the embedding model is Nomic Embed v1.5.
- **Changes:** no package was modified.

## D. Did you use human annotators or research with human participants?

No. Correctness is scored automatically against each dataset's own answers. No
annotators or participants were recruited, so D1–D5 do not apply.

## E. Did you use AI assistants in your research, coding, or writing?

Yes.

**E1. Did you include information about your use of AI assistants?**
Yes. See the *Ethics Statement*, "AI assistance":

> We used an AI coding and writing assistant to help implement, run and test the
> code and to draft and edit the text. It also proposed analyses. The authors
> decided which to run, made every design and analysis decision, checked every
> reported number against the result files, and take responsibility for all
> claims.

This follows ACL's publication ethics policy: the assistance is disclosed, and the
assistant is not an author.
