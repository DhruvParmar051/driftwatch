# ARR Responsible NLP Research checklist: draft answers

For the ACL/ARR submission (`paper/acl/driftwatch_acl.tex`). Section and appendix
references are to that version. ARR's rule: answer Yes with the section, or No with
a justification. A No is not grounds for rejection.

Before pasting into OpenReview, check:

- E1 (marked **TEAM DECISION**): the team must agree the answer.
- D (marked **VERIFY**): one fact to confirm against the WildChat paper.

---

## A. For every submission

**A1. Did you describe the limitations of your work?**
Yes. See the *Limitations* section, items (i)–(vii): one vendor's local models,
replayed rather than live streams, known task families, v2/v3 designed after F4,
jsonx scoring, reference-free judges, and one real update from one provider.
§6 reports where the monitor fails:

- §6.2, doses of 10% or less;
- §6.3, faults confined to classification or extraction slices;
- §6.5, the gemma3 transfer failure.

Appendix A states the assumptions behind the false-alarm guarantee and where they
hold only approximately (the hosted host's seeds).

**A2. Did you discuss any potential risks of your work?**
Yes. The *Ethics Statement*, "Use and misuse", covers the main risk of misreading
the monitor:

- An alarm signals change, not harm; the real update in §7 improved accuracy.
- An alarm calls for a person to look, not for an automatic decision.
- Silence is not evidence that a model is safe or correct (§6.3).

Other risks:

- **Privacy:** the real-traffic data is filtered and stripped of metadata (*Ethics
  Statement*, "Data"; Appendix H).
- **Environmental cost:** small. One laptop and about $10 of hosted inference
  (Appendix J).
- **Fairness:** the work studies no model bias, so this does not apply.

## B. Did you use or create scientific artifacts?

Yes.

**B1. Did you cite the creators of artifacts you used?**
Yes.

- **Datasets:** every dataset paper is cited in §4, plus WildChat in Appendix H.
  Appendix I gives the Hugging Face id and split of each. The exact pinned
  revisions are in the supplementary code (`src/corpus/build.py`,
  `src/config/chat_study.yaml`).
- **Models:** named in §4, §5 (judges) and §7 (DeepSeek), with sizes in
  Appendix J. Their build digests are in the code.
- **Embedding model and software:** cited in §3.2; versions in Appendix J.

**B2. Did you discuss the license or terms for use and/or distribution of any artifacts?**
Yes. Appendix I (Table "Datasets and licences") lists the licence each dataset card
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
Yes. The *Ethics Statement*, "Data", and Appendix I:

- **Benchmark datasets:** used for research evaluation, as released.
- **WildChat:** used as a research release under ODC-BY.
- **Our own artifacts:** we release code and frozen threshold records only, no
  question or answer text, so nothing derived from the datasets is redistributed.

**B4. Did you discuss the steps taken to check whether the data contains personally identifying information or offensive content?**
Yes. The *Ethics Statement*, "Data", and Appendix H.

- **WildChat**, the only source of real user text: we keep a first user turn only
  if no flag marks the conversation or the turn as toxic or redacted (the dataset's
  PII flag), and the logged moderation result did not flag it.
- **Metadata:** only the turn's text is read. Country, state, IP hash, request
  headers, timestamps and model fields are never read into the corpus.
- **Benchmark datasets:** standard public research sets; we did not screen them
  further.
- **Sending text out:** no personal information was collected, and only these
  filtered prompts were sent to the hosted API.

**B5. Did you provide documentation of the artifacts?**
Yes.

- **Tasks and quotas:** Table 1 (§4) lists the seven task families, their quota
  per window and the oracle used.
- **Sources and fields:** Appendix I gives sources, splits and the fields read.
- **Real traffic:** Appendix H describes the WildChat sample, filters and split.
- **Language and demographics:** all data is English. The demographics of WildChat
  users are not known to us and are not reported.
- **jsonx:** our synthesised task is described in §4, and its generator and JSON
  schema are in the supplementary code.

**B6. Did you report relevant statistics like the number of examples, details of train/test/dev splits?**
Yes. There is no training. The splits are:

| Study | Where | Questions | Calibration | Held-out | Fault |
|---|---|---|---|---|---|
| Local | §4 | 5,000 | 2,500 | 1,000 | 1,500 |
| Real traffic | Appendix H | 3,000 | 1,500 | 600 | 900 |

- **Answer counts:** 64,000 local, 38,000 hosted (§7) and 19,200 real-traffic
  (Appendix H).
- **Per-slice detail:** per-slice counts and per-window quotas are in Table 1.

## C. Did you run computational experiments?

Yes.

**C1. Did you report the number of parameters in the models used, the total computational budget, and computing infrastructure used?**
Yes. Appendix J (Compute and software) gives:

- the parameters of every local model, judge and the embedding model;
- the hardware: one Apple M5 Pro laptop with 48 GB, running Ollama 0.40.3;
- the time: about 31 hours of generation for the main answers, at most 162 hours
  for the transfer rounds (including interruptions) and about 4.5 hours for the
  14B judge;
- the hosted work: about 8 hours, for about $10.

**C2. Did you discuss the experimental setup, including hyperparameter search and best-found hyperparameter values?**
Yes. §3, §4 and Appendix J.

- **No hyperparameter search:** decoding (k = 4, temperature 0.7, 512 tokens),
  window (W = 200, S = 50), budget (0.1%) and bootstrap (20,000 windows, GPD tail)
  were fixed before calibration.
- **Calibration:** thresholds are calibrated on the calibration split only and
  frozen before evaluation.
- **Sensitivity:** calibration size and W are varied after the fact, as
  sensitivity analyses (Appendix L).

**C3. Did you report descriptive statistics about your results?**
Yes.

- **Intervals and streams:** every detection and false-alarm rate has a Wilson 95%
  interval, from 200 replayed streams per condition (§4, Tables 3–5).
- **Paired tests:** comparisons against baselines use exact McNemar tests (§5).
- **Single seed vs ranges:** the main results are from one frozen calibration
  seed. Ranges over ten calibration seeds and ten replay seeds are in Appendix B,
  and intervals from 50 question half-samples in Appendix K.
- **Splits:** the calibration, held-out and fault splits are separate. Held-out
  false alarms are reported in every main table.

**C4. If you used existing packages, did you report the implementation, model, and parameter settings used?**
Yes. Appendix J:

- **Software:** Ollama 0.40.3; Python 3.11 with NumPy 2.4.6, SciPy 1.17.1,
  pandas 3.0.5, PyArrow 25.0.1 and Hugging Face Datasets 5.0.1. The supplementary
  lock file pins every dependency.
- **Models:** all local models are Q4_K_M builds pinned by digest; the embedding
  model is Nomic Embed v1.5.
- **Changes:** no package was modified.

## D. Did you use human annotators or research with human participants?

No. Correctness is scored automatically against each dataset's own answers. No
annotators or participants were recruited, so D1–D5 do not apply.

Optional note, **VERIFY** against the WildChat paper (Zhao et al., 2024) before
using: the real-traffic prompts come from WildChat, whose creators collected the
conversations from users who agreed to their release for research. Our use and
filtering are in B4.

## E. Did you use AI assistants in your research, coding, or writing?

**TEAM DECISION. Answer truthfully.** ACL's publication ethics policy requires
disclosing AI assistance beyond minor language polishing, and an AI tool cannot be
an author.

If AI tools were used for coding or writing, a suitable answer is:

> Yes. We used an AI coding and writing assistant to help implement and test the
> analysis code and to edit drafts of the text. All experimental design, analysis
> decisions, and claims are the authors', and every reported number was checked
> against the result files.

The paper should then carry a short acknowledgement of the same use. For the
anonymous submission it can go at the end of the Ethics Statement.
