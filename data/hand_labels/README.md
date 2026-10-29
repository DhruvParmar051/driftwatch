# Hand labels for the automatic scoring check

Pre-registration: `src/config/hand_label_study.yaml`. Sample drawn on
29 Oct 2026 with `uv run python scripts/hand_labels.py build`.

Key (`key.json`, which holds the condition, model and checker verdict for every
item) sha256, recorded before any labelling:

    85c898ec399ade26b3508b21fa9803f9ad8616c59ec213c1abcf5af3dac17492

The key stays out of git, and nobody opens it, until all four sheets are filled.

## How to label your sheet

The easy way is the labelling page, which opens in your browser and saves each
label straight into your sheet as you go (you can stop and come back):

```bash
uv run python scripts/label_sheet.py <your name>
```

Keys: `1` correct, `0` wrong, `?` or `/` can't tell (each moves on), arrows to
browse, `n` for a note, `g` to flag the reference as wrong. Ctrl-C in the
terminal when you're done. It never shows the checker's verdict or which model
answered.

Or open `sheet_<your name>.csv` in a spreadsheet (in Excel, save as "CSV
UTF-8"; don't sort or delete rows). Either way: 50 rows, 10 of them also on
someone else's sheet, so label on your own, without comparing notes.

For each row, read the question, the reference answer and the model answer, and
fill in **label**:

- `1` the model's answer is correct, given the reference answer
- `0` it is wrong
- `?` you cannot tell

Judge meaning, not format. "Paris, the Trojan prince" is correct for "Paris";
"$18" or "18 dollars" is correct for 18; for gsm8k, judge the final answer the
model gives, not its working. For squad2, "unanswerable" means the passage
does not contain the answer, and the model should say so. For banking77, the
model's label must be the reference intent.

Use **note** for anything odd, and write `gold wrong` if the reference answer
itself looks wrong. Don't change any other column. Save as CSV, same file name.
It takes about 20-30 minutes.

When all four sheets are in:

```bash
uv run python scripts/hand_labels.py score
```

writes `results.json` and `results.md` here.
