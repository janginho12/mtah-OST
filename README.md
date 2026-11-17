# mtah-OST

Team OST’s entry for More than a Hack 2027, Phase 1. A deterministic CPU pipeline labels every tactical-video frame as home possession (`H`), away possession (`A`), dead ball (`D`), or outside the active halves (`X`). For a corroborated goal, the first controlled reception of the conceding team’s centre pass ends the goal stoppage.

## Reproduce the submission

Use 64-bit Python 3.11.18 and the versions in `requirements.txt`. The competition bundle may be stored outside the checkout. The directory given to `--data-dir` must contain `tactical/`. [Required input files](data/README.md)

```sh
git clone https://github.com/janginho12/mtah-OST.git
cd mtah-OST
python -m venv .venv
```

Activate the environment on Windows PowerShell with `.\.venv\Scripts\Activate.ps1`. On macOS or Linux use `source .venv/bin/activate`. Then run these commands from the repository root.

```sh
python -m pip install -r requirements.txt
python predict.py --data-dir "/path/to/data" --output outputs/OST.csv --verify
python evaluation/validate_submission.py outputs/OST.csv
```

With the bundle in `./data`, simply run `python predict.py --verify`. The entry point also accepts the `MTAH_DATA_DIR` environment variable. It reads the tactical tracking text, metadata XML and Smart Tagging XML. All intermediate arrays are computed locally. There are no weights to fetch, no model-fitting step and no inference service.

The generated UTF-8 CSV uses LF line endings, the header `frame,label`, and 199,319 ordered records. `--verify` checks the **exact file bytes** against the submitted file’s SHA-256. The separate official validator checks the frame and label contract. Use a fresh output path when comparing versions.

## Method and checks

- [Short technical document](OST.md) — inputs, decision criteria, assumptions, limitations and reproduction.
- `ost/` contains goalkeeper assignment, pass-chain refinement and goal/restart decisions. `starter/` contains the tracking, stoppage, control and sequence-decoding code used by the entry point.
- `python -m unittest discover -s tests` exercises the centre-pass recovery with synthetic coordinates, including missing preparation frames, an opponent’s intervention and midfield crossings.

The repository keeps the upstream license notice and official validator. Licensed match data, visualizations and submission CSVs stay outside the public source tree. [Organizer repository](https://github.com/metrica-sports/more-than-a-hack-2027) · [Rules](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/RULES.md) · [Submission specification](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/SUBMISSION.md)
