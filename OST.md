# Inferring team possession from tactical tracking

**Team OST · More than a Hack 2027 · Phase 1**

## Inputs and observations

The system assigns one label to each tactical-video frame. It first estimates when the ball is dead, then determines which team controls it during live play. A separate goal rule can extend a stoppage through the ensuing centre pass. Separating these decisions matters because a player can stand next to the ball throughout a stoppage.

| Input | Use in the released pipeline |
| --- | --- |
| Tactical ATD text | Frame-aligned player and ball positions, including missing observations. |
| Tactical ATD metadata | Track-to-team assignments, goalkeeper candidates and opening attack directions. |
| Tactical Smart Tagging | Approximate goal intervals and after-goal kick-off intervals. The release corroborates them with tracking. |

**Frame alignment.** The reader verifies the one-based row numbers in the tactical tracking file. It maps row 1 to video frame 0, scales normalized coordinates to a 105.3 × 68 m pitch, and processes the official half intervals separately at 25 Hz. Frames outside those intervals receive `X`. Smart Tagging times already follow the tactical video clock.

**Ball gaps.** At the last observed in-bounds ball position, select the three nearest visible player slots. Estimate ball displacement from their subsequent displacements, weighting each by the inverse of its initial distance plus one metre. Renormalize over the selected slots still visible in each frame. Leave outward line-crossing gaps unresolved. Raw observations have reliability 0.85 and reconstructed observations 0.20. Only raw observations can establish control.

**Player tracks and goalkeepers.** Mask outfield samples beyond the pitch tolerance, jumps above 12 m/s and uncertain identity associations. Infer each team’s defending side from its opening outfield formation. Assign the available goalkeeper candidates by physical half. If two occupy the same half, select the one with higher smoothed activity. Near ties retain the previous source where possible. This prevents an unreliable goalkeeper slot from giving a pass to the wrong team.

**Training and assets.** This is an offline rule-based method with zero labelled training examples, no learned weights and no external event feed. The executable consumes the three assets above. Video pixels, broadcast tracking, homographies and public possession labels are outside its inference inputs.

## Dead ball and team control

**Stoppage candidates.** The system combines visible line exits, outward tracking loss near a line, and collective slowing. For example, a ball disappearing within 2 m of a boundary must be moving outward and projected to reach the line within 0.75 s. The pooled motion rule requires at least 3 s of stable outfield activity at or below 1.5 m/s, following a drop of at least 0.5 m/s. This captures some stoppages even when the ball is missing.

**Restarts.** Refine candidate endpoints using a stationary ball followed by observed departure, nearby players and pitch geometry. A corner arc, goal area, penalty spot or centre spot provides a location cue. Collective reactivation helps identify set-piece restarts. Line-exit and supported set-piece intervals receive protection from the slow-build-up rule.

**Live team.** For each side, compare ball–player distance and relative velocity using the cleaned tracks. Typical control requires raw ball support, distance within 2 m, relative speed at most 2.8 m/s, and sufficient separation from the opposing candidate. Control can be confirmed by 0.20 s of sustained evidence or an exceptionally clear close encounter. A touch is weaker evidence. Last-confirmed team memory and a Viterbi decoder penalize unsupported H/A switches across a loose ball, deflection or challenge. A D interval resets that live sequence.

**Slow build-up.** A slowing team can still be passing. Two linked same-team transfers between distinct player slots can restore live play across a proposed D onset. Each transfer must cover at least 3 m within 4 s. Together they must cover at least 10 m within 12 s, reverse direction, have sufficient raw observations and lie predominantly in the team’s defensive part of the pitch. The rule restores only the supported overlap. A final pass merges live gaps of at most 0.5 s between two D intervals in the same half. A confirmed goal-restart reception takes priority over that merge.

## Goals, restarts and reproduction

**Start D.** Pair an automatic `GOALS` interval with at least four distinct outfield tracks from one team converging on the same corner. Runners must be within 25 m of the corner, moving at 2–12 m/s with a cornerward component of at least 1.5 m/s and a heading cosine of at least 0.7. Require 0.6 s of group support and agreement with the tagged team when known. Start D at the first supported convergence frame. This estimates the start of the celebration.

**End D.** Look for the conceding team at the centre. A stationary preparation within 2.5 m of centre for 0.4 s can establish the origin. When a linked automatic `Kick Off (After Goal)` interval is available, the ball may already be moving when tracking resumes. After at least 0.4 s without a valid ball, accept a new raw track starting within the centre region and within 2 m of a conceding player who is closer than the opposition. In both cases, follow at least 2 m of observed departure. Keep D through the pass.

Release D on the **first frame of confirmed receiving control** by a different conceding-team player. The receiver must be within 2 m of the ball, at least 3 m from centre, and supported for 0.20 s within 6 s of the observed departure. Require distinct, continuous player tracks, an unbroken ball path and no intervening opponent control. Protect that reception from gap merging. If the association fails, the goal hold continues to the next corroborated goal or the end of the half.

**Limitations.** A quiet celebration can miss the corner gate. A celebration can start after the ball actually crosses the line. Missing receiver observations can prolong D, while using reception as the live boundary can lag the referee’s restart. Ball height is unobserved, and track swaps or ties can still distort team ownership. The private full-match accuracy is unknown.

**Reproduction.** Install the pinned versions in `requirements.txt` with Python 3.11.18 and place the licensed bundle in `data/`. From the repository root run the following commands. The first verifies the exact CSV bytes against the submitted file’s SHA-256. The second runs the organizer’s format checks.

```sh
python predict.py --data-dir data --output outputs/OST.csv --verify
python evaluation/validate_submission.py outputs/OST.csv
```

[Public source and installation instructions](https://github.com/janginho12/mtah-OST) · [Organizer’s possession definitions](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/CHALLENGE.md)
