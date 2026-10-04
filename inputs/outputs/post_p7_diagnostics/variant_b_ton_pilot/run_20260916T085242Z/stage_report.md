# ToN Variant B mini-pilot

It’s all about the **No-go**. Stop when it’s done.

## Input, sample and threshold registration

ToN/UNSW Raw cache using only freeze RF-All49; There is no weight training, Platt, probability or features change. According to Prof. reply, Section 4 and the implementation program within the project. Both the cache and the labels are aligned with the existing availability rank, with the original line number being completely unrepeated, and the endpoints of the window aligning with the manifest. Complete summary of uncalculated or verified documents.

Scanning the 4,020 points in rank order, collecting Benign=200, Attack=200; Stop rank=4019 (included), next rank=4020, original =4042. No replicated C1/C2 samples.

Threshold for registration time: 2026-09-16T09:04:40.319330+00:00. The 200 non-conformity score for each of the two categories is elevated to the 181 sequence; ε=0.10. Evaluation of additional uploaded registration documents.

| Thresholds | Numerical values |
| --- | --- |
| q_Benign | 0.16166667640209198 |
| q_Attack | 0.9805555418133736 |
| threshold_sum | 1.1422222182154655 |
| min_p_Benign | 0.838333323597908 |
| min_p_Attack | 0.019444458186626434 |

Benign in concentrations p ≤ q_Benign; Attack Input Condition 1−p ≤ q_Attack , e.g. Input. min_p_Benign/min_p_Attack is a subset of the probabilities of each of the two classes. The set generates a non-readable labels rating; singleton is measured with the only labels error, Raw independent with p≥0.5.

The singleton and Raw labels record are different: 2,052,584. The threshold value and ≥1 can be expressed as two sets of elements.

## Scope and Complete Results

ToN [100000,27520000): 27,420,000 bar, 2,742 original heldout 10k window; The entire start-up period and tail 260 items are excluded. UNSW [2130000,2365424): 235,424 clause, retaining 5,424 clause tail, using the same ToN threshold.

| Indicators | Counting / molecules | The dividend | Results |
| --- | --- | --- | --- |
| Singleton {Benign} rate | 2218519 | 27420000 | 8.090879% |
| Singleton {Attack} rate | 10913123 | 27420000 | 39.799865% |
| Empty-set rate | 0 | 27420000 | 0.000000% |
| Two-label rate | 14288358 | 27420000 | 52.109256% |
| Error among {Benign} singletons | 392555 | 2218519 | 17.694462% |
| Error among {Attack} singletons | 2526008 | 10913123 | 23.146518% |
| Correctly accepted attacks TP | 8387115 | — | 8387115 |
| TP / all true attacks | 8387115 | 10712072 | 78.295917% |
| Total acceptance | 13131642 | 27420000 | 47.890744% |
| Overall singleton error | 2918563 | 13131642 | 22.225423% |
| Raw full-acceptance error | 4361537 | 27420000 | 15.906408% |
| UNSW Benign class coverage | 215384 | 215407 | 99.989323% |
| UNSW Attack class coverage | 20017 | 20017 | 100.000000% |

Raw − singleton error rate: -6.3190154104736305 percentage points.

## Four descriptive statistics

| Indicators | 1 | 2 | 3 | 4 |
| --- | --- | --- | --- | --- |
| rank range | [100000,6960000) | [6960000,13820000) | [13820000,20670000) | [20670000,27520000) |
| Number of windows | 686 | 686 | 685 | 685 |
| Singleton {Benign} rate | 811610/6860000 = 11.831050% | 970850/6860000 = 14.152332% | 396624/6850000 = 5.790131% | 39435/6850000 = 0.575693% |
| Singleton {Attack} rate | 1501705/6860000 = 21.890743% | 2383247/6860000 = 34.741210% | 3225989/6850000 = 47.094730% | 3802182/6850000 = 55.506307% |
| Empty-set rate | 0/6860000 = 0.000000% | 0/6860000 = 0.000000% | 0/6850000 = 0.000000% | 0/6850000 = 0.000000% |
| Two-label rate | 4546685/6860000 = 66.278207% | 3505903/6860000 = 51.106458% | 3227387/6850000 = 47.115139% | 3008383/6850000 = 43.918000% |
| Error among {Benign} singletons | 10053/811610 = 1.238649% | 300135/970850 = 30.914662% | 60258/396624 = 15.192727% | 22109/39435 = 56.064410% |
| Error among {Attack} singletons | 1161209/1501705 = 77.326039% | 642284/2383247 = 26.949955% | 187370/3225989 = 5.808141% | 535145/3802182 = 14.074681% |
| Correctly accepted attacks TP | 340496 | 1740963 | 3038619 | 3267037 |
| TP / all true attacks | 340496/609424 = 55.871774% | 1740963/2658838 = 65.478341% | 3038619/3200566 = 94.940051% | 3267037/4243244 = 76.993852% |
| Total acceptance | 2313315/6860000 = 33.721793% | 3354097/6860000 = 48.893542% | 3622613/6850000 = 52.884861% | 3841617/6850000 = 56.082000% |
| Overall singleton error | 1171262/2313315 = 50.631323% | 942419/3354097 = 28.097548% | 247628/3622613 = 6.835618% | 557254/3841617 = 14.505715% |
| Raw full-acceptance error | 987676/6860000 = 14.397609% | 1318855/6860000 = 19.225292% | 292202/6850000 = 4.265723% | 1762804/6850000 = 25.734365% |
| Total TP | 340496/8387115 = 4.059751% | 1740963/8387115 = 20.757591% | 3038619/8387115 = 36.229609% | 3267037/8387115 = 38.953049% |
| Raw − singleton ( percentage points ) | -36.233714117641775 | -8.872256021264297 | -2.5698957504886564 | 11.228650312615754 |

## Pairing the bootstrap block

One block of continuous 10 window, one block of 275 and one block of 2 window; Each copy has a pulled back 275 block, a 2,000 copy; PCG64 seed 20260915. The same number of synchronous polymers of Raw and singleton molecule dividers is used to calculate the risk and the difference, not the average window risk. 95% percentile range (linear)The interval is conditioned on the calibration sample, threshold value and model of the 400 fixed in this round, and does not contain the uncertainty of re-sampling the calibration; Four paragraphs only describe.

| Indicators | Estimated | 95% lower boundary | 95% boundary | effective copy | Unit |
| --- | --- | --- | --- | --- | --- |
| raw_error_rate | 0.15906407731582786 | 0.1364052218181818 | 0.18370938149491642 | 2000 | fraction |
| singleton_error_rate | 0.22225423142056416 | 0.19195072450335202 | 0.2542838966980304 | 2000 | fraction |
| raw_minus_singleton_pp | -6.3190154104736305 | -9.54806037454053 | -3.1024721403548137 | 2000 | percentage_points |

## Inspection and Delivery

validation.json all PASS: 200 for each class and sample is unique, with no intersection with evaluation; Four classes of set counts, singleton confusion counts and four combined counts; In addition, the whole vector is clustered in a independent check-in window. Equivalent, non-asymmetric thresholds, vacuum clusters, binary elements, zero acceptance and singleton are different from Raw through border checks. For each molecule of the ratio, the denominator is CSV; The difference value also preserves the precise integer molecular denominator; The prime factor is NA.

## Gate determined

Decision: **No-go**.

- No-go 1:: TP=0:: not established; TP=8,387,115.
- No-go 2: singleton error rate ≥ and range Raw error rate: valid; singleton=22.225423%,Raw=15.906408%.
- Go 1: There are at least two phases of TP: satisfy quantitative conditions (4/4 paragraph). The four contributions are as follows; Whether or not the numerical threshold is mainly driven by a single paragraph.
- Go 2:: materially lower:: not satisfied; singleton risk did not decrease.

| The time period | TP | Total TP | TP / Real attack | singleton error rate | Raw error rate |
| --- | --- | --- | --- | --- | --- |
| 1 | 340496 | 4.059751% | 55.871774% | 50.631323% | 14.397609% |
| 2 | 1740963 | 20.757591% | 65.478341% | 28.097548% | 19.225292% |
| 3 | 3038619 | 36.229609% | 94.940051% | 6.835618% | 4.265723% |
| 4 | 3267037 | 38.953049% | 76.993852% | 14.505715% | 25.734365% |

Maximum single contribution: 4, 38.953049%. In terms of the contribution distribution, the detection is not primarily driven by a single time period: 3 and 4 sections contribute 36.229609% and 38.953049% respectively, two sections total 75.182658%; However, the time distribution is clearly biased towards the back half, and only the singleton error rate in the 4 segment is lower than the Raw. Overall, No-go has been explicitly triggered by risk conditions, without relying on additional consistency thresholds.

This round has been stopped, regardless of the verdict. No adjustment of sample, threshold, start time or ε, no Week 1 initiation, and no instruction messages.


The output directory: `/home/karri/projects/NIDSProject/outputs/post_p7_diagnostics/variant_b_ton_pilot/run_20260916T085242Z`. the script `pilot.py`, reproduce the command `commands.sh` and execute the log `prepare.log` / `evaluate.log`; The scripts must be copied to the new independent empty directory and the original directory must not be covered.
