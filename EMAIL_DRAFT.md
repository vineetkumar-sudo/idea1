# Request to the CADC+ authors for the sequence pairing table

**To:** mqtang@uwaterloo.ca  (Mei Qi Tang — first author, built the pairing)
**Cc:** k2czarne@uwaterloo.ca, sean.sedwards@uwaterloo.ca, c.huang@uwaterloo.ca
**Subject:** CADC+ — request for the snowy↔clear sequence pairing table

---

Dear Mei Qi Tang,

I'm working on a controlled evaluation of weather-robustness training strategies for
LiDAR 3D object detection, and CADC+ is central to the design — as far as I know it's
the only dataset that lets me separate the effect of snowfall from the effect of
location shift, rather than confounding the two.

I have the sequence listings from both wiselab.uwaterloo.ca/cadc/ and /cadc-clear/, and
the train/val split membership from the public Segments.ai projects. The one thing I
haven't been able to find is the actual one-to-one mapping between each CADC sequence
and its matched CADC-clear sequence. I've looked at the file servers, the per-sequence
3d_ann.json files, the cadc_devkit repository, the IV 2025 paper, and your thesis
(UWSpace 10012/21442). Section 4.5 confirms that "All sequences in CADC have been
matched one-to-one with a clear weather CADC++ sequence," and Figure 4.6 shows the
resulting coverage, but I couldn't locate the pair list itself.

Would you be willing to share it, in whatever form you already have — CSV, JSON, or
even a spreadsheet?

Two related questions, if they're quick to answer:

1. Section 4.5 notes that 21 of the 74 CADC sequences could not be matched spatially —
   15 were matched manually to an alternate but similar scene, and 6 on the type of road
   agents. Is there a per-pair record of which matching method was used? I'd like to
   report results separately for the spatially-matched pairs, since those are the ones
   where location is genuinely held constant.

2. Which CADC sequence is the one matched to two half-length CADC-clear sequences, and
   which sequence has the corrupted annotations that was excluded from matching?

I'm happy to cite the CADC+ paper and the thesis, and glad to share back anything
useful that comes out of the evaluation.

Thank you for building this — the paired design is exactly what this question needed.

Best regards,
Vineet Kumar
vineet.kumar@polymathai.co

---

## Notes before sending

- **Add your affiliation** under your name if you want it read as an academic request.
- Mei Qi Tang's thesis is dated **2024**, so `mqtang@uwaterloo.ca` may have lapsed.
  That's why Prof. Czarnecki (`k2czarne`, WISE Lab director, permanent) and Sean Sedwards
  are on Cc — one of them will reach a live mailbox. Don't drop the Cc.
- All four addresses are printed verbatim in the paper itself:
  `{mqtang, sean.sedwards, c.huang, k2czarne}@uwaterloo.ca`
- Citing §4.5 back to them is deliberate: it shows you've read the thesis, not just the
  paper, and makes the request specific enough to be quick to answer.
- If nobody replies in ~10 days, follow up via the WISE Lab contacts page:
  https://uwaterloo.ca/waterloo-intelligent-systems-engineering-lab/contacts
