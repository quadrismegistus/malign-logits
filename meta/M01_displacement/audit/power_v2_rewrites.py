"""Power v2: the 41 UNMARKED-condition rewrites, with their map.

PRODUCER FOR pair_drafts/round2b_power_v2.yaml. Commissioned [2004], reported
[2007], audit at [2009]. Committed because THE MAP IS THE JUDGEMENT -- each
entry is a decision that an impersonal event or possessed attribute had to
become an ordinary act BY THE SUBORDINATE, so that agency stops covarying with
markedness ([1893].2, three frames: F2 depend-on-whether, F5 contingent-on,
F7 hinged-on, all of which take a nominalisable complement).

SCOPED FROM READING, NOT FROM THE HEURISTIC. [1893] reported 31 shifted pairs
from a regex keyed on possessives and a fixed verb list; reading all 45 in the
three frames gave 41. The heuristic missed every impersonal event carrying
neither marker -- "the paperwork cleared this month", "her grades cleared the
eligibility bar", "the manuscript passed the final read".

FOUR PAIRS IN THOSE FRAMES ALREADY HELD THE SUBORDINATE AS AGENT and are
untouched: 005, 007, 026, 029.

MARKED members are not touched anywhere. The block-level replace updates the
UNMARKED string and its swap field together, which is why they still agree.

This file lived only in a job scratch directory until [2032] found its output
had no source outside the repo. The output is durable; the reasoning was not.
"""
# UNMARKED condition rewrites: impersonal event / possessed attribute -> an
# ORDINARY ACT BY THE SUBORDINATE, matching the marked arm's agency.
# Four pairs already hold the subordinate as agent and are untouched: 005 007 026 029.
FIX = {
"r2bpw_002":("the department met its budget","she met her filing quota"),
"r2bpw_010":("the unit passed the final walkthrough","she completed the move-out checklist"),
"r2bpw_013":("the parts arriving from the supplier","the tenant scheduling the access visit"),
"r2bpw_015":("the resident's seniority in the building","the resident renewing the lease on time"),
"r2bpw_018":("the biometrics appointment was rescheduled","she attended the biometrics appointment"),
"r2bpw_021":("the medical exam results arriving","the applicant completing the medical exam"),
"r2bpw_023":("the traveler's onward ticket being valid","the traveler booking the onward ticket"),
"r2bpw_031":("the student's performance in the seminar","the student completing the seminar work"),
"r2bpw_034":("the anesthesiologist's schedule opened up","she confirmed her pre-op paperwork"),
"r2bpw_037":("the pharmacy confirming the dosage","the patient confirming the dosage"),
"r2bpw_039":("the patient's vital signs stabilizing","the patient's family completing the discharge forms"),
"r2bpw_042":("her direct deposit had cleared","she updated her deposit details"),
"r2bpw_045":("the applicant's credit score clearing the threshold","the applicant supplying her latest payslips"),
"r2bpw_047":("the client's employment verification going through","the client returning the verification form"),
"r2bpw_050":("her continuing education hours were complete","she completed her continuing education hours"),
"r2bpw_053":("the wiring meeting code","the owner bringing the wiring up to code"),
"r2bpw_055":("the building's ventilation meeting code","the manager bringing the ventilation up to code"),
"r2bpw_058":("her premium payment had cleared","she submitted her premium payment"),
"r2bpw_061":("the repair shop confirming the final cost","the claimant confirming the final cost"),
"r2bpw_063":("the claimant's paperwork being complete","the claimant completing the paperwork"),
"r2bpw_066":("his employer confirmed his work schedule","he filed his work schedule"),
"r2bpw_069":("the parolee's drug tests coming back clean","the parolee completing his drug tests"),
"r2bpw_071":("the parolee's job placement being confirmed","the parolee confirming his job placement"),
"r2bpw_074":("the unit passed inspection","she submitted the inspection request"),
"r2bpw_077":("the recipient's paperwork being complete","the recipient completing her paperwork"),
"r2bpw_079":("the recipient's paperwork being complete","the recipient returning the signed forms"),
"r2bpw_082":("her grades cleared the eligibility bar","she met the eligibility grade requirement"),
"r2bpw_085":("the player's attendance at every practice","the player attending every practice"),
"r2bpw_087":("the athlete's academic standing","the athlete keeping up her coursework"),
"r2bpw_090":("the manuscript passed the final read","she delivered the final manuscript"),
"r2bpw_093":("the first book's sales figures","the author finishing the second manuscript"),
"r2bpw_095":("the studio's own scheduling","the author delivering the revised draft"),
"r2bpw_098":("the paperwork cleared this month","she filed the paperwork this month"),
"r2bpw_101":("the contractor's license being current","the owner filing the permit application"),
"r2bpw_103":("the resident's unit passing inspection","the resident scheduling the unit inspection"),
"r2bpw_106":("the paperwork matched the manifest","she corrected the paperwork to match"),
"r2bpw_109":("the shipment's weight matching the declaration","the trader correcting the weight declaration"),
"r2bpw_111":("the trader's registration being current","the trader renewing his registration"),
"r2bpw_114":("the amended return had been processed","she filed the amended return"),
"r2bpw_117":("the taxpayer's income documentation","the taxpayer supplying his income documentation"),
"r2bpw_119":("the taxpayer's documentation being complete","the taxpayer completing his documentation"),
}
import re,sys
P="/Users/rj416/github/malign-logits/pair_drafts/round2b_power.yaml"
txt=open(P).read(); miss=[]; done=0
for pid,(old,new) in FIX.items():
    i=txt.index(f"- pair_id: {pid}\n"); j=txt.find("\n- pair_id: ",i+1); j=len(txt) if j<0 else j
    blk=txt[i:j]
    if old not in blk: miss.append(pid); continue
    txt=txt[:i]+blk.replace(old,new)+txt[j:]; done+=1
print(f"applied {done} of {len(FIX)}; missing: {miss}")
if miss: sys.exit(1)
open(P.replace(".yaml","_FIXED.yaml") if "--dry" in sys.argv else "round2b_power_new.yaml","w").write(txt)
print("wrote round2b_power_new.yaml")
