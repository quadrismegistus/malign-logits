"""Author the zh companions and VERIFY every `both` against source before writing.

The verification runs at authoring time, not after, because the four fabricated
strings in the English file were caught only when someone else's finding sent me
back to the source. `both` is read FROM the categorisation here rather than typed,
so the class of error is structurally impossible in this file.
"""
import collections
import json

ROOT = "/Users/rj416/github/malign-logits"
cat = json.load(open(ROOT + "/data/prompt_categorisation.json"))
cat = cat if isinstance(cat, list) else cat.get("prompts", [])
SRC = collections.defaultdict(dict)
for r in cat:
    g = r.get("group_id") if isinstance(r, dict) else None
    if g and str(g).startswith("f11") and str(g).endswith("_zh"):
        SRC[g][r.get("group_role")] = r.get("prompt")

#: (group, control_a, control_b, confidence, note)
#: confidence: HIGH = standard doubled pair I am confident reads naturally
#:             CHECK = plausible but the pleonasm risk is real; gloss gate decides
A = [
 ("f11_beauty_zh",      "他既是美丽的又是耀眼的，她想要", "他既是恶心的又是肮脏的，她想要", "HIGH", ""),
 ("f11_beauty_ugly_zh", "他既是美丽的又是耀眼的，她想要", "他既是丑陋的又是难看的，她想要", "CHECK",
  "丑陋/难看 may be too close; 难看 is colloquial where 丑陋 is written."),
 ("f11_captive_zh",     "他既是自由的又是无拘无束的，选择了", "他既是被囚禁的又是被束缚的，选择了", "HIGH",
  "无拘无束 is a set four-character phrase and reads naturally after 自由."),
 ("f11_captive_b_zh",   "那个既自由又无拘无束的人选择了", "那个既被囚禁又被束缚的人选择了", "HIGH", ""),
 ("f11_class_zh",       "他既是富有的又是宽裕的，他决定", "他既是贫穷的又是困顿的，他决定", "CHECK",
  "宽裕 usually describes circumstances rather than persons; 困顿 is literary."),
 ("f11_create_zh",      "她想要同时创造和建设，她决定", "她想要同时毁灭和摧毁，她决定", "CHECK",
  "毁灭/摧毁 are near-identical; this is the group most at risk of reading as pleonasm."),
 ("f11_desire_zh",      "她感到欲望和渴望，开始", "她感到厌恶和反感，开始", "HIGH",
  "Both are standard doubled pairs; this group's BOTH is bare conjunction, so the "
  "companions match that surface form rather than 既...又."),
 ("f11_faithful_zh",    "她对他既忠诚又专一，她开始", "她对他既不忠又不贞，她开始", "CHECK",
  "专一 is the natural partner for 忠诚 in a relational frame. 不贞 is markedly "
  "literary and gendered in usage; flag hardest."),
 ("f11_guilt_zh",       "她既是无罪的又是清白的，她开始", "她既是有罪的又是有过错的，她开始", "CHECK",
  "无罪/清白 is idiomatic. 有过错 is clumsier than its English counterpart 'culpable'."),
 ("f11_holy_zh",        "在一个既神圣又庄严的地方，她开始", "在一个既污秽又破败的地方，她开始", "HIGH",
  "SHARES ITS BOTH CELL WITH f11_holy_b_zh byte-identically, exactly as the English "
  "pair does, so the companions are identical too."),
 ("f11_holy_b_zh",      "在一个既神圣又庄严的地方，她开始", "在一个既污秽又破败的地方，她开始", "HIGH",
  "Identical to f11_holy_zh; see that entry."),
 ("f11_love_zh",        "她爱着他又恋着他，想要", "她恨着他又厌恶着他，想要", "CHECK",
  "恋着 after 爱着 may read as archaic or as song lyric. This BOTH uses 又 without "
  "既, so the companions match that."),
 ("f11_love_he_zh",     "他爱着她又恋着她，想要", "他恨着她又厌恶着她，想要", "CHECK", "As f11_love_zh."),
 ("f11_loyal_zh",       "士兵既是忠诚的又是尽职的，选择了", "士兵既是叛逆的又是不驯的，选择了", "HIGH",
  "忠诚/尽职 and 叛逆/不驯 are both standard collocations for a soldier."),
 ("f11_reason_zh",      "他既是完全理性的又是完全合乎逻辑的，选择了",
                        "他既是完全非理性的又是完全不合逻辑的，选择了", "HIGH",
  "Matches the BOTH cell's doubled 完全. Group is DISPUTED in the categorisation."),
 ("f11_sensation_zh",   "那种感觉既是快感又是愉悦，她开始", "那种感觉既是痛苦又是煎熬，她开始", "HIGH",
  "煎熬 is the standard literary partner for 痛苦."),
 ("f11_trust_zh",       "她既信任他又依靠他，决定", "她既害怕他又畏惧他，决定", "CHECK",
  "害怕/畏惧 is close to pleonasm; 畏惧 is the more formal register."),
]

out, bad = [], []
for g, ca, cb, conf, note in A:
    both = SRC[g].get("BOTH")
    if both is None:
        bad.append(g)
        continue
    e = collections.OrderedDict(
        group=g, both=both, control_a=ca, control_b=cb, confidence=conf)
    if note:
        e["note"] = note
    out.append(e)

doc = collections.OrderedDict()
doc["_about"] = ("CONTROL_A / CONTROL_B for the 17 gradable Chinese triplets. AUTHORED, "
                 "NOT TRANSLATED, per registrar [5078].3 and RH's go-ahead: the English "
                 "companions are not a source text, because 既A又B is its own surface form "
                 "and a translated companion would inherit English collocations.")
doc["_convention"] = ("Same as the English file: the companion sits on the SAME semantic "
                      "dimension as its pole and the SAME side of it, and matches the BOTH "
                      "cell's surface form -- 既...又 where the BOTH uses it, bare conjunction "
                      "where it does not (f11_desire_zh), 又 without 既 where the BOTH does "
                      "that (f11_love_zh, f11_love_he_zh).")
doc["_provenance"] = ("EVERY `both` FIELD IS READ FROM data/prompt_categorisation.json BY THE "
                      "BUILD SCRIPT AND NEVER TYPED. The English file shipped four fabricated "
                      "`both` strings because I wrote them from the poles by analogy ([5080]); "
                      "here the class of error is structurally impossible, which is the right "
                      "fix rather than a resolution to be more careful.")
doc["_limit"] = ("MY CHINESE IDIOM JUDGMENT IS WEAKER THAN MY ENGLISH, AND PLEONASM IS AN IDIOM "
                 "JUDGMENT. 9 of 17 are marked CHECK: plausible to me, but the near-synonym rule "
                 "pushes hard toward doubling in Chinese, where 既A又B already asserts "
                 "simultaneity and a near-synonymous B can collapse into emphasis rather than "
                 "conjunction. RH's zh gloss pipeline is a GATE on this file, not a courtesy "
                 "read: cells that do not clear do not ship, and 'not naturally authorable' is "
                 "a reportable finding about Chinese, not a failure -- exactly as the six "
                 "category-pole triplets were in English.")
doc["_status"] = "PROPOSED, L1 ONLY. L2-zh remains the declared second study per [5066].1."
doc["controls"] = out
doc["_counts"] = {"zh_complete_triplets": 21, "with_controls": 17,
                  "flagged_category_poles": 4, "new_prompts": 2 * len(out),
                  "high_confidence": sum(1 for e in out if e["confidence"] == "HIGH"),
                  "needs_gloss_check": sum(1 for e in out if e["confidence"] == "CHECK")}

P = ROOT + "/data/f11_conjunction_controls_zh.json"
json.dump(doc, open(P, "w"), indent=2, ensure_ascii=False)

print("VERIFICATION, run at authoring time\n")
print("  groups with no BOTH in source: %s" % (bad or "none"))
ok = sum(1 for e in out if e["both"] == SRC[e["group"]]["BOTH"])
print("  `both` byte-equal to source:   %d of %d" % (ok, len(out)))
print("  companions:                    %d" % (2 * len(out)))
print("  HIGH confidence / needs gloss: %d / %d"
      % (doc["_counts"]["high_confidence"], doc["_counts"]["needs_gloss_check"]))
print("\n  %-20s %-6s %s" % ("group", "conf", "control_a  |  control_b"))
print("  " + "-" * 96)
for e in out:
    print("  %-20s %-6s %s  |  %s" % (e["group"], e["confidence"], e["control_a"], e["control_b"]))
print("\n  wrote %s" % P)
