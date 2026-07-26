#!/usr/bin/env python3
"""Merge the mechanical and judgment halves into findings frontmatter.

Mechanical half (this repo): date, data, scripts, families -- everything the
repo can prove, paths verified on disk.
Judgment half (TheoryMachines): status, grade, chapters, role, instruments,
superseded_by -- the audit record, which lives in the claims ledger.

Neither seat fills the other's fields. Anything missing from either half is
reported and the finding is SKIPPED rather than defaulted, because a defaulted
status is an audit claim nobody made.
"""
import json, os, re, sys

JUDGMENT = ("/Users/rj416/Dropbox/Prof/Articles/TheoryMachines/notes/"
            "frontmatter-judgment-fields.md")


def parse_judgment(path):
    out, cur = {}, None
    for line in open(path):
        m = re.match(r"^##\s+(F\d+)\s*$", line)
        if m:
            cur = m.group(1); out[cur] = {"_raw": []}
            continue
        if cur:
            out[cur]["_raw"].append(line.rstrip("\n"))
    for num, d in out.items():
        txt = "\n".join(d["_raw"])
        sg = re.search(r"status:\s*([\w-]+)\s*\|\s*grade:\s*([A-D])\s*\|\s*chapters:\s*\[([^\]]*)\]", txt)
        if sg:
            d["status"], d["grade"] = sg.group(1), sg.group(2)
            d["chapters"] = [c.strip() for c in sg.group(3).split(",") if c.strip()]
        ins = re.search(r"instruments:\s*\[([^\]]*)\]", txt)
        if ins:
            d["instruments"] = [c.strip() for c in ins.group(1).split(",") if c.strip()]
        sup = re.search(r"superseded_by:\s*(.+?)(?:\n- |\Z)", txt, re.S)
        if sup:
            v = " ".join(sup.group(1).split())
            d["superseded_by"] = None if v in ("n/a", "none", "-") else v
        role = re.search(r"role:\s*(.+?)(?:\n- \w+:|\Z)", txt, re.S)
        if role:
            d["role"] = " ".join(role.group(1).split())
    return out


def yaml_list(vals):
    return "[" + ", ".join(f'"{v}"' if any(c in v for c in ':,#"') else v for v in vals) + "]"


def main():
    mech = json.load(open("data/frontmatter_mechanical.json"))
    jud = parse_judgment(JUDGMENT)
    wrote, skipped = [], []
    for num in sorted(mech, key=lambda x: int(x[1:])):
        m, j = mech[num], jud.get(num)
        path = f"findings/{m['file']}"
        if not j or "status" not in j:
            skipped.append((num, "no judgment entry")); continue
        if j["status"] in ("rescoped", "retracted") and not j.get("superseded_by"):
            skipped.append((num, "rescoped/retracted without superseded_by")); continue
        body = open(path).read()
        if body.startswith("---"):
            skipped.append((num, "already has frontmatter")); continue
        # role is a controlled TYPE {finding, addendum, capstone, ledger}, not a
        # description -- both seats initially read it as prose. Desktop's prose
        # goes to `description`, which the linter does not constrain.
        fm = ["---", f"status: {j['status']}", f"grade: {j['grade']}", f"date: {m['date']}",
              "role: finding"]
        if j.get("role"):
            fm.append(f"description: {json.dumps(j['role'])}")
        if j.get("instruments"):
            fm.append(f"instruments: {yaml_list(j['instruments'])}")
        if m["families"]:
            fm.append(f"families: {yaml_list(m['families'])}")
        if j.get("chapters"):
            fm.append(f"chapters: {yaml_list(j['chapters'])}")
        if m["data"]:
            fm.append(f"data: {yaml_list(m['data'])}")
        if m["scripts"]:
            fm.append(f"scripts: {yaml_list(m['scripts'])}")
        if j.get("superseded_by"):
            fm.append(f"superseded_by: {json.dumps(j['superseded_by'])}")
        fm.append("---\n")
        open(path, "w").write("\n".join(fm) + body)
        wrote.append((num, j["status"], j["grade"], len(m["data"]), len(m["scripts"])))

    print(f"{'F':6s}{'status':16s}{'grade':7s}{'data':>5s}{'scr':>5s}")
    for num, st, gr, nd, ns in wrote:
        print(f"{num:6s}{st:16s}{gr:7s}{nd:>5d}{ns:>5d}")
    print(f"\nwrote {len(wrote)} | skipped {len(skipped)}")
    for num, why in skipped:
        print(f"   SKIP {num}: {why}")


if __name__ == "__main__":
    main()
