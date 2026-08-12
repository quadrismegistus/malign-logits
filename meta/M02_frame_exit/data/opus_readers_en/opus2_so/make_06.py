import json

yes = {
    10521: ("the thin barrier between reason and fantasy that keeps her captive within", "container"),
    10580: ("It was a double-edged sword", "oxymoron"),
    10585: ("protect her from herself", "reflexive"),
}

out = []
for i in range(10500, 10600):
    if i in yes:
        span, kind = yes[i]
        out.append({"id": i, "verdict": "YES", "span": span, "kind": kind})
    else:
        out.append({"id": i, "verdict": "NO", "span": "", "kind": ""})

path = "/private/tmp/claude-502/-Users-rj416-Dropbox-Prof-Articles-TheoryMachines-agents-lacan/cdbe9c9e-a018-45bf-95e9-6bf81e96e908/scratchpad/opus2_so/out_06.json"
with open(path, "w") as f:
    json.dump({"judgements": out}, f, ensure_ascii=False)
print(len(out), sum(1 for o in out if o["verdict"] == "YES"))
