import json
yes = {
 715: ("Here Sanchez is contradicting his own Justification Dogma", "reflexive"),
 716: ("A prison from which he could not escape. A prison of his own making.", "container"),
 734: ("she was serene and whole inside - good and evil, gone", "abstract"),
 737: ("How does rational and irrational converge?", "abstract"),
 794: ("Caught in a network, a paradox. Between two worlds, a nameless figure.", "abstract"),
}
# fix capitalization for 716 to match verbatim
yes[716] = ("a prison from which he could not escape. A prison of his own making.", "container")
out = []
for i in range(701, 801):
    if i in yes:
        out.append({"id": i, "verdict": "YES", "span": yes[i][0], "kind": yes[i][1]})
    else:
        out.append({"id": i, "verdict": "NO", "span": "", "kind": ""})
p = "/private/tmp/claude-502/-Users-rj416-Dropbox-Prof-Articles-TheoryMachines-agents-lacan/cdbe9c9e-a018-45bf-95e9-6bf81e96e908/scratchpad/opus/out_08.json"
json.dump({"judgements": out}, open(p, "w"), ensure_ascii=False)
print(sum(1 for o in out if o["verdict"]=="YES"), sum(1 for o in out if o["verdict"]=="NO"), len(out))
