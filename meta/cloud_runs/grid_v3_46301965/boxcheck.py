import importlib.util as ilu, transformers
sp=ilu.spec_from_file_location("tc","/workspace/twp_cloud.py")
tc=ilu.module_from_spec(sp); sp.loader.exec_module(tc)
print("transformers", transformers.__version__)
PROBES={"english":"She was so angry she wanted to",
        "contract":"He didn't know what she'd say",
        "numeric":"The salary is $100,000 and rising",
        "cjk":"她非常生气，想要","mixed":"The word 她 means she"}
for M in ["deepseek-ai/deepseek-llm-7b-base","allenai/OLMo-2-0425-1B",
          "Qwen/Qwen2.5-0.5B","LLM360/Amber"]:
    tok,loader=tc.load_tokenizer(M)
    bad=[]
    for k,p in PROBES.items():
        ids=tok.encode(p, add_special_tokens=False)
        if " ".join(tok.decode(ids).split()) != " ".join(p.split()): bad.append(k)
    verdict = "CORRUPT:"+",".join(bad) if bad else "clean"
    name = M.split("/")[-1][:26]
    print("%-28s loader=%-22s %s" % (name, loader, verdict))
