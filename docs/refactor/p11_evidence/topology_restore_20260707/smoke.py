import json, urllib.request, urllib.error, concurrent.futures, time
GW="http://10.99.21.145/v1/completions"
MODELS=["dsqwen-7b","dsllama-8b","dsqwen-14b"]
def one(model):
    payload=json.dumps({"model":model,"prompt":"Return exactly one short sentence about capacity planning.","max_tokens":32,"temperature":0}).encode()
    req=urllib.request.Request(GW,data=payload,method="POST")
    req.add_header("Content-Type","application/json"); req.add_header("model",model)
    try:
        with urllib.request.urlopen(req,timeout=45) as r:
            d=json.loads(r.read()); 
            return bool(d.get("choices"))
    except Exception as e:
        return repr(e)[:120]
for m in MODELS:
    ok=0; errs=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        for res in ex.map(lambda _: one(m), range(20)):
            if res is True: ok+=1
            else: errs.append(res)
    print(f"{m}: {ok}/20 ok"+ (f" errs={errs[:3]}" if errs else ""))
