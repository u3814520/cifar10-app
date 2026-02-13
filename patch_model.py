import json
from pathlib import Path

p = Path("model/model.json")
j = json.loads(p.read_text(encoding="utf-8"))

# 1) Fix InputLayer: batch_shape -> batch_input_shape
layers = (
    j.get("modelTopology", {})
     .get("model_config", {})
     .get("config", {})
     .get("layers", [])
)
for L in layers:
    if L.get("class_name") == "InputLayer":
        cfg = L.get("config", {})
        if "batch_shape" in cfg and "batch_input_shape" not in cfg:
            cfg["batch_input_shape"] = cfg.pop("batch_shape")

# 2) Remove "sequential/" prefix in weightsManifest names
for m in j.get("weightsManifest", []):
    for w in m.get("weights", []):
        name = w.get("name", "")
        if name.startswith("sequential/"):
            w["name"] = name[len("sequential/"):]

p.write_text(json.dumps(j, ensure_ascii=False), encoding="utf-8")

s = p.read_text(encoding="utf-8")
print("patched ✅")
print("sequential/ count after =", s.count("sequential/"))
print("has batch_shape =", '"batch_shape"' in s)
