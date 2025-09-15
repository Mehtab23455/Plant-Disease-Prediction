# fix_h5_full.py
import h5py
import json
import shutil

old_file = "PlantDNet.h5"
new_file = "PlantDNet_fixed.h5"

# 1) Make a safe copy first
shutil.copyfile(old_file, new_file)
print(f"Copied {old_file} -> {new_file}")

def fix_model_config_attr(f):
    # model_config is stored in attrs as JSON bytes/string under key "model_config"
    if "model_config" not in f.attrs:
        print("No model_config attr found. Cannot fix config.")
        return False

    raw = f.attrs["model_config"]
    if hasattr(raw, "decode"):
        raw = raw.decode("utf-8")

    try:
        cfg = json.loads(raw)
    except Exception as e:
        print("Failed to parse model_config JSON:", e)
        return False

    changed = False

    # Recursively walk the JSON structure and replace '/' with '_' for any "name" keys
    def walk_and_fix(obj):
        nonlocal changed
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k == "name" and isinstance(v, str) and "/" in v:
                    newv = v.replace("/", "_")
                    obj[k] = newv
                    changed = True
                else:
                    walk_and_fix(v)
        elif isinstance(obj, list):
            for i in range(len(obj)):
                walk_and_fix(obj[i])
        # other primitives: ignore

    walk_and_fix(cfg)

    if changed:
        new_raw = json.dumps(cfg).encode("utf-8")
        f.attrs["model_config"] = new_raw
        print(" model_config updated (replaced '/' -> '_' in 'name' fields).")
    else:
        print("No 'name' fields with '/' found in model_config.")
    return True

def fix_model_weights_groups(f):
    # Rename any group under model_weights that contains '/'
    if "model_weights" not in f:
        print("No model_weights group found.")
        return False

    mw = f["model_weights"]
    # collect keys first to avoid runtime mutation problems
    keys = list(mw.keys())

    # Also need to update layer_names attribute stored in mw.attrs['layer_names']
    # Read current layer_names if present (could be split into chunks)
    def read_layer_names(group):
        # tries attrs['layer_names'] or attrs['layer_names0..n']
        if "layer_names" in group.attrs:
            arr = group.attrs["layer_names"]
            names = [n.decode("utf8") if hasattr(n, "decode") else n for n in arr]
            return names
        else:
            # try chunked versions
            names = []
            i = 0
            while True:
                key = f"layer_names{i}"
                if key in group.attrs:
                    arr = group.attrs[key]
                    names.extend([n.decode("utf8") if hasattr(n, "decode") else n for n in arr])
                    i += 1
                else:
                    break
            return names

    def write_layer_names(group, names):
        # store as bytes under layer_names (single attr) if size is ok
        bnames = [n.encode("utf8") for n in names]
        # remove any chunked original attrs first
        # clean up possible layer_namesX attrs
        keys_to_delete = [k for k in group.attrs.keys() if str(k).startswith("layer_names")]
        for k in keys_to_delete:
            try:
                del group.attrs[k]
            except Exception:
                pass
        group.attrs["layer_names"] = bnames

    # rename groups and collect mapping
    mapping = {}
    for key in keys:
        if "/" in key:
            new_key = key.replace("/", "_")
            print(f"Renaming group model_weights/{key} -> {new_key}")
            mw.move(key, new_key)
            mapping[key] = new_key

    # update layer_names attribute
    layer_names = read_layer_names(mw)
    if layer_names:
        new_layer_names = []
        for name in layer_names:
            if name in mapping:
                new_layer_names.append(mapping[name])
            else:
                # maybe names are bytes including slashes inside; just replace globally
                new_layer_names.append(name.replace("/", "_"))
        write_layer_names(mw, new_layer_names)
        print("Updated model_weights layer_names attribute.")
    else:
        print("No layer_names attribute found under model_weights (or it was empty).")

    # Also handle 'top_level_model_weights' group: its internal dataset names may include '/' pattern
    if "top_level_model_weights" in f:
        g = f["top_level_model_weights"]
        # weight names are stored in attr "weight_names" (as bytes)
        if "weight_names" in g.attrs:
            wlist = [n.decode("utf8") if hasattr(n, "decode") else n for n in g.attrs["weight_names"]]
            new_wlist = [w.replace("/", "_") for w in wlist]
            g.attrs["weight_names"] = [n.encode("utf8") for n in new_wlist]
            print("Updated top_level_model_weights weight_names attr.")
    return True

# Open the new file in r+ and attempt fixes
with h5py.File(new_file, "r+") as f:
    ok1 = fix_model_config_attr(f)
    ok2 = fix_model_weights_groups(f)

print("Done. Try loading PlantDNet_fixed.h5 in Keras now.")
