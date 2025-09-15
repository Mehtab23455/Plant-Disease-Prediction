import h5py

old_file = "PlantDNet.h5"
new_file = "PlantDNet_fixed.h5"

with h5py.File(old_file, "r") as f:
    with h5py.File(new_file, "w") as g:
        f.copy("/", g)

# Now fix names inside the new file
with h5py.File(new_file, "r+") as f:
    def rename_items(group):
        for key in list(group.keys()):
            if "/" in key:
                new_key = key.replace("/", "_")
                print(f"Renaming {key} -> {new_key}")
                group.move(key, new_key)
            if isinstance(group[key], h5py.Group):
                rename_items(group[key])

    rename_items(f)

print("Saved fixed model as PlantDNet_fixed.h5")
