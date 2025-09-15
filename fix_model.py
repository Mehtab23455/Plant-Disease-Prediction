import tensorflow as tf

# Load your original model
model = tf.keras.models.load_model("PlantDNet.h5", compile=False)

# Fix illegal layer names
for layer in model.layers:
    if "/" in layer.name:
        print(f"Renaming {layer.name} -> {layer.name.replace('/', '_')}")
        layer._name = layer.name.replace("/", "_")

# Save new fixed model
model.save("PlantDNet_fixed.h5")
print(" Fixed model saved as PlantDNet_fixed.h5")
