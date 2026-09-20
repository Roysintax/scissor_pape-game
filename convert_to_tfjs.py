"""
Convert trained Keras model to TensorFlow.js format for in-browser inference.
Outputs web/model/model.json and web/model/*.bin
Transforms Keras 3 serialization into Keras 2 / TensorFlow.js compatible format.
Resolves DepthwiseConv2D weight naming differences (kernel -> depthwise_kernel).
"""

import os
import sys
import shutil
import json
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import models
from tensorflowjs.converters import keras_h5_conversion

# Allow Keras 3 models during conversion
keras_h5_conversion._check_version = lambda h5: None

MODEL_PATH = os.path.join("model", "rock_paper_scissors.keras")
OUTPUT_DIR = os.path.join("web", "model")

def convert():
    # 1. Validate Keras model exists and can run inference
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        print("Please run train.py first.")
        sys.exit(1)

    print(f"Loading Keras model from {MODEL_PATH}...")
    trained = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={"preprocess_input": tf.keras.applications.mobilenet_v2.preprocess_input}
    )
    print("Model loaded successfully.")

    # Validate Python dummy prediction
    dummy_input = tf.zeros((1, 224, 224, 3))
    dummy_pred = trained(dummy_input, training=False)
    print(f"Python dummy prediction verified: shape {dummy_pred.shape}, output: {dummy_pred.numpy()}")

    # 2. Clean old converted artifacts in web/model/
    target_dir = Path(OUTPUT_DIR)
    if target_dir.exists():
        print(f"Cleaning stale files in {OUTPUT_DIR}...")
        for file in target_dir.glob("*"):
            if file.is_file() and file.name != "classes.json":
                file.unlink()
    else:
        target_dir.mkdir(parents=True, exist_ok=True)

    # 3. Unnest MobileNetV2 into a flat Model graph so TensorFlow.js can deserialize all layers
    base = trained.get_layer("mobilenetv2_1.00_224")
    x = base.output
    x = trained.get_layer("global_avg_pool")(x)
    x = trained.get_layer("dropout")(x)
    out = trained.get_layer("predictions")(x)
    export_model = models.Model(base.input, out, name="rock_paper_scissors")

    print(f"Converting model ({len(export_model.layers)} layers) to TensorFlow.js in {OUTPUT_DIR}...")
    keras_h5_conversion.save_keras_model(export_model, OUTPUT_DIR)

    # 4. Transform Keras 3 JSON schema to Keras 2 / TensorFlow.js compatible schema
    json_path = os.path.join(OUTPUT_DIR, "model.json")
    with open(json_path, "r", encoding="utf-8") as f:
        model_json = json.load(f)

    topo = model_json.get("modelTopology", {})
    topo["keras_version"] = "2.15.0"
    if topo.get("model_config", {}).get("class_name") == "Functional":
        topo["model_config"]["class_name"] = "Model"

    config = topo.get("model_config", {}).get("config", {})

    # Ensure input_layers and output_layers are lists of lists for Keras 2
    if isinstance(config.get("input_layers"), list):
        if len(config["input_layers"]) > 0 and not isinstance(config["input_layers"][0], list):
            config["input_layers"] = [config["input_layers"]]

    if isinstance(config.get("output_layers"), list):
        if len(config["output_layers"]) > 0 and not isinstance(config["output_layers"][0], list):
            config["output_layers"] = [config["output_layers"]]

    # Track DepthwiseConv2D layer names
    depthwise_layer_names = set()

    # Convert layer configurations
    for layer in config.get("layers", []):
        layer_name = layer.get("name")
        class_name = layer.get("class_name")
        if class_name == "DepthwiseConv2D":
            depthwise_layer_names.add(layer_name)

        c = layer.get("config", {})
        # Flatten dtype
        if isinstance(c.get("dtype"), dict):
            c["dtype"] = c["dtype"].get("config", {}).get("name", "float32")
        # InputLayer batch_shape -> batch_input_shape
        if class_name == "InputLayer":
            if "batch_shape" in c:
                c["batch_input_shape"] = c.pop("batch_shape")
        # Inbound nodes conversion (Keras 3 dict -> Keras 2 nested list)
        new_inbound = []
        for node in layer.get("inbound_nodes", []):
            if isinstance(node, dict) and "args" in node:
                node_entries = []
                for arg in node["args"]:
                    if isinstance(arg, dict) and "config" in arg and "keras_history" in arg["config"]:
                        hist = arg["config"]["keras_history"]
                        node_entries.append([hist[0], hist[1], hist[2], {}])
                    elif isinstance(arg, list):
                        for sub in arg:
                            if isinstance(sub, dict) and "config" in sub and "keras_history" in sub["config"]:
                                hist = sub["config"]["keras_history"]
                                node_entries.append([hist[0], hist[1], hist[2], {}])
                if node_entries:
                    new_inbound.append(node_entries)
            elif isinstance(node, list):
                new_inbound.append(node)
        layer["inbound_nodes"] = new_inbound

    # 5. Fix DepthwiseConv2D weight names in weightsManifest:
    # Keras 3 names DepthwiseConv2D kernel as 'kernel', but TFJS expects 'depthwise_kernel'.
    renamed_depthwise = 0
    for group in model_json.get("weightsManifest", []):
        for weight in group.get("weights", []):
            wname = weight.get("name", "")
            parts = wname.split("/")
            if len(parts) == 2 and parts[0] in depthwise_layer_names and parts[1] == "kernel":
                weight["name"] = f"{parts[0]}/depthwise_kernel"
                renamed_depthwise += 1

    print(f"Mapped {renamed_depthwise} DepthwiseConv2D weights to 'depthwise_kernel' for TFJS compatibility.")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(model_json, f, indent=2)

    # 6. Validate output
    assert os.path.exists(json_path), "model.json missing"
    manifest = model_json.get("weightsManifest", [])
    total_weights = sum(len(g.get("weights", [])) for g in manifest)
    shard_paths = [p for g in manifest for p in g.get("paths", [])]
    print(f"Validation: {total_weights} weights defined across {len(shard_paths)} shards.")
    for shard in shard_paths:
        shard_file = os.path.join(OUTPUT_DIR, shard)
        assert os.path.exists(shard_file), f"Weight shard missing: {shard_file}"
        print(f"  Shard {shard}: {os.path.getsize(shard_file)} bytes")

    # Verify block_10_depthwise/depthwise_kernel is present in manifest
    manifest_names = {w["name"] for g in manifest for w in g.get("weights", [])}
    assert "block_10_depthwise/depthwise_kernel" in manifest_names, "block_10_depthwise/depthwise_kernel missing in manifest!"
    print("Verified block_10_depthwise/depthwise_kernel is present and properly mapped in manifest.")

    # Copy classes.json to web/model for frontend access
    classes_src = os.path.join("model", "classes.json")
    classes_dst = os.path.join(OUTPUT_DIR, "classes.json")
    if os.path.exists(classes_src):
        shutil.copy2(classes_src, classes_dst)
        print(f"Copied metadata to {classes_dst}")

    print("\nConversion and validation complete successfully.")

if __name__ == "__main__":
    convert()
