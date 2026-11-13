#!/usr/bin/env python
"""Exports the NCR NPU graphs (ncr_npu_graphs.py) for the on-device probe.

For every configuration:
  1. training graphs: check the hand-written gradients against autograd (float64);
  2. write <name>.onnx (float32; the NPU computes in fp16) and list its operators;
  3. write test vectors next to it: <name>.in<i>.bin (fp16 inputs) and <name>.out<i>.bin (float32
     outputs of the float32 graph on those inputs), for configurations small enough to push to a phone;
  4. --om: convert to a portable <name>_fp16.om with the HiAI DDK's omg (Linux only, through WSL);
  5. --omc[=Kirin9050]: precompile <name>_fp16_<soc>.omc with the CANN toolkit's atc (WSL). Its log
     tells, without a phone, how the graph is split between NPU and CPU and which operators the NPU refused.

Run with a Python that has torch and onnx, e.g. E:/MyProjects/MRRCGI/.venv/Scripts/python.
The converters are the ones set up for the Device MRRC project (see its CLAUDE.md): omg from the
HiAI DDK tools (HIAI_OMG overrides the path), atc under /usr/local/Ascend in the default WSL distro.
"""
import argparse
import os
import shlex
import subprocess
import sys
import tempfile
import warnings

import numpy as np
import torch

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, HERE)
import ncr_npu_graphs as graphs  # noqa: E402
import ncr_reference as ref  # noqa: E402

DEFAULT_OUT = os.path.join(os.path.dirname(HERE), "_build", "npu_models")
DEFAULT_OMG = r"E:\MyProjects\MRRCGI\device_mrrc_benchmark-ddk-chs\scripts\hiai\DDK_tools_100.600.010.010\tools\tools_omg\omg"
OPSET = 9
VECTOR_LIMIT = 16 << 20  # bytes of fp16 input above which no test vectors are written


def real_configurations(folder):
    """Graphs fed with what the CUDA implementation actually computed on: the inputs and weights of
    one training step and one render inference, dumped by a Falcor run with NCR_DUMP_DIR set. They
    show the NPU on real value ranges (HDR targets, trained weights) instead of random numbers."""
    dump = ref.Dump(folder)
    after = ref.load_checkpoint(os.path.join(folder, "train_after.ckpt"))
    nets = {"diff": (graphs.DIFF, ref.DIFF_LAYERS), "spec": (graphs.SPEC, ref.SPEC_LAYERS)}

    def kernels(params, layers):
        return [torch.from_numpy(k.astype(np.float32)) for k in ref.split_layers(params, layers)]

    def inference(name):
        spec, layers = nets[name]
        model = graphs.Inference(spec, 256, 256)
        rows = dump[f"infer_{name}_input"][:65536].astype(np.float32)
        x = torch.from_numpy(np.ascontiguousarray(rows.T).reshape(1, spec.input_dim, 256, 256))
        model.inputs = [x] + kernels(after[name + "_ema"], layers)
        return model

    def training(name, batch=4096):
        spec, layers = nets[name]
        model = graphs.Training(spec, batch)
        rows = dump[f"train_{name}_input"][:batch].astype(np.float32)
        target = np.zeros((batch, ref.OUTPUT_PAD), dtype=np.float32)
        target[:, :ref.OUTPUT_DIM] = dump[f"train_{name}_target"][:batch]
        model.inputs = kernels(dump[f"before_{name}_params_half"], layers) + [
            torch.from_numpy(ref.rows_to_planar(rows)), torch.from_numpy(ref.rows_to_planar(target))]
        return model

    return {
        "ncr_real_diff_inf_256x256": lambda: inference("diff"),
        "ncr_real_spec_inf_256x256": lambda: inference("spec"),
        "ncr_real_diff_train_4096": lambda: training("diff"),
        "ncr_real_spec_train_4096": lambda: training("spec"),
    }


def configurations():
    """name -> factory. The shapes cover what the probe has to answer: cost per sample of each
    network, whether the tensor shape matters, how the training graph scales with the batch."""
    D, S = graphs.DIFF, graphs.SPEC
    return {
        "ncr_diff_inf_256x256": lambda: graphs.Inference(D, 256, 256),
        "ncr_spec_inf_256x256": lambda: graphs.Inference(S, 256, 256),
        "ncr_spec_inf_65536x1": lambda: graphs.Inference(S, 65536, 1),
        "ncr_spec_inf_256x256_out4": lambda: graphs.Inference(S, 256, 256, graphs.OUTPUT_DIM),
        "ncr_diff_inf_512x512": lambda: graphs.Inference(D, 512, 512),
        "ncr_spec_inf_512x512": lambda: graphs.Inference(S, 512, 512),
        "ncr_diff_train_4096": lambda: graphs.Training(D, 4096),
        "ncr_spec_train_2048": lambda: graphs.Training(S, 2048),
        "ncr_spec_train_4096": lambda: graphs.Training(S, 4096),
        "ncr_spec_train_8192": lambda: graphs.Training(S, 8192),
        "ncr_spec_train_16384": lambda: graphs.Training(S, 16384),
        "ncr_spec_train_4096_sign": lambda: graphs.Training(S, 4096, mask="sign"),
        "ncr_spec_train_4096_clip": lambda: graphs.Training(S, 4096, mask="clip"),
    }


def as_list(outputs):
    return list(outputs) if isinstance(outputs, (list, tuple)) else [outputs]


def relative_error(a, b):
    """(largest element error / largest element, L2 error / L2 norm)."""
    return (float((a - b).abs().max() / b.abs().max().clamp_min(1e-30)),
            float((a - b).norm() / b.norm().clamp_min(1e-30)))


# Accepted L2 error of a gradient tensor relative to its norm. Only "sign" is an exact ReLU derivative
# (see graphs.relu_mask); with the others the few activations close to zero get a partial mask, which
# moves single elements by several percent but the tensor as a whole very little.
GRADIENT_TOLERANCE = {"sign": 1e-6, "clip": 5e-2, "div": 5e-2}


def check_gradients(model):
    """Hand-written gradients vs autograd in float64, on a batch of 512."""
    torch.manual_seed(1)
    double = type(model)(model.spec, 512, model.mask).double()
    inputs = [t.double() for t in double.inputs]
    n_layers = len(double.shapes)
    with torch.no_grad():
        ours = double(inputs)
    reference = graphs.reference_gradients(double.spec, inputs[:n_layers], inputs[n_layers], inputs[n_layers + 1])
    errors = [relative_error(o, r) for o, r in zip(ours, reference)]
    return max(e[0] for e in errors), max(e[1] for e in errors)


def export_onnx(model, path):
    kwargs = dict(input_names=model.input_names, output_names=model.output_names)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            torch.onnx.export(model, (model.inputs,), path, opset_version=OPSET, dynamo=False, **kwargs)
        except TypeError:  # torch without the dynamo keyword
            torch.onnx.export(model, (model.inputs,), path, opset_version=OPSET, **kwargs)


def onnx_op_types(path):
    import onnx
    counts = {}
    for node in onnx.load(path).graph.node:
        counts[node.op_type] = counts.get(node.op_type, 0) + 1
    return counts


def write_vectors(model, base):
    """fp16 inputs and the float32 graph's outputs on exactly those values."""
    inputs = [t.half() for t in model.inputs]
    if sum(t.numel() * 2 for t in inputs) > VECTOR_LIMIT:
        return False
    with torch.no_grad():
        outputs = as_list(model([t.float() for t in inputs]))
    for i, t in enumerate(inputs):
        t.numpy().tofile(f"{base}.in{i}.bin")
    for i, t in enumerate(outputs):
        t.float().numpy().astype(np.float32).tofile(f"{base}.out{i}.bin")
    return True


def wsl_path(path):
    path = os.path.abspath(path)
    drive, rest = os.path.splitdrive(path)
    return f"/mnt/{drive[0].lower()}{rest.replace(os.sep, '/')}"


def convert_to_om(model, base, omg):
    args = [wsl_path(omg), "--model", wsl_path(base + ".onnx"), "--output", wsl_path(base + "_fp16"),
            "--framework", "5", "--weight_merge=false",
            "--input_type", ";".join(f"{n}:FP16" for n in model.input_names),
            "--output_type", ";".join(f"{n}:FP16" for n in model.output_names)]
    result = subprocess.run(["wsl.exe", "-u", "root", "--", "bash", "-c", shlex.join(args)],
                            capture_output=True, text=True, errors="replace")
    om = base + "_fp16.om"
    if result.returncode != 0 or not os.path.exists(om):
        print((result.stdout + result.stderr).replace("\0", "").strip())
        sys.exit(f"omg failed for {os.path.basename(base)} (exit code {result.returncode})")
    print(f"     {os.path.basename(om)}  ({os.path.getsize(om)} bytes)")


def convert_to_omc(model, base, soc):
    omc_base = f"{base}_fp16_{soc.lower()}"
    omc = omc_base + ".omc"
    if os.path.exists(omc):
        os.remove(omc)
    # With several outputs and no --out_nodes, atc only takes the bare type for --output_type.
    atc = ["atc", f"--model={wsl_path(base + '.onnx')}", f"--output={wsl_path(omc_base)}", "--framework=5",
           "--target=omc", f"--soc_version={soc}", "--weight_merge=false",
           "--input_type=" + ";".join(f"{n}:FP16" for n in model.input_names), "--output_type=FP16"]
    # A script file, not `bash -c`: wsl.exe hands the command line to a shell that would expand the $-variables.
    script = f"""#!/bin/bash
for env in /usr/local/Ascend/set_env_wsl1.sh /usr/local/Ascend/cann/set_env.sh; do
    if [ -f "$env" ]; then source "$env"; break; fi
done
command -v atc > /dev/null || {{ echo "atc not found: install the CANN toolkit under /usr/local/Ascend"; exit 127; }}
work=$(mktemp -d)
cd "$work"
{shlex.join(atc)}
status=$?
cd /
rm -rf "$work"
exit $status
"""
    with tempfile.TemporaryDirectory() as folder:
        script_path = os.path.join(folder, "atc.sh")
        with open(script_path, "w", newline="\n") as f:
            f.write(script)
        result = subprocess.run(["wsl.exe", "-u", "root", "--", "bash", wsl_path(script_path)],
                                capture_output=True, text=True, errors="replace")
    log = (result.stdout + result.stderr).replace("\0", "").strip()
    log = "\n".join(line for line in log.splitlines() if not line.startswith("wsl:"))
    with open(omc_base + ".atc.log", "w", encoding="utf-8", newline="\n") as f:
        f.write(log + "\n")
    # atc's first step always prints an "E19999 ... Run custom pass" error and carries on; what counts is
    # the exit code and the "OMG generate offline model success" of the last step.
    if result.returncode != 0 or "OMG generate offline model success" not in log or not os.path.exists(omc):
        print(log[-4000:])
        print(f"     atc FAILED for {os.path.basename(base)} (exit code {result.returncode}); log: {omc_base}.atc.log")
        return False
    partitions = sorted({line[line.index("partition type"):].strip() for line in log.splitlines() if "partition type" in line})
    rejected = sorted({line.split("the op name [")[1].split("]")[0] for line in log.splitlines()
                       if "is not supported in npucl store" in line and "the op name [" in line})
    print(f"     {os.path.basename(omc)}  ({os.path.getsize(omc)} bytes)  {'; '.join(partitions) or 'no partition line'}")
    if rejected:
        print(f"     refused by the NPU: {', '.join(rejected)}")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--only", action="append", help="export only the configurations whose name contains this (repeatable)")
    parser.add_argument("--om", action="store_true")
    parser.add_argument("--omc", nargs="?", const="Kirin9050", default=None, metavar="SOC")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--real", metavar="DUMP_DIR",
                        help="add the ncr_real_* configurations, built from a Falcor dump (see verify_tcnn.py)")
    options = parser.parse_args()

    available = configurations()
    if options.real:
        available.update(real_configurations(options.real))
    selected = {name: factory for name, factory in available.items()
                if not options.only or any(part in name for part in options.only)}
    if options.list:
        print("\n".join(selected))
        return
    os.makedirs(options.out, exist_ok=True)
    omg = os.environ.get("HIAI_OMG", DEFAULT_OMG)
    if options.om and not os.path.exists(omg):
        sys.exit(f"omg not found at {omg}; set HIAI_OMG")

    for spec in graphs.NETS.values():
        print(f"{spec.name}: layers {spec.layer_shapes()}, {spec.n_params()} parameters")
    for name, factory in selected.items():
        torch.manual_seed(0)
        model = factory().eval()
        base = os.path.join(options.out, name)
        shapes = ", ".join(f"{n}{tuple(t.shape)}" for n, t in zip(model.input_names, model.inputs)
                           if not n.startswith("w"))
        print(f"{name}: {shapes}")
        if isinstance(model, graphs.Training):
            worst, error = check_gradients(model)
            print(f"     gradients vs autograd ({model.mask} mask): L2 error {error:.1e}, worst element {worst:.1e}")
            if error > GRADIENT_TOLERANCE[model.mask]:
                sys.exit(f"{name}: the hand-written gradients do not match autograd")
        export_onnx(model, base + ".onnx")
        ops = onnx_op_types(base + ".onnx")
        print("     ops: " + ", ".join(f"{op} x{count}" for op, count in sorted(ops.items())))
        print("     test vectors: " + ("written" if write_vectors(model, base) else "skipped (too large)"))
        if options.om:
            convert_to_om(model, base, omg)
        if options.omc:
            convert_to_omc(model, base, options.omc)


if __name__ == "__main__":
    main()
