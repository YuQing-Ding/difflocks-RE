#!/usr/bin/env python3

# ./inference_difflocks.py \
#   --img_path=./samples/medium_11.png \
#   --out_path=./outputs_inference/ 

from inference.img2hair import DiffLocksInference
import subprocess
import os
import argparse
import torch
import numpy as np
import random

torch.manual_seed(5)
np.random.seed(5)
random.seed(5)


def _run_and_capture(cmd):
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def _looks_like_windows_path(path):
    if not path:
        return False
    return ":\\" in path or path.startswith("\\\\") or "\\" in path


def _to_wsl_path(path):
    if not path:
        return path
    if os.path.exists(path):
        return os.path.abspath(path)
    if _looks_like_windows_path(path):
        return _run_and_capture(["wslpath", "-u", path])
    return path


def _to_windows_path(path):
    return _run_and_capture(["wslpath", "-w", path])


def _resolve_blender_executable(blender_path):
    blender_path = _to_wsl_path(blender_path)
    if not blender_path:
        return ""

    candidates = []
    if os.path.isdir(blender_path):
        candidates.extend(
            [
                os.path.join(blender_path, "blender.exe"),
                os.path.join(blender_path, "blender-launcher.exe"),
                os.path.join(blender_path, "blender"),
            ]
        )
    else:
        candidates.append(blender_path)

    for candidate in candidates:
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)

    raise FileNotFoundError(
        f"Could not find a Blender executable from '{blender_path}'. "
        f"Tried: {candidates}"
    )


def _prepare_blender_command_args(blender_executable, repo_path, input_npz, out_path):
    blender_script = os.path.join(repo_path, "inference", "npz2blender.py")

    if blender_executable.lower().endswith(".exe"):
        return (
            _to_windows_path(blender_script),
            _to_windows_path(input_npz),
            _to_windows_path(out_path),
        )

    return blender_script, input_npz, out_path

def run():

    #argparse
    parser = argparse.ArgumentParser(description='Get the weights of each dimensions after training a strand VAE')
    parser.add_argument('--strand_checkpoint_path', default="./checkpoints/strand_vae/strand_codec.pt", type=str, help='Path to the strandVAE checkpoint')
    parser.add_argument('--difflocks_checkpoint_path', default="./checkpoints/difflocks_diffusion/scalp_v9_40k_06730000.pth", type=str, help='Path to the difflocks checkpoint')
    parser.add_argument('--difflocks_config_path', default="./configs/config_scalp_texture_conditional.json", type=str, help='Path to the difflocks config')
    parser.add_argument('--rgb2mat_checkpoint_path', default="./checkpoints/rgb2material/rgb2material.pt", type=str,  help='Path to the rgb2material checkpoint')
    parser.add_argument('--blender_path', type=str, default="", help='Path to the blender executable')
    parser.add_argument('--blender_nr_threads', default=8, type=int, help='Number of threads for blender to use')
    parser.add_argument('--blender_strands_subsample', default=1.0, type=float, help='Amount of subsample of the strands(1.0=full strands, 0.5=half strands)')
    parser.add_argument('--blender_vertex_subsample', default=1.0, type=float, help='Amount of subsample of the vertices(1.0=all vertex, 0.5=half number of vertices per strand)')
    parser.add_argument('--alembic_resolution', default=7, type=int, help='Resolution of the exported alembic')
    parser.add_argument('--export_alembic', action='store_true', help='weather to export alembic or not')
    parser.add_argument('--do_shrinkwrap', action='store_true', help='applies a shrinkwrap modifier in blender that pushes the strands away from the scalp so they dont pass through the head')
    parser.add_argument('--ue', action='store_true', help='apply UE-friendly rotation to the exported abc (x=-90, z=180)')
    parser.add_argument('--img_path', type=str, required=True, help='Path to the image to do inference on')
    parser.add_argument('--out_path', type=str, required=True, help='Path to the image to do inference on')
    args = parser.parse_args()

    print("args is", args)
    
    difflocks= DiffLocksInference(args.strand_checkpoint_path, args.difflocks_config_path, args.difflocks_checkpoint_path, args.rgb2mat_checkpoint_path)


    #run----
    # img_path="./samples/medium_11.png"
    base_name = os.path.splitext(os.path.basename(args.img_path))[0]
    out_dir = os.path.join(args.out_path, base_name)
    strand_points_world, hair_material_dict=difflocks.file2hair(args.img_path, out_dir, base_name=base_name) 
    print("hair_material_dict",hair_material_dict)


    #create blender file and optionally an alembic file
    if args.blender_path!="":
        repo_path = os.path.dirname(os.path.abspath(__file__))
        blender_executable = _resolve_blender_executable(args.blender_path)
        input_npz = os.path.abspath(os.path.join(out_dir, f"{base_name}_difflocks_output_strands.npz"))
        out_path = os.path.abspath(out_dir)
        blender_script, input_npz, out_path = _prepare_blender_command_args(
            blender_executable, repo_path, input_npz, out_path
        )
        should_export_alembic = True

        print("Using Blender executable:", blender_executable)
        print("Alembic export enabled:", should_export_alembic)

        cmd=[blender_executable, "-t", str(args.blender_nr_threads), "--background", "--python", blender_script, "--", "--input_npz", input_npz, "--out_path", out_path, "--basename", base_name, "--strands_subsample", str(args.blender_strands_subsample), "--vertex_subsample", str(args.blender_vertex_subsample), "--alembic_resolution", str(args.alembic_resolution) ]
        if args.do_shrinkwrap:
            cmd.append("--shrinkwrap")
        if args.ue:
            cmd.append("--ue")
        if should_export_alembic or args.export_alembic:
            cmd.append("--export_alembic")
        subprocess.run(cmd, capture_output=False, check=True)

        hair_abc_path = os.path.join(os.path.abspath(out_dir), f"{base_name}.abc")
        if not os.path.exists(hair_abc_path):
            raise FileNotFoundError(f"Blender finished but did not create {hair_abc_path}")

    print("Finished writing to ", out_dir)

if __name__ == '__main__':

    run()
