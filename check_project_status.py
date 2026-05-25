#!/usr/bin/env python3
"""
Project Status Checker for GPT-OSS Multimodal EO
Checks the completion status of all project phases
"""

import os
import torch
import numpy as np

def check_project_status():
    print("GPT-OSS MULTIMODAL EO PROJECT STATUS")
    print("=" * 50)

    # Check data directory
    print("\nDATA FILES:")
    processed_dir = "data/processed"
    if os.path.exists(processed_dir):
        files = os.listdir(processed_dir)
        for file in files:
            filepath = os.path.join(processed_dir, file)
            size = os.path.getsize(filepath)
            print(f"  [OK] {file} ({size:,} bytes)")
    else:
        print("  [ERROR] No processed data directory")

    # Check checkpoints
    print("\nMODEL CHECKPOINTS:")
    checkpoints_dir = "checkpoints"
    if os.path.exists(checkpoints_dir):
        files = os.listdir(checkpoints_dir)
        for file in files:
            filepath = os.path.join(checkpoints_dir, file)
            size = os.path.getsize(filepath)
            print(f"  [OK] {file} ({size:,} bytes)")
    else:
        print("  [ERROR] No checkpoints directory")

    # Check embeddings
    print("\nEMBEDDINGS STATUS:")
    text_embeddings_file = "data/processed/text_embeddings.pt"
    if os.path.exists(text_embeddings_file):
        try:
            embeddings = torch.load(text_embeddings_file, map_location="cpu", weights_only=True)
            print(f"  [OK] Text embeddings: {embeddings.shape}")
            print(f"     Device: cpu")
            print(f"     Normalized: {torch.allclose(torch.norm(embeddings, dim=1), torch.ones(embeddings.shape[0]), atol=1e-6)}")
        except Exception as e:
            print(f"  [ERROR] Error loading text embeddings: {e}")
    else:
        print("  [ERROR] Text embeddings not found")

    eo_embeddings_file = "data/processed/eo_embeddings.npy"
    if os.path.exists(eo_embeddings_file):
        try:
            embeddings = np.load(eo_embeddings_file)
            print(f"  [OK] EO embeddings: {embeddings.shape}")
        except Exception as e:
            print(f"  [ERROR] Error loading EO embeddings: {e}")
    else:
        print("  [ERROR] EO embeddings not found")

    # Phase completion status
    print("\nPHASE COMPLETION STATUS:")
    phases = {
        "Phase 1 (Data Preparation)": os.path.exists("data/processed") and len(os.listdir("data/processed")) > 0,
        "Phase 2 (Vision Training)": os.path.exists("checkpoints/vision_best.pth"),
        "Phase 3 (LLM/Text Processing)": os.path.exists("data/processed/text_embeddings.pt"),
        "Phase 4 (Multimodal Fusion)": os.path.exists("checkpoints/fusion_best.pth"),
        "Phase 5 (Inference)": os.path.exists("results/fusion_test_results.json"),
    }

    for phase, completed in phases.items():
        status = "[COMPLETED]" if completed else "[PENDING]"
        print(f"  {phase}: {status}")

    # Component status
    print("\nCOMPONENT STATUS:")
    components = {
        "Vision Model": os.path.exists("vision/model.py"),
        "Multimodal Model": os.path.exists("multimodal/multimodal_model.py"),
        "Fusion Model": os.path.exists("fusion/fusion_model.py"),
        "Training Scripts": all([
            os.path.exists("vision/train.py"),
            os.path.exists("multimodal/train_multimodal.py"),
            os.path.exists("fusion/train_fusion.py")
        ]),
        "Inference Scripts": all([
            os.path.exists("inference/run_inference.py"),
            os.path.exists("inference/evaluate.py")
        ]),
    }

    for component, exists in components.items():
        status = "[IMPLEMENTED]" if exists else "[MISSING]"
        print(f"  {component}: {status}")

    print("\n" + "=" * 50)
    completed_phases = sum(phases.values())
    print(f"OVERALL PROGRESS: {completed_phases}/{len(phases)} phases completed")

    if completed_phases == len(phases):
        print("PROJECT COMPLETE! Ready for deployment.")
    else:
        print("Continue with remaining phases.")

if __name__ == "__main__":
    check_project_status()