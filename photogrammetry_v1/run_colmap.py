import os
import subprocess
import sys

COLMAP_BIN = r"C:\Users\bhaskar\Downloads\colmap-x64-windows-cuda\bin\colmap.exe"

def run_cmd(args):
    print(f"Running command: {' '.join(args)}")
    sys.stdout.flush()
    result = subprocess.run(args, check=True)
    return result

def main():
    # Make sure we are in the script's directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Check if sparse reconstruction is already done
    sparse_done = os.path.exists("sparse/0/points3D.bin")
    
    if not sparse_done:
        # 1. Feature extraction
        feature_extractor_args = [
            COLMAP_BIN, "feature_extractor",
            "--database_path", "database.db",
            "--image_path", "images"
        ]
        print("=== STEP 1: Feature Extraction ===")
        run_cmd(feature_extractor_args)
        
        # 2. Feature matching
        exhaustive_matcher_args = [
            COLMAP_BIN, "exhaustive_matcher",
            "--database_path", "database.db"
        ]
        print("\n=== STEP 2: Feature Matching ===")
        run_cmd(exhaustive_matcher_args)
        
        # 3. Sparse mapping
        mapper_args = [
            COLMAP_BIN, "mapper",
            "--database_path", "database.db",
            "--image_path", "images",
            "--output_path", "sparse"
        ]
        print("\n=== STEP 3: Sparse Reconstruction ===")
        run_cmd(mapper_args)
    else:
        print("=== Sparse reconstruction already exists. Skipping steps 1-3. ===")

    # Check if dense reconstruction is already done
    dense_done = os.path.exists("output/pointcloud.ply")
    
    if not dense_done:
        # 4. Dense Image Undistortion
        os.makedirs("dense", exist_ok=True)
        undistorter_args = [
            COLMAP_BIN, "image_undistorter",
            "--image_path", "images",
            "--input_path", "sparse/0",
            "--output_path", "dense",
            "--output_type", "COLMAP"
        ]
        print("\n=== STEP 4: Image Undistortion ===")
        run_cmd(undistorter_args)

        # 5. Patch Match Stereo dense matching
        patch_match_args = [
            COLMAP_BIN, "patch_match_stereo",
            "--workspace_path", "dense",
            "--workspace_format", "COLMAP",
            "--PatchMatchStereo.geom_consistency", "true",
            "--PatchMatchStereo.max_image_size", "2000"
        ]
        print("\n=== STEP 5: Patch Match Stereo (Dense Matching) ===")
        run_cmd(patch_match_args)

        # 6. Stereo Fusion to generate dense point cloud
        os.makedirs("output", exist_ok=True)
        fusion_args = [
            COLMAP_BIN, "stereo_fusion",
            "--workspace_path", "dense",
            "--workspace_format", "COLMAP",
            "--input_type", "geometric",
            "--output_path", "output/pointcloud.ply"
        ]
        print("\n=== STEP 6: Stereo Fusion (Dense Point Cloud Generation) ===")
        run_cmd(fusion_args)
    else:
        print("=== Dense point cloud already exists. Skipping steps 4-6. ===")

    # 7. Poisson Mesher to generate 3D mesh
    os.makedirs("output", exist_ok=True)
    mesher_args = [
        COLMAP_BIN, "poisson_mesher",
        "--input_path", "output/pointcloud.ply",
        "--output_path", "output/mesh.ply",
        "--PoissonMeshing.trim", "10",
        "--PoissonMeshing.depth", "10"
    ]
    print("\n=== STEP 7: Poisson Meshing ===")
    run_cmd(mesher_args)
    
    print("\nCOLMAP meshing completed successfully!")

if __name__ == "__main__":
    main()
