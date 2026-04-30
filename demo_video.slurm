#!/bin/bash
#SBATCH --job-name=cut3r_video_render
#SBATCH --output=logs/cut3r_video_render_%j.out
#SBATCH --error=logs/cut3r_video_render_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=workq

echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"

source ~/miniforge3/bin/activate
conda activate pytorch-cu128

echo "Python: $(python --version)"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

CUT3R_DIR="/lus/lfs1aip2/scratch/b6ar/trvbale.b6ar/VLM-memory/CUT3R"
cd "$CUT3R_DIR"

mkdir -p logs

export PYTHONWARNINGS=ignore
export TOKENIZERS_PARALLELISM=false

MODEL_PATH="${MODEL_PATH:-src/cut3r_512_dpt_4_64.pth}"
SEQ_PATH="${SEQ_PATH:-examples/001}"
OUTPUT_DIR="${OUTPUT_DIR:-./demo_tmp}"

python demo.py \
    --model_path "$MODEL_PATH" \
    --seq_path "$SEQ_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --output_video \
    --orbit_frames 120 \
    --video_fps 12 \
    --orbit_radius 2.0 \
    --right_offset 1.0 \
    --size 512

echo "End Time: $(date)"