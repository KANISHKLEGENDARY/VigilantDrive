# Record and label demo videos
1. Record short, consented clips using the same camera position as the demo.
2. Include separate normal and drowsy examples; do not induce unsafe sleepiness while driving.
3. Name files with conditions, such as `day_normal.mp4` or `night_glasses.mp4`.
4. Create `labels.csv` with columns `video,start_sec,end_sec,label` and one interval per row.
5. Use `drowsy` for labeled events and `normal` for verified normal intervals; times are seconds from clip start.
6. Run `python scripts/test_videos.py` and review the clips and generated `results/results.md` manually.
