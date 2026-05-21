# paired_w0_w1_proof Chunked Progress

- Status: `complete`
- Selected worker count: `8`
- Worker fallback reason: `none`
- Scheduled chunks: `4`
- Completed chunks: `4`
- Failed chunks: `0`

Recommended command:

```powershell
python 03_Control/04_Scenarios/run_paired_w0_w1_partitioned_planning.py --run-id 13 --paired-scale-mode proof --proof-target-trials-per-environment 2500 --partition-rows 2500
python 03_Control/04_Scenarios/run_paired_w0_w1_archive_chunked.py --run-id 14 --planning-run-id 13 --workers 8 --max-workers 8 --resume
python 03_Control/04_Scenarios/aggregate_paired_w0_w1_archive.py --run-id 14 --planning-run-id 13 --expected-trials-per-environment 2500 --build-upload-package --build-governor-package
```
