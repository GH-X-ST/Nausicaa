# v11.4 Fixed-Gate W0/W1 Post-Run Triage

- Run scope: $(System.Collections.Specialized.OrderedDictionary.run_scope)
- Chunk counts: $(System.Collections.Specialized.OrderedDictionary.chunk_counts.completed_chunk_count) completed, $(System.Collections.Specialized.OrderedDictionary.chunk_counts.skipped_chunk_count) skipped, $(System.Collections.Specialized.OrderedDictionary.chunk_counts.failed_chunk_count) failed, $(System.Collections.Specialized.OrderedDictionary.chunk_counts.corrupt_chunk_count) corrupt; $(System.Collections.Specialized.OrderedDictionary.chunk_counts.complete_chunk_manifest_count) complete chunk manifests present.
- Candidate rows: $(System.Collections.Specialized.OrderedDictionary.candidate_row_count)
- Rollout rows: $(System.Collections.Specialized.OrderedDictionary.rollout_row_count)
- W0 rows by branch: $(System.Collections.Specialized.OrderedDictionary.w0_row_count_by_branch | ConvertTo-Json -Compress)
- W1 rows by branch: $(System.Collections.Specialized.OrderedDictionary.w1_row_count_by_branch | ConvertTo-Json -Compress)
- W1 measured-updraft rows by branch: $(System.Collections.Specialized.OrderedDictionary.w1_measured_updraft_row_count_by_branch | ConvertTo-Json -Compress)
- W1 scheduled independently of W0 success: $(System.Collections.Specialized.OrderedDictionary.w1_scheduled_independently_of_w0_success)
- Branch coverage conclusion: $(System.Collections.Specialized.OrderedDictionary.branch_coverage_conclusion)
- Reachable-state extraction: $(System.Collections.Specialized.OrderedDictionary.downstream_status.reachable_state_extraction.status) with $(System.Collections.Specialized.OrderedDictionary.downstream_status.reachable_state_extraction.rows) rows
- Primitive-envelope clustering: $(System.Collections.Specialized.OrderedDictionary.downstream_status.primitive_envelope_clustering.status) with $(System.Collections.Specialized.OrderedDictionary.downstream_status.primitive_envelope_clustering.governor_candidate_rows) governor candidates
- W2 focused replay: $(System.Collections.Specialized.OrderedDictionary.downstream_status.w2_focused_replay.status) with $(System.Collections.Specialized.OrderedDictionary.downstream_status.w2_focused_replay.rows) rows
- W3 domain-randomised replay: $(System.Collections.Specialized.OrderedDictionary.downstream_status.w3_domain_randomised_replay.status) with $(System.Collections.Specialized.OrderedDictionary.downstream_status.w3_domain_randomised_replay.rows) rows
- Policy evaluation: $(System.Collections.Specialized.OrderedDictionary.downstream_status.repeated_launch_policy_eval.status) with $(System.Collections.Specialized.OrderedDictionary.downstream_status.repeated_launch_policy_eval.episodes) episodes; toy candidates used: $(System.Collections.Specialized.OrderedDictionary.downstream_status.repeated_launch_policy_eval.default_toy_candidates_used)

No real-flight transfer, mission success, same-flight recapture, perching, all-arena validity, hardware-ready agile-turn, true delayed-state-feedback, full W2/W3 robustness, or real repeated-launch validation claim is made.
