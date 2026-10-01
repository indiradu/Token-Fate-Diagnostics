# Locking FLOP accounting

Purpose: aligned standard LLaDA comparison

Convention: 1 MAC = 2 FLOPs. Backbone includes attention projections, QK/AV attention matmuls, and gated FFN only. LM head and controller/projector GEMMs are separate.

Complete paired cases: 124
Controller profile status: joined

|dataset|policy_id|cases|correct|accuracy_pct|speedup_pct|backbone_tflops_per_example|backbone_reduction_pct|total_with_lm_head_tflops_per_example|total_with_lm_head_reduction_pct|controller_profile_expected_cases|controller_profile_covered_cases|controller_profile_pending_cases|controller_total_tflops_per_example|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|humaneval|B0|70|38|54.2857|0|1465.37|0|1572.48|0|0|0|0|0|
|humaneval|Confidence-0.9|70|39|55.7143|652.168|195.119|86.6847|209.372|86.6852|0|0|0|0|
|humaneval|RepKL-freeze-0.0005|70|39|55.7143|11.1342|1106.16|24.5132|1187.02|24.5131|0|0|0|0|
|humaneval|SureLock-gen-0.005|70|38|54.2857|8.48855|1091.31|25.5267|1171.08|25.5267|0|0|0|0|
|humaneval|SureLock-prompt-0.005|70|40|57.1429|28.4224|571.316|61.0122|613.091|61.0112|0|0|0|0|
|humaneval|TraceLock-0.95|70|37|52.8571|467.916|250.963|82.8738|269.298|82.8743|70|70|0|0.531626|
|math500|B0|54|5|9.25926|0|1464.34|0|1571.28|0|0|0|0|0|
|math500|Confidence-0.9|54|5|9.25926|122.759|659.532|54.9606|707.673|54.962|0|0|0|0|
|math500|RepKL-freeze-0.0005|54|4|7.40741|9.70912|1018.86|30.4218|1093.23|30.4242|0|0|0|0|
|math500|SureLock-gen-0.005|54|6|11.1111|6.76868|1025.81|29.9473|1100.69|29.9497|0|0|0|0|
|math500|SureLock-prompt-0.005|54|5|9.25926|27.5007|516.305|64.7415|554.046|64.7392|0|0|0|0|
|math500|TraceLock-0.95|54|6|11.1111|91.9887|745.004|49.1237|799.38|49.1255|54|54|0|1.57336|
|overall|B0|124|43|34.6774|0|1464.92|0|1571.96|0|0|0|0|0|
|overall|Confidence-0.9|124|44|35.4839|271.746|397.363|72.8748|426.374|72.8762|0|0|0|0|
|overall|RepKL-freeze-0.0005|124|43|34.6774|10.5161|1068.15|27.0853|1146.17|27.0862|0|0|0|0|
|overall|SureLock-gen-0.005|124|44|35.4839|7.74125|1062.79|27.451|1140.42|27.452|0|0|0|0|
|overall|SureLock-prompt-0.005|124|45|36.2903|28.0239|547.36|62.6356|587.378|62.634|0|0|0|0|
|overall|TraceLock-0.95|124|43|34.6774|208.128|466.11|68.182|500.14|68.1836|124|124|0|0.985285|
