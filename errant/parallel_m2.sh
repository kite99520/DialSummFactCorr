dataset=SAMSum
errant_parallel -orig test/${dataset}_test_origin.txt \
-cor test/${dataset}_test_tgt.txt \
-out test/${dataset}_test_ref_all_fact.m2 \
-merge all-merge \
-tok 