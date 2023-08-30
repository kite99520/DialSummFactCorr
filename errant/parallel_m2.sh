dataset=DialogSum
errant_parallel -orig /home/gaomq/DialSummFactCorr/${dataset}/data/gector/test_origin.txt \
-cor /home/gaomq/DialSummFactCorr/${dataset}/data/gector/test_tgt.txt \
-out ${dataset}_test_ref_all_fact.m2 \
-merge all-merge \
-tok 