dataset=SAMSum
errant_compare -hyp test/${dataset}_test_hypo_all_fact.m2 \
 -ref test/${dataset}_test_ref_all_fact.m2 \
 -filt U:TrivE M:TrivE R:TrivE \
 -cat 2