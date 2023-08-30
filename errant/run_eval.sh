# DATASET=DialogSum
# MODEL_OUT=/home/gaomq/DialSummFactCorr/Baseline_bart/BART-LARGE-DialogSum_real-finetune_DialogSum_useref-devloss

# errant_parallel -orig /home/gaomq/DialSummFactCorr/${DATASET}/data/gector/test_origin.txt \
# -cor ${MODEL_OUT}/${DATASET}_real.txt \
# -out ${MODEL_OUT}/test_hypo_all_fact.m2 \
# -merge all-merge \
# -tok 

# errant_compare -hyp ${MODEL_OUT}/test_hypo_all_fact.m2 \
#  -ref ${DATASET}_test_ref_all_fact.m2 \
#  -filt U:TrivE M:TrivE R:TrivE \
#  -cat 2



# DATASET=SAMSum
# MODEL_OUT=/home/gaomq/DialSummFactCorr/gector/ROBERTA-BASE-SAMSum_mix_1_1/output

# errant_parallel -orig /home/gaomq/DialSummFactCorr/${DATASET}/data/gector/test_origin.txt \
#  -cor ${MODEL_OUT}/${DATASET}_real_itr1.txt \
#  -out ${MODEL_OUT}/test_hypo_all_fact.m2 \
#  -merge all-merge \
#  -tok 

# errant_compare -hyp ${MODEL_OUT}/test_hypo_all_fact.m2 \
#   -ref ${DATASET}_test_ref_all_fact.m2 \
#   -filt U:TrivE M:TrivE R:TrivE \
#   -cat 2


DATASET=SAMSum
# MODEL_OUT=/home/gaomq/DialSummFactCorr/Baseline_bart/BART-LARGE-${DATASET}-retrainFactEdit-devloss
MODEL_OUT=/home/gaomq/DialSummFactCorr/faithful_summarization/BART-BASE_${DATASET}
errant_parallel -orig /home/gaomq/DialSummFactCorr/${DATASET}/data/gector/test_origin.txt \
 -cor ${MODEL_OUT}/${DATASET}_real.txt \
 -out ${MODEL_OUT}/${DATASET}_test_hypo_all_fact.m2 \
 -merge all-merge \
 -tok 

errant_compare -hyp ${MODEL_OUT}/${DATASET}_test_hypo_all_fact.m2 \
  -ref ${DATASET}_test_ref_all_fact.m2 \
  -filt U:TrivE M:TrivE R:TrivE \
  -cat 1 > /home/gaomq/DialSummFactCorr/faithful_summarization/BART-BASE_SAMSum/errant_all_result_1_${DATASET}.txt


errant_compare -hyp ${MODEL_OUT}/${DATASET}_test_hypo_all_fact.m2 \
  -ref ${DATASET}_test_ref_all_fact.m2 \
  -filt U:TrivE M:TrivE R:TrivE \
  -cat 2 > /home/gaomq/DialSummFactCorr/faithful_summarization/BART-BASE_SAMSum/errant_all_result_2_${DATASET}.txt

