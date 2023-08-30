import subprocess
import os


if __name__ == '__main__':


    dir_list = []
    # parent_dir = '/home/gaomq/DialSummFactCorr/Baseline_bart'
    parent_dir = '/home/gaomq/DialSummFactCorr/factpegasus'
    dataset = 'DialogSum'
    for x in os.listdir(parent_dir):
        dir_path = os.path.join(parent_dir, x)
        pred_output = os.path.join(dir_path, '{}_real.txt'.format(dataset))
        eval_output = os.path.join(dir_path, '{}_test_hypo_all_fact.m2'.format(dataset))
        if os.path.exists(pred_output) and not os.path.exists(eval_output):
        # if os.path.exists(pred_output):
            dir_list.append(dir_path)

    for outdir in dir_list:
        
        p = subprocess.Popen("""
        errant_parallel -orig /home/gaomq/DialSummFactCorr/{}/data/gector/test_origin.txt \
        -cor {}/{}_real.txt \
        -out {}/{}_test_hypo_all_fact.m2 \
        -merge all-merge \
        -tok
        """.format(dataset, outdir, dataset, outdir, dataset), shell=True)
        p.wait()
        
        p = subprocess.Popen("""
        errant_compare -hyp {}/{}_test_hypo_all_fact.m2 \
        -ref {}_test_ref_all_fact.m2 \
        -filt U:TrivE M:TrivE R:TrivE \
        -cat 2 > {}/errant_all_result_2_{}.txt
        """.format(outdir, dataset, dataset, outdir, dataset), shell=True)
        p.wait()
    
        p = subprocess.Popen("""
        errant_compare -hyp {}/{}_test_hypo_all_fact.m2 \
        -ref {}_test_ref_all_fact.m2 \
        -filt U:TrivE M:TrivE R:TrivE \
        -cat 1 > {}/errant_all_result_1_{}.txt
        """.format(outdir, dataset, dataset, outdir, dataset), shell=True)
        p.wait()
    
