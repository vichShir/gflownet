for ckpt in {100..5000..100}
do
    echo "Running checkpoint: $ckpt"
    ./gfn-env/bin/python eval.py \
        device=cpu \
        rundir=/home/victor/projects/sardinha/gflownet/logs/local/2026-11-06_18-01-48_988528 \
        n_samples=1000 \
        randominit=False \
        ckpt_id=$ckpt
done