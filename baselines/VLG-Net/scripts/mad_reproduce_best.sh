output_dir=./models/mad_best_model/
CONFIG=$output_dir/config.yml
CKPT=$output_dir/best_checkpoint.pth

python test_net.py --config-file $CONFIG \
                    --split test \
                    --ckpt $CKPT \
                    SOLVER.BATCH_SIZE 512 \
                    TEST.STRIDE 64 \
                    OUTPUT_DIR $output_dir \
                    TEST.NMS_THRESH 0.3