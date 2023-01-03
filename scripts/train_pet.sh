#train pet model
checkpoint_dir=../checkpoints/pet

mkdir -p $checkpoint_dir
mkdir -p $checkpoint_dir/warm_up
mkdir -p $checkpoint_dir/post_training

echo "Warm up the pet student"
CUDA_VISIBLE_DEVICES=0, python3 ../src/pet_train.py \
       	../data-bin/iwslt14.tokenized.de-en \
	--user-dir ../src \
	--arch pet_iwslt_de_en --share-decoder-input-output-embed \
	--task pet_translation \
   	--optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
	--lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
	--dropout 0.3 --weight-decay 0.0001 \
    	--criterion pet_warm_up_criterion \
       	--label-smoothing 0.1 --max-tokens 4096 --validate-interval 2 \
       	--eval-bleu --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' --eval-bleu-detok moses --eval-bleu-remove-bpe \
   	--best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
       	--save-dir "$checkpoint_dir/warm_up" \
       	--max-epoch 3 --activation-fn relu \
	--source-lang de --target-lang en

echo "Post training the model with KD"
CUDA_VISIBLE_DEVICES=0, python3 ../src/pet_train.py \
        ../data-bin/iwslt14.tokenized.de-en \
        --user-dir ../src \
        --arch pet_iwslt_de_en --share-decoder-input-output-embed \
        --task pet_translation \
        --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
        --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
        --dropout 0.3 --weight-decay 0.0001 \
        --criterion pet_pkd_criterion \
        --label-smoothing 0.1 --max-tokens 4096 --validate-interval 2 \
        --eval-bleu --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' --eval-bleu-detok moses --eval-bleu-remove-bpe \
        --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
	--finetune-from-model "$checkpoint_dir/warm_up/checkpoint2.pt" \
        --save-dir "$checkpoint_dir/post_training" \
        --patience 30 --max-epoch 800 --activation-fn relu \
        --source-lang de --target-lang en
