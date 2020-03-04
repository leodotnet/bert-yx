export GLUE_DIR='/home/yxzhou/experiments/bert/data'
export BERT_DIR='/home/yxzhou/experiments/bert_pretrained/uncased_L-12_H-768_A-12'
export OUTPUT_DIR='/home/yxzhou/experiments/bert/output'



CUDA_VISIBLE_DEVICES=1 python run_classifier.py \
  --task_name MRPC \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir $GLUE_DIR/MRPC/ \
  --bert_model $BERT_DIR \
  --max_seq_length 128 \
  --train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 9.0 \
  --output_dir $OUTPUT_DIR/mrpc_output_2/