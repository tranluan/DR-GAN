python main_pretrain.py --dataset VGG2 --batch_size 256 --is_train True --learning_rate 0.045 --image_size 224  --gf_dim 32 --df_dim 32 --dfc_dim 320 --gfc_dim 320 --z_dim 20 --c_dim 3 --checkpoint_dir ./checkpoints/pretrain_mobilenetv2 --gpu 3 #2>&1 | tee -a ./checkpoints/pretrain/CASIA_MTPIE_64_96_32_320_32_320/run.log

