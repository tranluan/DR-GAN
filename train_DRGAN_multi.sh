python main_DR_GAN_multi.py --dataset VGG2 --batch_size 24 --is_train True --learning_rate 0.0002 --image_size 96  --gf_dim 32 --df_dim 32 --dfc_dim 320 --gfc_dim 320 --z_dim 20 --c_dim 3 --checkpoint_dir ./checkpoints/DRGAN_n4 --gpu 3 #2>&1 | tee -a ./checkpoints/pretrain/CASIA_MTPIE_64_96_32_320_32_320/run.log

