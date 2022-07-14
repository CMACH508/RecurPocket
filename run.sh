batch_size=10
output="./checkpoint/seg0"
txt="./checkpoint.out"
gpu="2,3"
ite=1
epoch=100
is_cao=4
nohup python -u train_segmentation.py --is_cao $is_cao --iteration $ite --gpu $gpu -b $batch_size -o $output -e $epoch > $txt 2>&1 &
