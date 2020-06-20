
if [ $# -lt 3 ]; then
  echo -e "usage:\t bash run.sh [model_type] [cuda_num] [name] other_args(optional)"
  exit
fi

array=("$@")

model_type=$1
cuda_num=$2
model_name=$3.pth
log_name=$3.log
other_args=""

for ((i=3; i<$#; i++))
do
  other_args="$other_args ${array[$i]}"
done

python3 main.py --train $model_type --cuda $cuda_num --model_name $model_name --log_name $log_name $other_args
