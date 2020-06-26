# ADL Final Project - Shared Task Challenge

## Training

### Train Bert on DRCD dataset (from 2020 ADL hw2)

### Train the four models used in our final prediction

Train 4 models:
```
cd parent_sampler_longer
bash run.sh conv [cuda_num] parent_sampler_longer --train_data [train_data path] --dev_data [dev_data path] --dev_ref_file [dev_ref.csv file path] --epochs 30 --hw2_QA_bert [bert from above training] --kernel_size 7 --learning_rate 5e-6 --round 2000 --ratio 2.0
```
```
cd parent_sampler_meow
bash run.sh conv [cuda_num] parent_sampler_meow --train_data [train_data path] --dev_data [dev_data path] --dev_ref_file [dev_ref.csv file path] --epochs 30 --hw2_QA_bert [bert from above training] --kernel_size 7 --learning_rate 5e-6 --round 2000 --ratio 4.0
```
```
cd parent_sampler_higher_ratio
bash run.sh conv [cuda_num] parent_sampler_higher_ratio --train_data [train_data path] --dev_data [dev_data path] --dev_ref_file [dev_ref.csv file path] --epochs 30 --hw2_QA_bert [bert from above training] --kernel_size 7 --learning_rate 5e-6 --round 2000 --ratio 6.0
```
```
cd parent_sampler_ratio_8.0
bash run.sh conv [cuda_num] parent_sampler_ratio_8.0 --train_data [train_data path] --dev_data [dev_data path] --dev_ref_file [dev_ref.csv file path] --epochs 30 --hw2_QA_bert [bert from above training] --kernel_size 7 --learning_rate 5e-6 --round 2000 --ratio 8.0
```

After training, check the log for best null threshold for prediction (each model has its unique best null threshold)

## Testing

Change `null_threshold` in line 15 `test.py` (null threshold part) to the corresponding best null threshold for each models. Then, execute `python3 test.py [test file path]`.
