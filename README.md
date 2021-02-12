# Document Information Extraction 

This project is the final project of course ADL (Applied Deep Learning). My teammates and I constructed machine learning model(use BERT as a part) which extract set of tags from a Japanese bidding document, trained only on a little amount of dataset(about a hundred), and won the contest with a large gap.

## Project detail

* [Link](https://docs.google.com/presentation/d/1KYhOBy_xWjDR3iMWeB0GMMoVvRm-w0ygURxE5qnlPpc/edit#slide=id.p) to project slide.
* [Link](https://drive.google.com/file/d/13vOp9SWnpVd-wlx3arKLBeBb_tCF7h8P/view?usp=sharing) to dataset.
* [Link](https://www.kaggle.com/c/adl-final-project-shared-task-108-spring/leaderboard) to Kaggle competition website.

## Our performance

Ranked 1 in private dataset with score 0.97904

## Execution detail

### Training

#### Train Bert on DRCDv2 dataset (from 2020 ADL hw2)

First, prepare DRCDv2 dataset (the data used in ADL hw2). You can choose to download it [here](https://drive.google.com/drive/folders/1HTdj80dj3zFFJliv1EBoHgX-IEeP_4pQ?usp=sharing). Then, execute the following command:
```
python3.6 hw2/train.py [train_data_path]
```
The BERT model is stored at `hw2/bert.pth`.

#### Train the four models used in our final prediction

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

### Testing

Change `null_threshold` in line 15 `test.py` (null threshold part) to the corresponding best null threshold for each models. Then, execute `python3 test.py [test file path]`.
