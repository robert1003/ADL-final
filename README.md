
| folder | important hyper | info | comment |
| - | - | - | - |
| conv_robert_1 | kernel_size=5 | conv_dev.csv: 0.9218325073071548 | |
| conv_robert_2 | kernel_size=7 | conv_dev.csv: 0.9276401067479988, 30_conv_dev.csv: 0.9281865548354303 | larger kernel_size seems to work better, and pocket with CE loss is not good |
| conv_robert_11 | kernel_size=9 | conv_dev.csv: 0.9294573643410853 | |
| conv_robert_12 | kernel_size=11 | conv_dev.csv: 0.9289236243487101 | |
| conv_sampler_robert_7 | ratio=1.0 round=2000 | conv_sampler_dev.csv: 0.929838607192782 | |
| conv_sampler_robert_8 | ratio=2.0 round=2000 | conv_sampler_dev.csv: 0.9331681280975983 | |
| conv_sampler_robert_5 | ratio=3.0 round=2000 | conv_sampler_dev.csv: 0.9312364976490027 | |
| conv_sampler_robert_6 | ratio=4.0 round=2000 | conv_sampler_dev.csv: 0.9323929342991488 | | 
| conv_sampler_robert_4 | ratio=5.0 round=2000 | conv_sampler_dev.csv: 0.9320116914474523 | different ratios didn't have much difference except ratio=1.0. this might implies that we may not need sampler and model probably will not suffer from imbalanced data |
| conv_from_hw2_9 | bert=hw2(trained 10 epochs) ks=7 | conv_hw2_dev.csv: 0.9314398271699075, **conv_hw2_test.csv(pub): 0.93523** | loss decreases faster(dev_loss 0.21~ in 1200 steps |
| conv2_robert_3 | Conv1d: 768-786-1, relu | conv2_dev.csv: 0.9210700216037618, 30_conv2_dev.csv: 0.9288727919684839 | deeper conv layer didn't improve, and pocket with CE loss is not good |
| conv2_robert_13 | Conv1d: 786-30-1, relu | conv2_dev.csv: 0.9312401285333047 | |
| conv_distilbert_robert_15 | bert=distilbert-base-multilingual-cased | conv_dis_dev.csv: 0.9303977633752702 | | 
| conv_ltf0501 | kernel_size=7, ratio=3.0 | dev f1: 0.92911 | transfer learning from hw2 |
| conv_deep_ltf0501 | kernel_size=7, ratio=3.0, | dev f1: 0.92379 | transfer learning from hw2, CNN: 256 -> 64 -> 1 | 
