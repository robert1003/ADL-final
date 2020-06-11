
| folder | important hyper | info | comment |
| - | - | - | - |
| conv_robert_1 | kernel_size=5 | conv_dev.csv: 0.9218325073071548 | |
| conv_robert_2 | kernel_size=7 | conv_dev.csv: 0.9276401067479988, 30_conv_dev.csv: 0.9281865548354303 | larger kernel_size seems to work better, and pocket with CE loss is not good |
| conv2_robert_3 | stack two Conv1d layer | conv2_dev.csv: 0.9210700216037618, 30_conv2_dev.csv: 0.9288727919684839 | deeper conv layer didn't improve, and pocket with CE loss is not good |
| conv_sampler_robert_5 | ratio=3.0 round=2000 | conv_sampler_dev.csv: 0.9312364976490027 | |


