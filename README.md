
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
| linear_ltf0501 | ratio=3.0, | dev f1: 0.89568 | linear is not good | 
| conv_retry | no_sampler, 20 epochs, lr=3e-5 | best loss: 0.19300 (**test f1: 0.93802**), best f1: 0.93791 (**test f1: 0.93386**) | improved preprocess |
| conv_hw2_retry | ratio=2.0, round=2000, bert.pth | best loss: 0.17028, best f1: 0.94039(**test f1: 0.94991**) | improved preprocess with bert |
| parent_sampler | ratio=2.0, round=2000, lr=3e-5, bert.pth, preprocess_parent | best loss: 0.05278, best f1:0.94723(**test f1: 0.95025**) | thres=0.3, preprocess_parent better, merge_5 |
| parent_sampler_higher_ratio | ratio=6.0, lr=5e-6, ditto | best loss: 0.04124, best f1: 0.96454 | thres=0.4, merge_4 |
| parent_sampler_longer | ratio=2.0, ditto | best loss: 0.03799, best f1: 0.96410 | thres=0.4, merge_4 |
| parent_sampler_meow | ratio=4.0, ditto | best loss: 0.04119, best f1: 0.96370 | thres=0.6, merge_4 |
| parent_sampler_ratio_8.0 | ratio=8.0, ditto | best loss: 0.04206, best f1: 0.96444 | thres=0.1, merge_4 |
| blending/merge_4_dev.csv | merge_4, use postprocess & blend | f1: 0.96589 | merge with best f1 pth |
| blending/merge_5_dev.csv | merge_5, ditto | f1: 0.96562 | merge with best f1 pth |
| blending/merge2_4_dev.csv | merge_4, use postprocess2 & blend2 | f1: 0.96980(**merge2_4_test.csv test f1: 0.97654**) | merge with best f1 pth |
| blending/merge2_5_dev.csv | merge_4 & 5, ditto | f1: 0.96931 | merge with best f1 pth |
| blending/merge2_4_dev_upd_postprocess.csv | merge_4, p2 & b2, fix UNK | f1: 0.97285(**test f1: ??**) | |
