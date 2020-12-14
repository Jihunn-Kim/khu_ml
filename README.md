# khu_ml
- SGD(LR=0.1), MOMENTUM=0.9, WEIGHTDECAY=1e-4, BATCHSIZE=128, EPOCHS=300
- RandomCrop(32, padding=4), RandomHorizontalFlip, Cutout
- LABELSMOOTH, Mish Activation
- train, valid 데이터 전부 사용
- 제출 시 train 정확도 93.5 / test 정확도 91.7
- 추가 100 epoch 학습 시 train 정확도 94.1 / test 정확도 91.8