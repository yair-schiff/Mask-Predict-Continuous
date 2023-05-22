# Continuous Mask-Predict

This repository implements the experiments from "Continuous Mask Predict: Enabling Better Control for Parallel Decoding
Iterative Refinement."

The code is based off of the implementation of [Mask-Predict: Parallel Decoding of Conditional Masked Language Models
](https://arxiv.org/abs/1904.09324) available [here](https://github.com/facebookresearch/Mask-Predict).

### Preprocess

text=PATH_YOUR_DATA

output_dir=PATH_YOUR_OUTPUT

src=source_language

tgt=target_language

model_path=PATH_TO_MASKPREDICT_MODEL_DIR

python preprocess.py --source-lang ${src} --target-lang ${tgt} --trainpref $text/train --validpref $text/valid --testpref $text/test  --destdir ${output_dir}/data-bin  --workers 60  --srcdict ${model_path}/maskPredict_${src}_${tgt}/dict.${src}.txt --tgtdict ${model_path}/maskPredict_${src}_${tgt}/dict.${tgt}.txt

### Train

#### Training Continuous Mask Predict
Use the [`run_train.sh`](./run_train.sh) script to launch training jobs on the G2 cluster.
For usage type:
```shell
run_train.sh --help
```

#### Training the distilled classifier
Schedule classifier training using
`sbatch train_classifier.sh`.

#### AR model training
Schedule AR training using
`sbatch train_AR.sh`.

### Evaluation

Use the [`run_generate.sh`] script to perform evaluation.
For usage type:
```shell
run_generate.sh --help
```

# License
MASK-PREDICT is CC-BY-NC 4.0.
The license applies to the pre-trained models as well.
