# CSE517VoynaSlov

## environment setup
```
git clone https://github.com/BunsenFeng/CSE517VoynaSlov.git
cd CSE517VoynaSlov
conda env create -f environment.yml -n VoynaSlov
```

## dataset download
Please refer to the original repository of this work: https://github.com/chan0park/VoynaSlov

## preprocessing, training, and evaluation
Please refer to the readme in each of the three folders.

## pretrained models
The trained MFC classifier (AutoModelForSequenceClassification) is available at: https://huggingface.co/bunsenfeng/mfc

## table of results
Performance of the MFC Classifier is as follows:

| Setting   | Data        | F1 (orig.) | F1 (ours) | diff. |
|-----------|-------------|------------|-----------|-------|
| in-domain | MFC         | 67.5       | 65.3      | -2.2  |
| zero-shot | immigration | 52.7       | 52.7      | 0.0   |
| zero-shot | same-sex    | 50.4       | 49.5      | -0.9  |
| zero-shot | tobacco     | 51.0       | 50.3      | -0.7  |
| zero-shot | VoynaSlov   | 33.5       | 33.9      | +0.4  |
