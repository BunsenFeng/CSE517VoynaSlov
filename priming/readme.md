# Priming

## Data Setup
Modify the `DATA_DIR` variable in `config.py` to the directory where you store all the predicted MFC frames.

## Frame-setting results on engagement metrics (Table 3 in original paper)

### Pre-processing
Modify the `PREPROCESS_FLAG` variable in `reg.py` to `True`.

Under `priming` directory, run
```bash
python reg.py
```

The preprocessed regression feature file `regression_features.pkl` will be stored in `priming` folder.

### Evaluation
Modify the `PREPROCESS_FLAG` variable in `reg.py` to `False`.
Modify the `REG_FRAME_OWN_FLAG` variable in `reg.py` to `False`.

Under `priming` directory, run
```bash
python reg.py
```

The corresponding top-5 features alongside their linear regression weights will be printed to the standard output.


## Association between framing and media ownership (Figure 5 in original paper)
Modify the `PREPROCESS_FLAG` variable in `reg.py` to `False`.
Modify the `REG_FRAME_OWN_FLAG` variable in `reg.py` to `True`.

Under `priming` directory, run
```bash
python reg.py
```

The corresponding figure `frame_ownership.png` showing association between framing and media ownership will be stored in `priming` folder.

## Frame statistics of comments of independent and state-affiliated media for each frame used by posts (Figure 6 in original paper)
Modify the `PLOT_FLAG` variable in `reg.py` to `False`.

Under `priming` directory, run
```bash
python stats.py
```

The preprocessed statistics `final_stats_no_none.pkl` will be stored in `priming` folder.

Then, modify the `PLOT_FLAG` variable in `reg.py` to `True`.

Under `priming` directory, run
```bash
python stats.py
```

The corresponding figure `stats_no_none.png` showing frame statistics of comments will be stored in `priming` folder.

