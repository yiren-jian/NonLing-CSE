# Training logs of VisualCSE

Here we provide an example of our training logs.
```bash
nohup bash bash run_unsup_example.sh 48 0.07 1e-6 0.05 SupCon 0.0 &> unsup-VisualCSE-bert-base-uncased.txt &
```

Instead of opening `txt` by a text editor (too many lines by `tqdm`), it is best viewed by opening a Terminal and type `cat unsup-VisualCSE-bert-base-uncased.txt` to visualize the training log.
