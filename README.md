# Before you begin

This is fork repo of [Codon-optimization](https://github.com/drgoulet/Codon-optimization)

## citation

```
Codon Optimization Using a Recurrent Neural Network
Dennis R. Goulet, Yongqi Yan, Palak Agrawal, Andrew B. Waight, Amanda Nga-sze Mak, and Yi Zhu
Published Online:21 Jun 2022 https://doi.org/10.1089/cmb.2021.0458
```

# how to use

I tested at python 3.10

## 1. clone this repo

## 2. make virtual enviroment

```bash
$python3 -m venv venv
$source venv/bin/activate
(venv)$ pip install -r requirements.txt
```

## 3. edit sequence file

It should be amino acid code only, no stop codon(`*`), no line break(`\n`).

```bash
vim sequence.txt
```

## 4. run `rnn_predict.py`

```bash
$ python rnn_predict.py 
2022-09-07 05:24:10.133752: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcudart.so.11.0'; dlerror: libcudart.so.11.0: cannot open shared object file: No such file or directory
2022-09-07 05:24:10.133794: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
2022-09-07 05:24:22.256539: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcuda.so.1'; dlerror: libcuda.so.1: cannot open shared object file: No such file or directory
2022-09-07 05:24:22.256586: W tensorflow/stream_executor/cuda/cuda_driver.cc:269] failed call to cuInit: UNKNOWN ERROR (303)
2022-09-07 05:24:22.256626: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:156] kernel driver does not appear to be running on this host (codespaces-bb4c72): /proc/driver/nvidia/version does not exist
2022-09-07 05:24:22.257099: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 AVX512F FMA
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
1/1 [==============================] - 2s 2s/step
Optimized DNA sequence:
TCTGACACAGGCAGACCCTTTGTGGAGATGTACTCAGAAATCCCAGAAATCATCCACATGACAGAAGGCAGAGAGCTGGTGATCCCCTGCAGAGTGACCTCTCCCAACATCACTGTGACTCTGAAGAAGTTTCCTCTGGACACTCTGATTCCTGATGGAAAGAGAATCATCTGGGACAGCAGGAAGGGCTTCATCATCAGCAATGCCACCTACAAGGAAATTGGACTGCTGACCTGTGAAGCCACTGTGAATGGCCACCTCTACAAGACCAACTACCTCACTCACAGACAGACCAACACCATCATTGATGTGGTGCTGTCTCCTAGCCATGGCATTGAGCTGTCTGTGGGAGAGAAGCTGGTGCTGAACTGCACAGCCAGAACAGAACTGAATGTGGGAATTGACTTCAACTGGGAATACCCTAGCAGCAAGCACCAGCACAAGAAGCTGGTGAACAGAGACCTGAAGACACAGAGTGGCAGTGAGATGAAGAAGTTCCTGAGCACACTGACCATTGATGGAGTGACCAGAAGTGACCAGGGCCTCTACACCTGTGCAGCCTCCTCTGGCCTGATGACCAAGAAGAACAGCACCTTTGTGAGAGTTCATGAGAAGGACAAGACACACACCTGTCCTCCCTGCCCTGCCCCAGAGCTGCTGGGAGGCCCCTCTGTGTTCCTGTTTCCTCCCAAGCCCAAGGACACACTGATGATCAGCAGGACACCAGAAGTGACCTGTGTGGTGGTGGATGTGAGCCATGAAGACCCAGAAGTGAAGTTCAACTGGTATGTGGATGGAGTGGAAGTGCACAATGCCAAGACCAAGCCCAGAGAAGAGCAGTACAACAGCACCTACAGAGTGGTGTCTGTGCTGACTGTGCTGCACCAGGACTGGCTGAATGGAAAGGAATACAAGTGCAAGGTGAGCAACAAGGCCCTGCCAGCTCCCATTGAGAAGACCATCAGCAAGGCCAAGGGACAGCCCAGAGAGCCCCAGGTGTACACCCTGCCTCCCAGCAGAGATGAGCTGACCAAGAACCAGGTGTCCCTGACCTGCCTGGTGAAGGGCTTCTACCCCTCAGACATTGCTGTGGAGTGGGAGAGCAATGGACAGCCAGAGAACAACTACAAGACCACACCTCCTGTGCTGGACTCTGATGGCAGCTTCTTCCTGTACAGCAAGCTGACTGTGGACAAGAGCAGGTGGCAGCAGGGCAATGTGTTCTCCTGCTCTGTGATGCATGAAGCCCTGCACAACCACTACACACAGAAGAGCCTGAGCCTGTCTCCTGGCTGA
```

## 5. results are saved in `sequence_opt.txt` file

:smile:

# codon-optimization

Use machine learning to design codon-optimized DNA sequences for increased protein expression in CHO cells.
To predict the 'best' DNA sequence for a given amino acid sequence using the pre-trained model, use rnn_predict.py.

Packages required:
- Python (built using Python 3.7.4 via Ubuntu-18.04)
- Keras (tensorflow)
- json
- NumPy
- OS

Codon optimization procedure:
1) Place the following in the same directory:
- rnn_predict.py
- rnn_model.h5
- dna_tokenizer.json
- aa_tokenizer.json
- Text file containing amino acid sequence (save as "sequence.txt")
  - Single-letter amino acid abbreviations
  - Signal peptide included, if applicable
  - No stop codon (will be added to output DNA sequence)

2) In line 19 of rnn_predict.py, change to the working directory containing the above files.
3) Run rnn_predict.py.
4) The optimized DNA sequence is output as "sequence_opt.txt" in the same directory. 

