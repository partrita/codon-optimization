# Codon-optimization

Use machine learning to design codon-optimized DNA sequences for increased protein expression in CHO cells.

Description of project:

- https://www.liebertpub.com/doi/10.1089/cmb.2021.0458

To predict the 'best' DNA sequence for a given amino acid sequence using the pre-trained model, use rnn_predict.py.

Codon optimization procedure:
1) Ensure the following files are in the same directory:
   - rnn_predict.py
   - rnn_model.h5
   - dna_tokenizer.json
   - aa_tokenizer.json
   - Text file containing amino acid sequence (save as "sequence.txt")
     - Single-letter amino acid abbreviations
     - Signal peptide included, if applicable
     - No stop codon (will be added to output DNA sequence)

2) Run rnn_predict.py:
   ```
   python rnn_predict.py
   ```

3) The optimized DNA sequence is output as "sequence_opt.txt" in the same directory.

