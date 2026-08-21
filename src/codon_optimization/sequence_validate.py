# Purpose: Import fasta file containing list of DNA sequence, verify DNA and corresponding AA sequences, output verified sequences as json.
# Dennis R. Goulet
# 17 May 2020

import json
import random
from collections import OrderedDict

import numpy as np
import pandas as pd


def read_fasta_sequences(path):
    """Yield FASTA sequence strings without requiring an XML-capable dependency."""
    sequence_parts = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if sequence_parts:
                    yield "".join(sequence_parts).upper()
                    sequence_parts = []
                continue
            sequence_parts.append(line)
    if sequence_parts:
        yield "".join(sequence_parts).upper()


# Initialize lists and define acceptable DNA bases and amino acids.
dna_seq = []
dna_seq_new = []
aa_seq_new = []
bases = ['A', 'C', 'G', 'T']
residues = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', 'Z']

# Initialize list of tests to verify properties of the DNA and AA sequences.
test_total = [0, 0, 0, 0, 0, 0, 0]

# Import starting list of DNA sequences from fasta file containing human CDS sequences.
dna_seq.extend(read_fasta_sequences('cds_hamster.fna'))

# Verify properties of DNA sequences and their resulting AA sequences.
# If all tests pass (0), DNA and AA sequences are inserted into matching indices of two new lists.
for sequence in dna_seq:

    # Re-initialize test scores.
    test = [1, 0, 1, 1, 1, 0, 1]

    # The first test verifies that the length of the DNA sequence is divisible by 3, indicating it can be translated.
    if len(sequence) % 3 == 0:
        test[0] = 0

        # The second test verifies that the DNA sequence only contains the 4 standard DNA bases.
        for base in sequence:
            if base not in bases:
                test[1] = 1

        # If both of the DNA sequence tests pass, the DNA sequence is translated to AA sequence.
        codon_table = {
            'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
            'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
            'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*',
            'TGT': 'C', 'TGC': 'C', 'TGA': '*', 'TGG': 'W',
            'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
            'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
            'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
            'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
            'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
            'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
            'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
            'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
            'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
            'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
            'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
            'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G',
        }
        single_aa_seq_pre = ''.join(codon_table[sequence[i:i + 3]] for i in range(0, len(sequence), 3))
        single_aa_seq = single_aa_seq_pre.replace('*', 'Z')

        # The third test verifies that the AA sequence begins with Met.
        if single_aa_seq[0] == 'M':
            test[2] = 0

        # The fourth test verifies that the AA sequence ends with a stop codon (*).
        if single_aa_seq[-1] == 'Z':
            test[3] = 0

        # The fifth test verifies that the AA sequence only contains a single stop codon.
        if single_aa_seq.count('Z') == 1:
            test[4] = 0

        # The sixth test verifies that the AA sequence contains only the standard 20 AAs, plus stop (*).
        for aa in single_aa_seq:
            if aa not in residues:
                test[5] = 1

        # The seventh test verifies that dna_len = 3*aa_len
        if len(sequence) == 3 * len(single_aa_seq):
            test[6] = 0

    # The cumulative number of times each test failed is recorded and output during each iteration.
    test_total = [test_total[i] + test[i] for i in range(len(test_total))]
    print(test_total)

    # If all 7 tests succeed, the DNA sequence and corresponding AA sequence are added to new lists.
    if test == [0, 0, 0, 0, 0, 0, 0]:
        dna_seq_new.append(str(sequence))
        aa_seq_new.append(str(single_aa_seq))

# Shuffle items in the dictionary
seq_dict = {dna_seq_new[i]: aa_seq_new[i] for i in range(len(dna_seq_new))}
items = list(seq_dict.items())
random.shuffle(items)
dict_shuff = OrderedDict(items)

# Write to file
with open('cho_dict.json', 'w', encoding='utf-8') as f:
    json.dump(dict_shuff, f)
