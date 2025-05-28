# Use a trained recurrent neural network to codon optimize an amino acid sequence
# Predicts the 'best' DNA sequence for a given AA squence
# Based on Chinese hamster DNA and AA sequences
# Inputs: Trained RNN model (.h5), DNA and AA tokenizers (.json), and AA sequence to optimize (.txt)
# Formatting of AA sequence: Single-letter abbreviations, no spaces, no stop codon (this is added), include signal peptide
# Output: predicted/optimized DNA sequence (.txt)
# Dennis R. Goulet
# First upload to Github: 03 July 2020

import os
import numpy as np
import json
import click
import tensorflow as tf  # TensorFlow를 임포트합니다.
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model


# Encrypt the amino acid sequence
def encrypt(string, length):
    """
    아미노산 서열을 지정된 길이로 암호화합니다.
    예: "MAM" -> "M A M"
    """
    return " ".join(string[i : i + length] for i in range(0, len(string), length))


# Pad the amino acid sequence to the correct length (matching model)
def pad(x, length=None):
    """
    모델 입력에 맞게 시퀀스를 패딩합니다.
    """
    return pad_sequences(x, maxlen=length, padding="post")


# Combine tokenization and padding
def preprocess(x, aa_tokenizer):
    """
    아미노산 시퀀스를 토큰화하고 패딩합니다.
    """
    preprocess_x = aa_tokenizer.texts_to_sequences(x)
    preprocess_x = pad(preprocess_x)
    return preprocess_x


# Transform tokens back to DNA sequence
def logits_to_text(logits, tokenizer):
    """
    모델의 로짓(예측)을 DNA 서열 텍스트로 변환합니다.
    """
    index_to_words = {id: word for word, id in tokenizer.word_index.items()}
    return " ".join([index_to_words[prediction] for prediction in np.argmax(logits, 1)])


@click.command()
@click.option(
    "-i",
    "--input_aa_file",
    required=True,
    help="최적화할 아미노산 서열이 담긴 텍스트 파일 경로 (예: sequence.txt)",
)
@click.option(
    "-o",
    "--output_dna_file",
    default="optimized_dna.txt",
    help="최적화된 DNA 서열을 저장할 파일 경로 (기본값: optimized_dna.txt)",
)
@click.option(
    "--model_path",
    default="rnn_model.h5",
    help="훈련된 RNN 모델 파일 경로 (기본값: rnn_model.h5)",
)
@click.option(
    "--aa_tokenizer_path",
    default="aa_tokenizer.json",
    help="아미노산 토크나이저 파일 경로 (기본값: aa_tokenizer.json)",
)
@click.option(
    "--dna_tokenizer_path",
    default="dna_tokenizer.json",
    help="DNA 토크나이저 파일 경로 (기본값: dna_tokenizer.json)",
)
def main(
    input_aa_file, output_dna_file, model_path, aa_tokenizer_path, dna_tokenizer_path
):
    """
    훈련된 순환 신경망(RNN)을 사용하여 아미노산 서열을 코돈 최적화합니다.
    """
    # Resolve input and output file paths to be absolute BEFORE changing directory.
    # This ensures that the script can find these files regardless of the current working directory
    # after os.chdir().
    # 입력 및 출력 파일 경로를 디렉토리 변경 전에 절대 경로로 변환합니다.
    # 이는 os.chdir() 호출 후에도 스크립트가 이 파일들을 찾을 수 있도록 보장합니다.
    absolute_input_aa_file = os.path.abspath(input_aa_file)
    absolute_output_dna_file = os.path.abspath(output_dna_file)

    click.echo(f"아미노산 서열 파일: {absolute_input_aa_file}")
    click.echo(f"결과 DNA 서열 파일: {absolute_output_dna_file}")

    # Change current working directory to the script's directory.
    # This helps in resolving paths for model and tokenizer files which are typically
    # located relative to the script itself.
    # 현재 스크립트가 있는 디렉토리로 이동합니다.
    # 이는 일반적으로 스크립트 자체에 상대적으로 위치하는 모델 및 토크나이저 파일의 경로를
    # 해결하는 데 도움이 됩니다.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    # Load the amino acid sequence to optimize
    # 최적화할 아미노산 서열을 불러옵니다.
    try:
        with open(absolute_input_aa_file, "r") as f:
            aa_item = f.read()
    except FileNotFoundError:
        click.echo(
            f"오류: 입력 아미노산 파일 '{absolute_input_aa_file}'을 찾을 수 없습니다. 경로를 확인해주세요."
        )
        return

    # Load the trained model
    # 훈련된 모델을 불러옵니다.
    try:
        # GRU 레이어를 명시적으로 custom_objects로 전달하여 로드 오류를 방지합니다.
        # Pass GRU layer explicitly as custom_objects to prevent loading errors.
        model = load_model(model_path, custom_objects={"GRU": tf.keras.layers.GRU})
    except Exception as e:
        click.echo(f"오류: 모델 파일 '{model_path}'을 불러오는 데 실패했습니다: {e}")
        return

    # Load tokenizers
    # 토크나이저를 불러옵니다.
    try:
        with open(aa_tokenizer_path, "r") as f:
            aa_json = json.load(f)
        aa_tokenizer = tokenizer_from_json(aa_json)

        with open(dna_tokenizer_path, "r") as f:
            dna_json = json.load(f)
        dna_tokenizer = tokenizer_from_json(dna_json)
    except FileNotFoundError as e:
        click.echo(
            f"오류: 토크나이저 파일 중 하나를 찾을 수 없습니다: {e}. 경로를 확인해주세요."
        )
        return
    except json.JSONDecodeError as e:
        click.echo(f"오류: 토크나이저 파일의 JSON 형식이 올바르지 않습니다: {e}")
        return

    # Preprocess the amino acid sequence
    # 아미노산 서열을 전처리합니다.
    aa_item_with_stop = aa_item + "Z"  # Add stop codon identifier
    aa_item_with_stop = (
        aa_item_with_stop.replace(" ", "")
        .replace("\n", "")
        .replace(" ", "")
        .replace("\r", "")
        .replace("\t", "")
    )
    aa_list = [aa_item_with_stop]
    seq_len = len(aa_item_with_stop)

    aa_spaces = []
    for aa_seq in aa_list:
        aa_current = encrypt(aa_seq, 1)
        aa_spaces.append(aa_current)

    preproc_aa = preprocess(aa_spaces, aa_tokenizer)
    # Pad and reshape to match model input dimension
    # 모델 입력 차원에 맞게 패딩하고 형태를 변경합니다.
    tmp_x = pad(preproc_aa, 8801)
    tmp_x = tmp_x.reshape((-1, 8801))

    # Predict DNA sequence
    # DNA 서열을 예측합니다.
    click.echo("DNA 서열 예측 중...")
    seq_opt = logits_to_text(model.predict(tmp_x[:1])[0], dna_tokenizer)
    seq_opt_removepad = seq_opt[: (seq_len * 4)]  # Remove padding
    seq_opt_removespace = seq_opt_removepad.replace(" ", "")  # Remove spaces
    seq_opt_final = seq_opt_removespace.upper()  # Convert to uppercase

    # Output optimized DNA sequence and save to file
    # 최적화된 DNA 서열을 출력하고 파일에 저장합니다.
    click.echo("\n최적화된 DNA 서열:")
    click.echo(seq_opt_final)

    try:
        with open(absolute_output_dna_file, "w") as f:
            f.write(seq_opt_final)
        click.echo(
            f"\n최적화된 DNA 서열이 '{absolute_output_dna_file}'에 저장되었습니다."
        )
    except Exception as e:
        click.echo(f"오류: 최적화된 DNA 서열을 파일에 저장하는 데 실패했습니다: {e}")


if __name__ == "__main__":
    main()
