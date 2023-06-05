from transformers import Wav2Vec2Processor, AutoModelForCTC, Wav2Vec2CTCTokenizer
import argparse
import torch
import librosa
import logging
import evaluate
import glob
import datasets
from datasets import load_dataset, load_metric

parser = argparse.ArgumentParser()
parser.add_argument('--filename', type=str, default=None, help='The file that you want to transcribe')
parser.add_argument('--split_len', type=int, default=100000, help='Length of the segments that you want to recognize')
parser.add_argument('--decode', action='store_false')
parser.add_argument('--evaluate', action='store_true')
parser.add_argument('--evaluate_lang', type=str, default='both', help='Language to use for decoding. Options:[both, it, en]')
parser.add_argument('--model', type=str, default="./wav2vec2-common_voice-it_en/checkpoint-25200")
#./wav2vec2-common_voice-it_en-demo/checkpoint-7200
parser.add_argument('--dec_folder', action='store_true', help='whether to decode the current folder or the test set')
args = parser.parse_args([] if "__file__" not in globals() else None)

logger = logging.getLogger(__name__)

processor = Wav2Vec2Processor.from_pretrained(args.model)
tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(args.model)
model = AutoModelForCTC.from_pretrained(args.model)

if args.evaluate or not args.dec_folder:
  if args.evaluate_lang == 'both':
    it_dataset = load_dataset("audiofolder", data_dir="my_dataset_it")
    en_dataset = load_dataset("audiofolder", data_dir="my_dataset_en")
    dataset = datasets.concatenate_datasets([it_dataset['test'], en_dataset['test']])
  elif args.evaluate_lang == 'it':
    dataset = load_dataset("audiofolder", data_dir="my_dataset_it")['test']
  else:
    dataset = load_dataset("audiofolder", data_dir="my_dataset_en")['test']

  def prepare_dataset(batch):
    audio = batch["audio"]

    # batched output is "un-batched" to ensure mapping is correct
    batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]

    with processor.as_target_processor():
      batch["labels"] = processor(batch["text"]).input_ids
    return batch

  dataset = dataset.map(prepare_dataset, num_proc=4)

  def map_to_result(batch):
    with torch.no_grad():
      input_values = torch.tensor(batch["input_values"]).unsqueeze(0)
      logits = model(input_values).logits

    pred_ids = torch.argmax(logits, dim=-1)
    batch["pred_str"] = processor.batch_decode(pred_ids)[0]
    batch["text"] = processor.decode(batch["labels"], group_tokens=False)

    return batch

  results = dataset.map(map_to_result)

  if not args.dec_folder:
    for r in results:
      print(f'prediction: {r["pred_str"]}')
      print(f'label: {r["text"]}')
      print('###################################################')

  if args.evaluate:
    # Computing the test error
    wer_metric = evaluate.load("wer")
    print("Test WER: {:.3f}".format(wer_metric.compute(predictions=results["pred_str"], references=results["text"])))
    cer_metric = evaluate.load("cer")
    print("Test CER: {:.3f}".format(cer_metric.compute(predictions=results["pred_str"], references=results["text"])))


if args.dec_folder:
  path = glob.glob('*.wav')

  for p in path:
    global_transcr = ''

    # Load the audio with the librosa library
    input_audio, _ = librosa.load(p, sr=16000)

    print(f'Transcribing: {p}')
    # Splitting the input audio in smaller frames to avoid too large attention (it increases quadratically with the input size)
    splits = []
    for i in range(0, len(input_audio), args.split_len):
      splits.append(input_audio[i: args.split_len*(i+1)])

    for s in splits:
      # Tokenize the audio
      input_values = processor(s, return_tensors="pt", padding="longest", sampling_rate=16000).input_values

      # Feed it through Wav2Vec & choose the most probable tokens
      with torch.no_grad():
        logits = model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)

      # Decode & add to our caption string
      transcription = tokenizer.batch_decode(predicted_ids)[0]
      global_transcr += transcription + " "

    print(global_transcr)
