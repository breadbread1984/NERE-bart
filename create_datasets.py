#!/usr/bin/python3

from datasets import load_dataset, Split

def load_docred():
  trainset = load_dataset('thunlp/docred', split = 'train_annotated', trust_remote_code = True)
  valset = load_dataset('thunlp/docred', split = 'validation', trust_remote_code = True)
  return trainset, valset

if __name__ == "__main__":
  trainset, valset = load_docred()
  print(trainset)
