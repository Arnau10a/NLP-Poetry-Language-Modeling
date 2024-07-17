# first collect the words from the text...
def add2dict(dictionary, key, value):
  if key not in dictionary:
    dictionary[key] = []
  dictionary[key].append(value)

# { (I, saw) : [cat, cat, dog, dog, dog, dog, dog, mouse, ...] ... }
second_order = {} # the 2nd order transition "tensor" of probabilities

for line in open('robert_frost.txt'): 
  tokens = line.rstrip().lower().split()
  if not tokens:
    continue

  tokens = ['<ss>', '<s>'] + tokens + ['</s>']

  T = len(tokens)
  for i in range(T-2):  # we build a second order Markov model, hence three steps in a row

      # measure all the words distribution given two previous words
      add2dict(second_order, (tokens[i], tokens[i+1]), tokens[i+2])

# inspect the dict/model structure
for key, value in second_order.items():
  print(key,':',value)