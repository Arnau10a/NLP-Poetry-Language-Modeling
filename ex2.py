import numpy as np

def create_markov_model(filename):
    """
    Creates a 1st order Markov model from a text file using matrix representation.

    Args:
        filename: Path to the text file.

    Returns:
        A tuple containing:
            - word_to_index: Dictionary mapping words to unique integer indices.
            - transition_matrix: A 2D numpy array representing transition probabilities.
            - unk_index: The index assigned to the <UNK> token.
    """
    # Build word dictionary and assign unique indices
    word_to_index = {"<UNK>": 0, "<s>": 1, "</s>": 2}  # Include <s> and </s>
    index_to_word = ["<UNK>", "<s>", "</s>"]  # Reverse mapping
    word_count = len(word_to_index)

    with open(filename) as f:
        for line in f:
            tokens = line.rstrip().lower().split()
            if not tokens:
                continue
            for word in tokens:
                if word not in word_to_index:
                    word_to_index[word] = word_count
                    index_to_word.append(word)
                    word_count += 1

    # Initialize transition matrix with zeros
    vocab_size = len(word_to_index)
    transition_matrix = np.zeros((vocab_size, vocab_size))

    # Process text to build transition counts
    unk_index = word_to_index["<UNK>"]
    with open(filename) as f:
        for line in f:
            tokens = line.rstrip().lower().split()
            if not tokens:
                continue
            tokens = ['<s>'] + tokens + ['</s>']  # Add start and end tokens

            # Use loop variable for iteration
            for i in range(len(tokens) - 1):
                prev_word_index = word_to_index.get(tokens[i], unk_index)
                next_word_index = word_to_index.get(tokens[i + 1], unk_index)
                transition_matrix[prev_word_index, next_word_index] += 1

    # Apply add-1 (Laplace) smoothing and normalize transition counts to probabilities
    transition_matrix += 1  # Add 1 to all counts for smoothing
    for i in range(vocab_size):
        total_count = np.sum(transition_matrix[i])
        transition_matrix[i] /= total_count

    return word_to_index, transition_matrix, unk_index

def sentence_probability(sentence, word_to_index, transition_matrix, unk_index):
    """
    Calculates the probability of a given sentence using the Markov model.

    Args:
        sentence: The input sentence as a string.
        word_to_index: Dictionary mapping words to unique integer indices.
        transition_matrix: A 2D numpy array representing transition probabilities.
        unk_index: The index assigned to the <UNK> token.

    Returns:
        The probability of the sentence.
    """
    probability = 1.0  # Initialize with probability of 1
    tokens = sentence.rstrip().lower().split()

    # Handle empty sentence
    if not tokens:
        return 0.0

    # Get word indices and handle unseen words using unk_index
    for i in range(len(tokens) - 1):
        prev_word_index = word_to_index.get(tokens[i], unk_index)
        next_word_index = word_to_index.get(tokens[i + 1], unk_index)

        # Get the transition probability from the matrix
        transition_prob = transition_matrix[prev_word_index, next_word_index]
        print(f"Transition probability from '{tokens[i]}' to '{tokens[i + 1]}': {transition_prob}")
        # Update the sentence probability by multiplying transition probabilities
        probability *= transition_prob

    return probability

# Create the Markov model
word_to_index, transition_matrix, unk_index = create_markov_model("robert_frost.txt")

# Calculate the probability of a sentence
sentence = "<s> the young folk held some hope out to each other </s>"
probability = sentence_probability(sentence, word_to_index, transition_matrix, unk_index)
print(f"Probability of sentence '{sentence}': {probability}")
