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

    # Normalize transition counts to probabilities
    for i in range(vocab_size):
        total_count = np.sum(transition_matrix[i])
        if total_count > 0:
            transition_matrix[i] /= total_count

    return word_to_index, transition_matrix, unk_index

# Example usage
word_to_index, transition_matrix, unk_index = create_markov_model("robert_frost.txt")

word1 = "<s>"
word2 = "two"
print(f"Transition probability from '{word1}' to '{word2}': {transition_matrix[word_to_index[word1], word_to_index.get(word2, unk_index)]}")
