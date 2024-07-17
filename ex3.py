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

    tokens = ['<s>'] + tokens + ['</s>']  # Add start and end tokens

    # Get word indices and handle unseen words using unk_index
    for i in range(len(tokens) - 1):
        prev_word_index = word_to_index.get(tokens[i], unk_index)
        next_word_index = word_to_index.get(tokens[i + 1], unk_index)

        # Get the transition probability from the matrix
        transition_prob = transition_matrix[prev_word_index, next_word_index]
        
        # Update the sentence probability by multiplying transition probabilities
        probability *= transition_prob

    return probability

def calculate_perplexity(sentence, word_to_index, transition_matrix, unk_index):
    """
    Calculates the perplexity of a given sentence using the Markov model.

    Args:
        sentence: The input sentence as a string.
        word_to_index: Dictionary mapping words to unique integer indices.
        transition_matrix: A 2D numpy array representing transition probabilities.
        unk_index: The index assigned to the <UNK> token.

    Returns:
        The perplexity of the sentence.
    """
    tokens = sentence.rstrip().lower().split()
    tokens = ['<s>'] + tokens + ['</s>']  # Add start and end tokens

    probability_product = 1.0
    n = len(tokens) - 1

    for i in range(n):
        prev_word_index = word_to_index.get(tokens[i], unk_index)
        next_word_index = word_to_index.get(tokens[i + 1], unk_index)
        transition_prob = transition_matrix[prev_word_index, next_word_index]
        probability_product *= (1 / transition_prob)

    perplexity = probability_product ** (1 / n)
    return perplexity

def find_min_max_perplexity_sentences(filename, word_to_index, transition_matrix, unk_index):
    """
    Finds the sentences with the minimum and maximum perplexity from the corpus.

    Args:
        filename: Path to the text file.
        word_to_index: Dictionary mapping words to unique integer indices.
        transition_matrix: A 2D numpy array representing transition probabilities.
        unk_index: The index assigned to the <UNK> token.

    Returns:
        A tuple containing:
            - Sentence with minimum perplexity
            - Sentence with maximum perplexity
            - Minimum perplexity value
            - Maximum perplexity value
    """
    min_perplexity = float('inf')
    max_perplexity = float('-inf')
    min_sentence = ""
    max_sentence = ""

    with open(filename) as f:
        for line in f:
            tokens = line.rstrip().lower().split()
            if not tokens:
                continue
            sentence = ' '.join(tokens)
            perplexity = calculate_perplexity(sentence, word_to_index, transition_matrix, unk_index)
            if perplexity < min_perplexity:
                min_perplexity = perplexity
                min_sentence = sentence
            if perplexity > max_perplexity:
                max_perplexity = perplexity
                max_sentence = sentence

    return min_sentence, max_sentence, min_perplexity, max_perplexity

# Create the Markov model
word_to_index, transition_matrix, unk_index = create_markov_model("robert_frost.txt")

# Calculate the perplexity of a sentence
sentence = "the young folk held some hope out to each other. </s>"
perplexity = calculate_perplexity(sentence, word_to_index, transition_matrix, unk_index)
print(f"Perplexity of sentence '{sentence}': {perplexity}")

# Find the sentences with minimum and maximum perplexity from the corpus
min_sentence, max_sentence, min_perplexity, max_perplexity = find_min_max_perplexity_sentences("robert_frost.txt", word_to_index, transition_matrix, unk_index)
print(f"Sentence with minimum perplexity: '{min_sentence}' (Perplexity: {min_perplexity})")
print(f"Sentence with maximum perplexity: '{max_sentence}' (Perplexity: {max_perplexity})")


