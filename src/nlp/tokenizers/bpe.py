"""Byte-Pair Encoding (BPE) tokenizer."""

from tqdm.auto import tqdm


def bpe_train(corpus, vocab_size):
    """Uses the BPE algorithm to train a tokenizer on a corpus. Returns the vocabulary."""
    # Initialize the vocabulary
    corpus = [
        l if word[0] == l else "##" + l for word in corpus.split(" ") for l in word
    ]
    vocab = set(corpus)

    # Keep merging most likely pairs until the vocabulary is the desired size
    for i in tqdm(range(vocab_size - len(vocab))):
        counts = {}

        # Count how many times each pair appears
        for i in range(len(corpus) - 1):
            pair = corpus[i] + corpus[i + 1]
            counts[pair] = counts.get(pair, 0) + 1

        # Find the pair that appeared the most
        max_pair, max_count = None, -1
        for k in counts:
            if counts[k] > max_count:
                max_pair, max_count = k, counts[k]

        # Adding pair to vocabulary
        vocab.add(max_pair)

        # Updating corpus
        new_corpus, added = [], False
        for i in range(len(corpus) - 1):
            if added:
                added = False
                continue

            if corpus[i] + corpus[i + 1] == max_pair:
                new_corpus.append(max_pair)
                added = True
            else:
                new_corpus.append(corpus[i])

        corpus = new_corpus

    # Remove the "##" prefix and return vocabulary
    vocab = set(
        [
            ("##" if elem.startswith("##") else "") + elem.replace("##", "")
            for elem in vocab
        ]
    )
    return vocab


if __name__ == "__main__":
    # Example
    corpus = "machine learning and meta learning allow machines to learn how to learn"
    vocabulary = bpe_train(corpus, 30)
    print(vocabulary)
