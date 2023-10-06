"""Wordpiece tokenizer."""

from tqdm.auto import tqdm


def wordpiece_train(corpus, vocab_size):
    """Uses the wordpiece algorithm to train a tokenizer on a corpus. Returns the vocabulary."""
    # Initialize the vocabulary with letters
    corpus = [
        l if word[0] == l else "##" + l for word in corpus.split(" ") for l in word
    ]
    vocab = set(corpus)

    # Keep merging most likely pairs until the vocabulary is the desired size
    for i in tqdm(range(vocab_size - len(vocab))):
        counts = {}
        pair_counts = {}

        # Keep count of each word and each pair
        for i in range(len(corpus)):
            counts[corpus[i]] = counts.get(corpus[i], 0) + 1

            if i == len(corpus) - 1:
                continue

            pair = corpus[i] + corpus[i + 1]
            pair_counts[(corpus[i], corpus[i + 1])] = counts.get(pair, 0)

        # Find the pair that has the highest score
        # The score is count(w1, w2) / (count(w1) * count(w2))
        max_pair, max_score = None, -1
        for w1, w2 in pair_counts:
            p_count = pair_counts[(w1, w2)]
            pair_score = p_count / (counts[w1] * counts[w2])
            if pair_score > max_score:
                max_pair, max_score = w1 + w2, pair_score

        # Add the pair with the highest score to the vocabulary
        vocab.add(max_pair)

        # Update the corpus by merging the pair
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
    vocabulary = wordpiece_train(corpus, 30)
    print(vocabulary)
