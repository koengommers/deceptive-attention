def anonymize(words):
    gender_replacements = {
        "he": "they",
        "she": "they",
        "her": "their",
        "his": "their",
        "him": "them",
        "himself": "themself",
        "herself": "themself",
        "hers": "their",
    }

    anon_words = []
    for word in words.split():
        if word in gender_replacements:
            anon_words.append(gender_replacements[word])
        else:
            anon_words.append(word)
    return " ".join(anon_words)
