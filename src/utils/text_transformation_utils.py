import re
import ftfy
import emoji
from nltk.corpus import stopwords


def clean_text(text: str) -> str:
    """
    Cleanses the provided text corpus from various impurities.
    Also removes isolated letters (Articles, Conjunctions, Prepositions, and other meaningless wastes of characters).
    :param text:
    :return:
    """

    # replaces emoji with their textual code (:thumbs_up:)
    text = emoji.demojize(text, delimiters=(" ", " "))
    # fix mojibake (encoding mix-ups), by detecting patterns of UTF-8 characters that was decoded wrong
    text = ftfy.fix_text(text)

    # standardize text to lowercase
    text = text.lower()

    # clear html and xml tags
    html = re.compile(r'<[^>]+>')
    text = html.sub(r'', text)
    # remove urls
    text = re.sub("http\\w+", '', text)
    # remove username handles
    text = re.sub("@\\w+", '', text)

    # remove isolated letters
    text = re.sub("\\s[a-z]\\s", ' ', text)

    # remove multiple spaces (and tabs, and newlines)
    text = re.sub("\\s\\s+", ' ', text)

    # drop english stopwords
    english_stopwords = stopwords.words('english')
    text_list = text.split(" ")
    text_list = [t for t in text_list if t not in english_stopwords]
    text = " ".join(text_list)

    return text.strip()


def calculate_ari_score(text):
    """
    Calculates the Automated Readability Index, a readability test designed to gauge the understandability of a text.
    It considers the number of characters per word and the number of words per sentence to estimate the text complexity.
    :param text:
    :return:
    """
    cleaned_text = clean_text(text)
    characters = len(cleaned_text)
    words = len(re.split('[ \n.?!,]', cleaned_text))
    sentence = len(re.split('[.?!]', cleaned_text))
    ari_score = 4.71 * (characters / words) + 0.5 * (words / sentence) - 21.43
    return ari_score


def calculate_eflaw_score(text):
    """
    Calculates the McAlpine Estimated Friction for the Level of Academic Writing, it reflects the text complexity.
    It measures the ratio of words to sentences and words multiplied by sentences.
    :param text:
    :return:
    """
    W = len(re.split('[ \n.?!,]', text))
    S = len(re.split('[.?!]', text))
    eflaw_score = (W + S * W) / S
    return eflaw_score


def calculate_clri_score(text):
    """
    Calculates the Coleman-Liau Readability Index, a readability test specific for educational contexts.
    It uses characters per word and sentences per word to determine the readability level.
    :param text:
    :return:
    """
    characters = len(text)
    words = len(re.split('[ \n.?!,]', text))
    sentence = len(re.split('[.?!]', text))
    L = 100 * characters / words
    S = 100 * sentence / words
    clri_score = 0.0588 * L - 0.296 * S - 15.8
    return clri_score
