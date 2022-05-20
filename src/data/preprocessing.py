import re
from typing import Callable

import pandas as pd
import spacy
from tqdm import tqdm

from src.data import TEXT_COLUMN


class PreprocessTextPipeline:
    def __init__(self, stages: list[str]):
        self.preprocessors = [get_preprocessor(stage) for stage in stages]

    def __call__(self, texts: list[str]) -> list[str]:
        if not self.preprocessors:
            return texts

        new_texts = []
        for text in tqdm(texts, desc='Preprocessing texts'):
            for preprocessor in self.preprocessors:
                text = preprocessor(text)
            new_texts.append(text.strip())

        return new_texts


class NormalizeWhitespaces:
    def __call__(self, text: str) -> str:
        return ' '.join(text.split())


class Text2LowerCase:
    def __call__(self, text: str) -> str:
        return text.lower()


class RemoveUrls:
    def __init__(self):
        self.url_pattern = r'https?://\S+'

    def __call__(self, text: str) -> str:
        return re.sub(self.url_pattern, '', text)


class RemoveHashtags:
    def __init__(self):
        self.hashtag_pattern = r'#(\w+)\b'

    def __call__(self, text: str) -> str:
        return re.sub(self.hashtag_pattern, '', text)


class RemoveUserMentions:
    def __call__(self, text: str) -> str:
        usermention_patter = r'(^|[^\w@/\!?=&])@(\w{1,15})'
        return re.sub(usermention_patter, '', text)


class RemoveEmoji:
    def __init__(self):
        """
        reference: https://gist.github.com/slowkow/7a7f61f495e3dbb7e3d767f97bd7304b
        """
        self.emoji_pattern = re.compile(
            "["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags 
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            "]+",
            flags=re.UNICODE,
        )

    def __call__(self, text: str) -> str:
        return self.emoji_pattern.sub(r'', text)


class RemovePunctuation:
    def __init__(self):
        self.nlp = spacy.load('pl_core_news_sm', disable=["parser", "ner"])

    def __call__(self, text: str) -> str:
        cleaned_text = ' '.join([token.text for token in self.nlp(text) if not token.is_punct])
        return cleaned_text


class Lemmatize:
    def __init__(self):
        self.nlp = spacy.load('pl_core_news_sm', disable=["parser", "ner"])

    def __call__(self, text: str) -> str:
        cleaned_text = ' '.join([token.lemma_ for token in self.nlp(text)])
        return cleaned_text


class SpacyBasicClean:
    def __init__(self):
        self.nlp = spacy.load('pl_core_news_sm', disable=["parser", "ner"])

    def __call__(self, text: str) -> str:
        cleaned_text = ' '.join([
            token.lemma_.lower() for token in self.nlp(text)
            if not (
                token.is_stop
                or token.is_punct
                or token.like_email
                or token.like_url
                or token.like_num
                or token.is_digit
            )
        ])
        return cleaned_text


_PREPROCESSORS = {
    'norm_whitespaces': NormalizeWhitespaces,
    'lowercase': Text2LowerCase,
    'remove_punct': RemovePunctuation,
    'remove_urls': RemoveUrls,
    'remove_hashtags': RemoveHashtags,
    'remove_user_mentions': RemoveUserMentions,
    'remove_emoji': RemoveEmoji,
    'lemmatize': Lemmatize,
    'spacy_basic_clean': SpacyBasicClean,
}


def get_preprocessor(name: str) -> Callable:
    if name not in _PREPROCESSORS:
        raise ValueError("Unsupported preprocessor name.")

    return _PREPROCESSORS[name]()


def get_base_pipeline() -> PreprocessTextPipeline:
    return PreprocessTextPipeline(stages=[
        "remove_hashtags",
        "remove_user_mentions",
        "remove_urls",
        "remove_emoji",
        "norm_whitespaces",
    ])


def get_wordcloud_pipeline() -> PreprocessTextPipeline:
    return PreprocessTextPipeline(stages=[
        "remove_hashtags",
        "remove_user_mentions",
        "remove_urls",
        "remove_emoji",
        "norm_whitespaces",
        "spacy_basic_clean",
    ])


def get_drop_indices_by_min_words(df: pd.DataFrame, min_words: int) -> list[int]:
    base_pipeline = get_base_pipeline()
    texts = df[TEXT_COLUMN]
    processed_texts = base_pipeline(texts)
    drop_indices = [
        idx for idx, text in enumerate(processed_texts) if len(text.split()) < min_words
    ]
    return drop_indices
