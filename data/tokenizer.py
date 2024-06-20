from pathlib import Path

from tokenizers import Tokenizer  # type: ignore
from tokenizers.models import WordLevel  # type: ignore
from tokenizers.pre_tokenizers import Whitespace  # type: ignore
from tokenizers.trainers import WordLevelTrainer  # type: ignore

from data.data_utils import get_all_sentences


def get_or_build_tokenizer(config, dataset, lang):
    tokenizer_path = Path(config["tokenizer_file"].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(
            special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2
        )
        tokenizer.train_from_iterator(
            get_all_sentences()(dataset, lang), trainer=trainer
        )
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer
