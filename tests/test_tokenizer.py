import json
import tempfile
import unittest
from pathlib import Path

from mini_ai.tokenizer import BOS_ID, EOS_ID, PAD_ID, UNK_ID, Tokenizer
from mini_ai.tokenizer.bpe import BASE_VOCAB_SIZE, merge_pair

CORPUS = [
    "Bonjour le monde. Python est un langage de programmation.",
    "Django utilise l'architecture MTV. Django est écrit en Python.",
    "Le manager coordonne les agents. Le développeur propose des solutions.",
] * 5


class TestBPE(unittest.TestCase):
    def test_merge_pair(self):
        self.assertEqual(merge_pair([1, 2, 3, 1, 2], (1, 2), 9), [9, 3, 9])
        self.assertEqual(merge_pair([1, 1, 1], (1, 1), 9), [9, 1])

    def test_untrained_tokenizer_is_bytes(self):
        tok = Tokenizer()
        self.assertEqual(tok.vocab_size, BASE_VOCAB_SIZE)
        self.assertEqual(len(tok.encode("abc")), 3)
        self.assertEqual(tok.decode(tok.encode("abc")), "abc")


class TestTokenizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = Tokenizer.train(CORPUS, vocab_size=400)

    def test_training_grows_vocab(self):
        self.assertGreater(self.tok.vocab_size, BASE_VOCAB_SIZE)
        self.assertLessEqual(self.tok.vocab_size, 400)

    def test_roundtrip(self):
        for text in CORPUS + ["Ça marche : éàü, 日本語 — 123 !", "  espaces   multiples \n\n et sauts", ""]:
            norm = Tokenizer.normalize(text)
            self.assertEqual(self.tok.decode(self.tok.encode(text)), norm)

    def test_compression(self):
        text = CORPUS[0]
        self.assertLess(len(self.tok.encode(text)), len(text.encode("utf-8")))

    def test_special_tokens(self):
        ids = self.tok.encode("Bonjour", add_bos=True, add_eos=True)
        self.assertEqual(ids[0], BOS_ID)
        self.assertEqual(ids[-1], EOS_ID)
        self.assertEqual(self.tok.decode(ids), "Bonjour")
        self.assertEqual(self.tok.decode(ids, skip_special=False), "<BOS>Bonjour<EOS>")
        self.assertEqual(self.tok.token_to_id("<PAD>"), PAD_ID)
        self.assertEqual(self.tok.token_to_id("<UNK>"), UNK_ID)
        self.assertEqual(self.tok.id_to_token(EOS_ID), "<EOS>")

    def test_save_load(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "tok.json"
            self.tok.save(path)
            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(data["vocab_size"], self.tok.vocab_size)
            loaded = Tokenizer.load(path)
        self.assertEqual(loaded.vocab_size, self.tok.vocab_size)
        text = "Python est un langage"
        self.assertEqual(loaded.encode(text), self.tok.encode(text))

    def test_deterministic(self):
        other = Tokenizer.train(CORPUS, vocab_size=400)
        self.assertEqual(other.bpe.merge_list(), self.tok.bpe.merge_list())


if __name__ == "__main__":
    unittest.main()
