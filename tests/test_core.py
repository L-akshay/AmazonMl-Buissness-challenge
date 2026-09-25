import unittest
import unicodedata
from src.evaluate import entity_f05, evaluate
from src.normalize import conservative, name_views, address_views, accent_fold


class ScorerTests(unittest.TestCase):
    def test_organizer_example(self):
        self.assertAlmostEqual(entity_f05(["S2-00047", "S3-00812"],
                                         ["S2-00047", "S2-00193", "S3-00812"]), 5 / 7)

    def test_singletons_and_missed_matches(self):
        self.assertEqual(entity_f05([], []), 1)
        self.assertEqual(entity_f05([], ["S2-1"]), 0)
        self.assertEqual(entity_f05(["S2-1"], []), 0)
        self.assertEqual(entity_f05(["S2-1"], ["S2-1"]), 1)

    def test_macro_not_micro(self):
        result = evaluate({"a": set(), "b": {"x", "y"}}, {"a": set(), "b": {"x"}})
        self.assertAlmostEqual(result["macro_f05"], (1 + 5 / 6) / 2)

    def test_missing_s1_rejected(self):
        with self.assertRaises(ValueError):
            evaluate({"a": set()}, {})


class NormalizationTests(unittest.TestCase):
    def test_suffix_and_raw(self):
        raw = "Sharma Textiles Private Limited"
        views = name_views(raw)
        self.assertEqual(views["raw"], raw)
        self.assertEqual(views["core"], "sharma textiles")
        self.assertEqual(name_views("Acme Incorporated")["core"], "acme")

    def test_unicode_and_punctuation(self):
        self.assertEqual(conservative("Smith & Sons, Inc."), "smith and sons inc")
        self.assertEqual(accent_fold("école"), "ecole")
        self.assertEqual(accent_fold("मार्केटिंग"), "मार्केटिंग")
        self.assertEqual(conservative("मार्केटिंग"), "मार्केटिंग")

    def test_address_numbers_preserved(self):
        views = address_views("12/7 Market St., 560001")
        self.assertEqual(views["numbers"], ["12", "7", "560001"])
        self.assertEqual(views["standardized"], "12 7 market street 560001")

    def test_ascii_fast_path_matches_conservative_definition(self):
        raw = "".join(chr(i) for i in range(128)) + " SMITH__&-SONS 12/7 "
        expected = unicodedata.normalize("NFKC", raw).casefold().replace("&", " and ")
        expected = "".join(c if c.isalnum() or unicodedata.category(c).startswith("M") else " " for c in expected)
        self.assertEqual(conservative(raw), " ".join(expected.split()))


if __name__ == "__main__":
    unittest.main()
