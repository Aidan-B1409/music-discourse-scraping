import unittest
from feature_extractor_2 import FeatureExtractor

# Run me with: python3 -m unittest test_feature_extractor

#tdd is fun :3 
class TestFeatureExtractor(unittest.TestCase):
    def text_exist(self):
        try:
            FeatureExtractor()
        except NameError:
            self.fail("Could not instantiate feature extractor!")

    def test_has_wordlists(self):
        """
        The feature extractor object has a WordLists object
        """
        fe = FeatureExtractor()
        self.assertNotEqual(fe.wordlists, None)

    def test_has_comment_dir(self):
        """
        The feature extractor has a path to the comments
        """
        fe = FeatureExtractor
        self.assertEqual((fe.comments, None))


if __name__ == "__main__":
    unittest.main()