import unittest
from wordlists import WordLists

# Run me with: python3 -m unittest test_wordlists

#tdd is fun :3 
class TestFeatureExtractor(unittest.TestCase):
    def text_exist(self):
        """
        The wordlists can be created
        """
        try:
            wl = WordLists()
        except NameError:
            self.fail("Could not instantiate wordlists!")

    def test_has_dir(self):
        """
        Wordlists object has a directory where raw data is stored
        This director is `wordlists/` by default
        """
        wl = WordLists()
        self.assertEqual(wl.dir, 'wordlists/')

    def test_has_filepath_dict(self):
        """
        Wordlists has a constant, ugly dict storing the filenames for
        certain wordlists
        """
        wl = WordLists()
        self.assertEqual(type(wl.wordlists), dict)



if __name__ == "__main__":
    unittest.main()