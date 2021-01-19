from wordlists import WordLists
class FeatureExtractor:
    
    def __init__(self, wordlists = WordLists()) -> None:
        self.wordlists = wordlists