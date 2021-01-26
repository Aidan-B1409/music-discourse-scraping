import csv

class CSVBuilder:
    def __init__(self, csvfile) -> None:
        self.csvfile = csvfile
        self.csvwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        self.writeheader()

    def writeheader(self) -> None:
        self.csvwriter.writerow(
            ["Song_ID", "Song_Artist", "Song_Name",
            "n_comments", "n_words", "n_words_uniq",
            "EmoVAD_glob_v_mean", "EmoVAD_glob_v_stdev",
            "EmoVAD_glob_a_mean", "EmoVAD_glob_a_stdev",
            "EmoVAD_glob_d_mean", "EmoVAD_glob_d_stdev",
            "EmoVAD_glob_v_mean_uniq", "EmoVAD_glob_v_stdev_uniq",
            "EmoVAD_glob_a_mean_uniq", "EmoVAD_glob_a_stdev_uniq",
            "EmoVAD_glob_d_mean_uniq", "EmoVAD_glob_d_stdev_uniq"]
            )

    def writerow(self, features: list) -> None: 
        self.csvwriter.writerow(features)