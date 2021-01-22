import csv

class CSVBuilder:
    def __init__(self, csvfile) -> None:
        self.csvfile = csvfile
        self.csvwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        self.writeheader()

    def writeheader(self) -> None:
        self.csvwriter.writerow(
            ["Song ID", "Song Name", "Number of Comments",
            "EmoVAD_sum"]
            )

    def writerow(self, features: list) -> None: 
        self.csvwriter.writerow(features)