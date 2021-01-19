import csv
import os

directory = "/media/Red2/Datasets/Deezer_2018/reddit_new/validation/"
def main():
    for file in os.listdir(directory):
        fname = os.fsdecode(file)
        lines = list()
        with open(directory + fname, 'r') as readFile:
            csvReader = csv.reader(readFile)
            for row in csvReader:
                lines.append(row)
        if(len(lines)>2):
            lines.pop()
        with open(directory + fname, 'w') as writeFile:
            csvWriter = csv.writer(writeFile)
            csvWriter.writerows(lines)



if __name__ == "__main__":
    main()