import csv
import os
import errno
import wget 
from shutil import copy
import urllib

bp = "./posters"
def ensure_dir(file_path):
    try:
        os.makedirs(file_path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(file_path):
            pass
        else:
            raise

with open('MovieGenre.csv', mode='r') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            print(f'Column names are {", ".join(row)}')

        genres = row['Genre'].split("|")
        print(genres)
        try:
            wget.download(row["Poster"],"./poster_" + str(line_count))
        except urllib.error.HTTPError as e:
            pass
            continue
        except ValueError as ve:
            pass
            continue
            
        for genre in genres:
            ensure_dir(bp+"/"+genre)
            copy("./poster_" + str(line_count),"./posters" + "/" + genre )
        os.remove("./poster_" + str(line_count))
        line_count += 1
    print(f'Processed {line_count} lines.')