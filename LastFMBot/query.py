from pylast import LastFMNetwork
from pylast import WSError
import csv
from datetime import datetime

class Query:

    def __init__(self, cwd: str, track: str, artist: str, lastfm: LastFMNetwork):
        self.cwd = cwd
        self.track = track
        self.artist = artist
        self.lastfm = lastfm
        try:
            self.result = lastfm.get_track(artist, track)
        except(WSError):
            self.result = None

    def mine_tags(self, query_index: str, valence: str, arousal: str, song_id: str) -> None:
        dtime_string = datetime.now().strftime('%d-%m-%Y-%H-%M-%S')
        file_name='lastfm_tags_' + dtime_string + "_" + str(song_id) + ".csv"
        with open(self.cwd + file_name, 'w', newline='', encoding='utf-8') as csvfile:
            file_writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            file_writer.writerow(["Query Index", "Artist", "Track", "Valence", "Arousal", "Song ID"])
            file_writer.writerow([query_index, self.artist, self.track, valence, arousal, song_id])

            tag_names = list()
            if(self.result != None):
                tags = None
                try:
                    tags = self.result.get_top_tags()
                except(WSError):
                    tags = None
                if(tags != None):
                    for idx, tag in enumerate(tags):
                        tag_names.append(tag[0].get_name())
            else:
                tag_names.append("Track Not Found")

            file_writer.writerow(["Tags:", tag_names])