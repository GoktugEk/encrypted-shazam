import random
import librosa
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import warnings

warnings.filterwarnings("ignore")

### Spectogram Functions ###


def get_spectogram(
    mp3_file, n_fft=2205, record_length: None | int = None, random_seed=42
):
    random.seed(random_seed)
    # if record length exists, load the audio file with the duration, with random start
    # point
    if record_length is not None:
        y, sr = librosa.load(
            mp3_file,
            duration=record_length,
            offset=random.random()
            * (librosa.get_duration(filename=mp3_file) - record_length),
        )
    else:
        y, sr = librosa.load(mp3_file)

    # encrypt the audio file

    fft_results = librosa.amplitude_to_db(librosa.core.stft(y, n_fft=n_fft))

    return fft_results, sr


def save_spectograms(
    mp3_files: list, filepath: str, n_fft: int = 2205, record_length: None | int = None
):
    spectograms = {}
    for filename, path in mp3_files.items():
        fft_results, sr = get_spectogram(path, n_fft, record_length)

        spectograms[os.path.splitext(filename)[0]] = fft_results

    filepath = filepath + f"{n_fft}_{len(mp3_files)}.npy"
    np.save(
        filepath,
        spectograms,
    )


### Constellation Map Functions ###

LOGARITHMIC_BANDS = [(1, 20), (21, 40), (41, 80), (81, 160), (161, 512)]


def create_constellation_map(
    fft_results, frame_duration=0.1, sr=22050, hop_length=551, mean_coefficient=0.8
):
    frame_length = int(frame_duration * sr // hop_length)
    times = librosa.times_like(fft_results)
    selected_bins_over_time = []

    for frame_start in range(0, len(times), frame_length):
        frame_end = frame_start + frame_length
        frame_bins = []
        frame_bin_powers = []

        for start_bin, end_bin in LOGARITHMIC_BANDS:
            max_magnitude = -1
            strongest_bin = None

            for bin_num in range(start_bin, end_bin):
                band_fft = np.abs(fft_results[bin_num, frame_start:frame_end])
                max_magnitude_in_band = np.max(band_fft)

                if max_magnitude_in_band > max_magnitude:
                    max_magnitude = max_magnitude_in_band
                    strongest_bin = bin_num

            frame_bins.append(strongest_bin)
            frame_bin_powers.append(max_magnitude)

        threshold = mean_coefficient * np.mean(frame_bin_powers)
        selected_bins = np.where(np.array(frame_bin_powers) > threshold)[0]
        frame_bins = np.array(frame_bins)[selected_bins]

        selected_bins_over_time.append(frame_bins)

    constellation_map = []

    for i, frame_bins in enumerate(selected_bins_over_time):
        for bin in frame_bins:
            constellation_map.append((times[i * frame_length], bin))

    return constellation_map


def save_constellation_maps(
    spectograms: str,
    filepath,
    frame_duration=0.1,
    sr=22050,
    hop_length=551,
    mean_coefficient=0.8,
):
    spectograms = np.load(spectograms, allow_pickle=True).item()
    constellation_maps = {}
    for song_id, fft_results in spectograms.items():
        constellation_map = create_constellation_map(
            fft_results, frame_duration, sr, hop_length, mean_coefficient
        )
        constellation_maps[song_id] = constellation_map

    filepath = filepath + f"{len(spectograms)}_{mean_coefficient}.npy"
    np.save(filepath, constellation_maps)


def plot_constellation_map(
    constellation_map, with_indexes=True, duration=None, offset=0
):
    _constellation_map = [
        (time, freq) for time, freq in constellation_map if time >= offset
    ]
    if duration is not None:
        _constellation_map = list(
            filter(lambda x: x[0] < duration + offset, _constellation_map)
        )

    plt.figure(figsize=(10, 6))
    plt.scatter(
        [time for time, _ in _constellation_map],
        [bin for _, bin in _constellation_map],
        marker="x",
        color="b",
    )

    if with_indexes:
        for i in range(len(_constellation_map)):
            plt.annotate(str(i), _constellation_map[i])

    plt.title(f"Selected Frequency Bins Over Time (First {duration} Seconds)")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency Bins")
    plt.show()


### Address Value Couple Functions ###


def _find_target_zone_for_anchor(
    constellation_map, anchor_time, anchor_freq, anchor_index, target_zone_size=5
) -> int | None:
    step = int(np.ceil(target_zone_size / 2))
    interval = 0
    for i in range(anchor_index + 1, len(constellation_map)):
        time, freq = constellation_map[i]
        if time - anchor_time == 0:
            interval += 1
            continue
        elif interval < step:
            interval += 1
            continue
        else:
            return i

    return None


def _create_address_value_couples(constellation_map, song_id: str, zone_size=5):
    addresses_couples = []

    for i, (anchor_time, anchor_freq) in enumerate(constellation_map):
        target_zone_start = _find_target_zone_for_anchor(
            constellation_map, anchor_time, anchor_freq, i, zone_size
        )
        if target_zone_start is None:
            continue

        target_zone_end = (
            target_zone_start + zone_size
            if target_zone_start + zone_size < len(constellation_map)
            else None
        )

        if target_zone_end is None:
            continue

        target_zone = constellation_map[target_zone_start:target_zone_end]

        for t, freq in target_zone:
            address_couple = [
                anchor_freq,
                freq,
                t - anchor_time,
                anchor_time,
                float(song_id),
            ]

            addresses_couples.append(address_couple)

    return np.array(addresses_couples)


def create_address_couples(constellations_file: str):
    constellations = np.load(constellations_file, allow_pickle=True).item()
    addresses_couples = []

    for song_id, constellation_map in constellations.items():
        addresses_couples.append(
            _create_address_value_couples(constellation_map, song_id)
        )
    db = np.concatenate(addresses_couples)

    return db


def create_address_couples_by_song(constellations_file: str):
    constellations = np.load(constellations_file, allow_pickle=True).item()
    addresses_couples = []

    for song_id, constellation_map in constellations.items():
        addresses = _create_address_value_couples(constellation_map, song_id)

        addresses[:, 2] = addresses[:, 2] * 10
        addresses[:, 3] = addresses[:, 3] * 100

        addresses_couples.append(addresses.astype(np.uint16))

    return addresses_couples


def create_address_couples_from_spectograms(spectograms_file: str):
    spectograms = np.load(spectograms_file, allow_pickle=True).item()
    addresses_couples = []

    for song_id, spectogram in spectograms.items():
        constellation_map = create_constellation_map(spectogram)

        addresses_couples.append(
            _create_address_value_couples(constellation_map, song_id)
        )

    db = np.concatenate(addresses_couples)
    db = db.astype(np.uint16)

    return db

def create_query_integer(query_path: str, target_audio_record_seconds: float = 3):
    fft_results, sr = get_spectogram(query_path, record_length=target_audio_record_seconds)
    constellation_map = create_constellation_map(fft_results)

    addresses_couples = _create_address_value_couples(constellation_map, 0)

    addresses_couples[:, 2] = addresses_couples[:, 2] * 10
    addresses_couples[:, 3] = addresses_couples[:, 3] * 100

    addresses_couples = addresses_couples.astype(np.uint16)

    return addresses_couples[:, :3]


def create_address_couples_from_audios(audios: dict):
    addresses_couples = []

    for song_id, audio_path in audios.items():
        fft_results, sr = get_spectogram(audio_path)
        constellation_map = create_constellation_map(fft_results)

        addresses_couples.append(
            _create_address_value_couples(constellation_map, song_id)
        )

    db = np.concatenate(addresses_couples)

    return db


### Utility Functions ###


def load_audios(path, num_audios=10):
    audios = {}

    folders = os.listdir(path)

    loaded = 0

    for folder in folders:
        if not os.path.isdir(os.path.join(path, folder)):
            continue
        if loaded == num_audios:
            break
        folder_path = os.path.join(path, folder)
        audio_files = os.listdir(folder_path)
        for audio_file in audio_files:
            if not audio_file.endswith(".mp3"):
                continue
            if loaded == num_audios:
                break

            audio_path = os.path.join(folder_path, audio_file)
            audios[os.path.splitext(audio_file)[0]] = audio_path
            loaded += 1

    return audios


def search_address(db: np.array, query: np.array):
    indices = np.where((db[:, :3] == query[:, np.newaxis, :]).all(axis=2))[1]

    return db[indices]


def process_matches(matches: np.array):
    # Count the matches by song ID, count anchor times that the ones are larger than 5,
    # Create a dictionary with the song id as the key, and the value is also a
    # dictionary which has address_matches and target_zone_matches as the keys,
    # and the values are the counts.

    # Initialize the list of tuples to store the matches for each song id
    song_matches = []

    # Find unique song ids
    song_ids = np.unique(matches[:, 4])

    # Loop over each song id
    for song_id in song_ids:
        # Filter the list of matches to get the matches for the current song id
        song_matches_filtered = matches[matches[:, 4] == song_id]

        # Find the unique 4th numbers for the current song id
        _, counts_4th_numbers = np.unique(
            song_matches_filtered[:, 3], axis=0, return_counts=True
        )

        # Calculate the target zone matches for each unique 4th number
        target_zone_matches = np.sum(counts_4th_numbers // 5)

        song_matches.append(
            (song_id, np.sum(target_zone_matches), len(song_matches_filtered))
        )

    # sort the song matches by the number of target zone matches then the number of
    # address matches in descending order
    song_matches = sorted(song_matches, key=lambda x: (x[1], x[2]), reverse=True)

    return song_matches


def print_results(song_matches: list, num_results=3, song_id: float = None):
    # print green if the first result is the target song, print yellow if one of the
    # first three guesses is correct, otherwise print red
    if song_matches[0][0] == song_id:
        print("\033[92m" + "The target song is found correct!" + "\033[0m")
    elif song_matches[1][0] == song_id or song_matches[2][0] == song_id:
        print(
            "\033[93m"
            + "The target song is found in the first three guesses"
            + "\033[0m"
        )
    else:
        print("\033[91m" + "The target song is not found correct" + "\033[0m")
    i = 0
    for song_id, target_zone_matches, address_matches in song_matches:
        print(
            f"Song ID: {int(song_id)}, Target Zone Matches: {target_zone_matches},\
                Address Matches: {address_matches}"
        )
        i += 1
        if i == num_results:
            break


def search_song(
    db: np.array,
    target_audio_path: str,
    target_audio_record_seconds: int = 3,
    report: bool = True,
    n_fft=2205,
    mean_coefficient=0.8,
    zone_size=5,
):
    fft_results, sr = get_spectogram(
        target_audio_path, record_length=target_audio_record_seconds, n_fft=n_fft
    )
    constellation_map = create_constellation_map(
        fft_results, mean_coefficient=mean_coefficient, hop_length=n_fft // 4
    )
    song_id = os.path.splitext(os.path.basename(target_audio_path))[0]
    addresses_couples = _create_address_value_couples(
        constellation_map, song_id, zone_size=zone_size
    )
    # select the first 3 columns as the addresses
    addresses = addresses_couples[:, :3]
    results = search_address(db, addresses)
    processed_results = process_matches(results)
    if report:
        print_results(processed_results, song_id=float(song_id))

    if len(processed_results) == 0:
        return constellation_map, results, False, False
    found = processed_results[0][0] == float(song_id)

    found_in_first_three = float(song_id) in np.array(processed_results[:3])[:, 0]

    return constellation_map, results, found, found_in_first_three


### EVALUATION FUNCTIONS ###


def run_experiment(
    number_of_audios, n_fft, mean_coefficient, zone_size, recording_length
):
    # Load the audios
    audios = load_audios("data/fma_small/", num_audios=number_of_audios)

    # Save the spectograms if not exists
    if not os.path.exists(f"spectograms/{n_fft}_{len(audios)}.npy"):
        save_spectograms(audios, "cache/spectograms/", n_fft)

    time.sleep(0.5)

    # Save the constellation maps if not exists
    if not os.path.exists(f"constellation_maps/{len(audios)}_{mean_coefficient}.npy"):
        save_constellation_maps(
            f"cache/spectograms/{n_fft}_{number_of_audios}.npy",
            "cache/constellation_maps/",
            mean_coefficient=mean_coefficient,
            hop_length=n_fft // 4,
        )

    # Create the address couples
    addresses_couples = create_address_couples(
        f"cache/constellation_maps/{number_of_audios}_{mean_coefficient}.npy"
    )

    # Search all the audios in the database one by one
    found = 0
    found_in_first_three = 0

    for song_id, audio_path in audios.items():
        _, _, _found, _found_in_first_three = search_song(
            addresses_couples,
            audio_path,
            target_audio_record_seconds=recording_length,
            report=False,
            n_fft=n_fft,
            mean_coefficient=mean_coefficient,
            zone_size=zone_size,
        )
        if _found:
            found += 1
        if _found_in_first_three:
            found_in_first_three += 1

    # Append configurations and results to the csv file

    found = found / number_of_audios * 100
    found_in_first_three = found_in_first_three / number_of_audios * 100

    columns = [
        "Number of Audios",
        "n_fft",
        "Mean Coefficient",
        "Zone Size",
        "Recording Length",
        "Found",
        "Found in First Three",
    ]

    data = [
        [
            number_of_audios,
            n_fft,
            mean_coefficient,
            zone_size,
            recording_length,
            found,
            found_in_first_three,
        ]
    ]

    df = pd.DataFrame(data, columns=columns)

    reports = pd.read_csv("experiment_results_2.csv")
    reports = pd.concat([reports, df], ignore_index=True)
    reports.to_csv("experiment_results_2.csv", index=False)

    print(f"Found: {found}%")
    print(f"Found in first three: {found_in_first_three}%")

    return found, found_in_first_three


def grid_search():
    candidate_audio_numbers = [10, 50, 100]
    candidate_mean_coefficient = [0.8, 1]
    candidate_zone_size = [5, 10, 20]
    candidate_recording_length = [5, 10, 20]
    # Grid search

    searched_configs = pd.read_csv("experiment_results_2.csv")

    for number_of_audios in candidate_audio_numbers:
        for mean_coefficient in candidate_mean_coefficient:
            for zone_size in candidate_zone_size:
                for recording_length in candidate_recording_length:
                    config_row = {
                        "Number of Audios": [number_of_audios],
                        "n_fft": [2205],
                        "Mean Coefficient": [mean_coefficient],
                        "Zone Size": [zone_size],
                        "Recording Length": [recording_length],
                    }

                    if (
                        not searched_configs.iloc[:, :5]
                        .isin(config_row)
                        .all(axis=1)
                        .any()
                    ):
                        run_experiment(
                            number_of_audios,
                            2205,
                            mean_coefficient,
                            zone_size,
                            recording_length,
                        )
                    else:
                        print("Skipped")
                        print(config_row)
                        print("-" * 20)
                        print()


### Chunking Logic ###
def chunk_anchor_times_of_songs(db, number_of_entries=25):
    chunked_db = []

    for song in db:
        chunked_song = []

        anchor_times = song[:, 3]
        anchor_times = np.unique(anchor_times)

        for anchor_time in anchor_times:
            chunked = song[song[:, 3] == anchor_time]

            chunked_size = chunked.shape[0]

            if chunked_size < number_of_entries:
                zeros = np.zeros((number_of_entries - chunked_size, 5))
                chunked = np.concatenate([chunked, zeros]).astype(np.uint16)
                
            chunked_song.append(chunked)

        chunked_db.append(np.array(chunked_song).astype(np.uint16))

    return chunked_db


def chunkify_bits_of_array(target_array):
    assert target_array.shape[1] == 3

    chunked_array = []

    for i in range(2):
        bitted_array = (target_array[:, i] >> 6 & 0b111).reshape(-1, 1)
        chunked_array.append(bitted_array)
        bitted_array = (target_array[:, i] >> 3 & 0b111).reshape(-1, 1)
        chunked_array.append(bitted_array)
        bitted_array = (target_array[:, i] & 0b111).reshape(-1, 1)
        chunked_array.append(bitted_array)

    chunked_array.append(target_array[:, 2].reshape(-1, 1))
    chunked_array = np.concatenate(chunked_array, axis=1)

    return chunked_array


def chunkify_bits_of_db(db):
    new_db = []
    for song in db:
        chunked_song = []
        for chunk in song:
            shifted_chunk = chunkify_bits_of_array(chunk[:, :3])
            chunked_song.append(
                np.array(np.concatenate([shifted_chunk, chunk[:, 3:]], axis=1))
            )

        new_db.append(np.array(chunked_song))

    return new_db


# df = pd.read_csv("experiment_results.csv")
# df = df.sort_values(
#     by=["Found", "Found in First Three", "Number of Audios"], ascending=False
# )[df["Recording Length"] == 5]
# df.head(20)
if __name__ == "__main__":
    grid_search()
