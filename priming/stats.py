import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
from tqdm import tqdm
from config import DATA_DIR

PLOT_FLAG = True

frame_of_interests = [
    " Morality",
    'Crime and Punishment',
    'Cultural Identity',
    "Economic",
    'External Regulation and Reputation',
    'Fairness and Equality',
    "Political",
    "Public Sentiment",
    'Quality of Life',
]


if not PLOT_FLAG:
    # final_stats = {}
    final_stats_no_none = {}
    final_stats_no_other = {}

    for media_type in ["independent", "state-affiliated"]:
        # media_curr_stats = {}
        media_curr_stats_no_none = {}
        media_curr_stats_no_other = {}

        with open(Path(DATA_DIR) / "processed-vk" /
                  media_type / "comment_tids2mfcframe.pkl", "rb") as f:
            comment_tids2mfcframe = pickle.load(f)

        with open(Path(DATA_DIR) / "processed-vk" /
                  media_type / "comment_tids2info.pkl", "rb") as f:
            comment_tids2info = pickle.load(f)

        with open(Path(DATA_DIR) / "processed-vk" /
                  media_type / "tids2mfcframe.pkl", "rb") as f:
            tids2mfcframe = pickle.load(f)

        for limit_scope in [None, "Crime and Punishment",
                            "Fairness and Equality", "Public Sentiment"]:
            # stats = {}
            stats_no_none = {}
            stats_no_other = {}

            for comment_id, frame in tqdm(comment_tids2mfcframe.items()):
                info = comment_tids2info[comment_id]
                if limit_scope:
                    post_id = info["post_id"]
                    post_frame = tids2mfcframe.get(int(post_id), None)
                    if post_frame != limit_scope:
                        continue
                # stats[frame] = stats.get(frame, 0) + 1
                if frame is not None:
                    stats_no_none[frame] = stats_no_none.get(frame, 0) + 1
                if frame not in [None, "Other"]:
                    stats_no_other[frame] = stats_no_other.get(frame, 0) + 1

            # frame_of_interests = [" Morality", "Economic", "Political", "Public Sentiment", 'External Regulation and Reputation', 'Fairness and Equality', 'Quality of Life', 'Cultural Identity', 'Crime and Punishment']

            # total = len(comment_tids2mfcframe) - stats[None]
            # total = sum(stats.values())
            total_no_none = sum(stats_no_none.values())
            total_no_other = sum(stats_no_other.values())

            # media_curr_stats[limit_scope] = {
            #     frame: round(
            #         stats[frame] / total,
            #         3) * 100 for frame in frame_of_interests}

            media_curr_stats_no_none[limit_scope] = {
                frame: round(
                    stats_no_none[frame] / total_no_none,
                    3) * 100 for frame in frame_of_interests}

            # media_curr_stats_no_other[limit_scope] = {
            #     frame: round(
            #         stats_no_other[frame] / total_no_other,
            #         3) * 100 for frame in frame_of_interests}

        # final_stats[media_type] = media_curr_stats
        final_stats_no_none[media_type] = media_curr_stats_no_none
        final_stats_no_other[media_type] = media_curr_stats_no_other

    # with open("final_stats.pkl", "wb") as f:
    #     pickle.dump(final_stats, f)

    with open("final_stats_no_none.pkl", "wb") as f:
        pickle.dump(final_stats_no_none, f)

    # with open("final_stats_no_other.pkl", "wb") as f:
    #     pickle.dump(final_stats_no_other, f)

else:
    # final_stats = pickle.load(open("final_stats_no_none.pkl", "rb"))
    final_stats = pickle.load(open("final_stats_no_other.pkl", "rb"))
    # print(final_stats)

    weights_dict = {}
    species = []

    for limit_scope in [None, "Crime and Punishment",
                        "Fairness and Equality", "Public Sentiment"]:
        for media_type in ["independent", "state-affiliated"]:
            category = "All" if limit_scope is None else limit_scope
            category += "-Ind" if media_type == "independent" else "-SA"
            species.append(category)
            stats = final_stats[media_type][limit_scope]
            for frame in frame_of_interests:
                weight_list = weights_dict.get(frame, [])
                weight_list.append(round(stats[frame], 1))
                weights_dict[frame] = weight_list

    fig, ax = plt.subplots()
    bottom = np.zeros(len(species))

    for label, weight in weights_dict.items():
        rects = ax.bar(species, weight, width=0.5, bottom=bottom, label=label)
        # addlabels(species, weight)
        ax.bar_label(rects, label_type='center')
        bottom += weight

    ax.set_xticks(
        np.arange(
            len(species)),
        species,
        rotation=45,
        ha="right",
        rotation_mode="anchor")

    legend = ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.35), ncol=3)

    ax.set_ylabel("Comment Frame Proportion (%)")
    # ax.tick_params(labelbottom=False)

    # plt.tight_layout()
    plt.savefig(
        "stats_no_other.png",
        bbox_extra_artists=(
            legend,
        ),
        bbox_inches='tight')

print()
