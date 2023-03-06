import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
from tqdm import tqdm
from sklearn import linear_model
from config import DATA_DIR


frames = ['Capacity and Resources',
          'Crime and Punishment',
          'Economic',
          'Security and Defense',
          'Policy Prescription and Evaluation',
          'Legality, Constitutionality, Jurisdiction',
          'Health and Safety',
          'Cultural Identity',
          'Political',
          'Quality of Life',
          'Fairness and Equality',
          ' Morality',
          'Public Sentiment',
          'External Regulation and Reputation']

PREPROCESS_FLAG = False
REG_FRAME_OWN_FLAG = True


def parse_x(frame, ownership, info):
    '''
    x: dim of 18, 14 frames + ownership (independent:0, sa: 1) + has image/video/link (audio, podcast not included)
    '''
    # frame = info["frame"]
    # ownership = info["ownership"]
    has_image = int(info["has_image"])
    has_video = int(info["has_video"])
    has_link = int(info["has_link"])
    return parse_frame(frame) + [ownership, has_image, has_video, has_link]


def parse_frame(frame):
    if frame in ["Other", None]:
        return [0] * 14
    else:
        return [1 if frame == f else 0 for f in frames]


def preprocess():
    X = []
    y_view = []
    y_like = []
    y_repost = []
    y_engagement = []

    for vk_media_type in ["independent", "state-affiliated"]:
        # print(len(X))
        with open(Path(DATA_DIR) / "processed-vk" /
                  vk_media_type / "tids2info.pkl", "rb") as f:
            vk_tids2info = pickle.load(f)

        with open(Path(DATA_DIR) / "processed-vk" /
                  vk_media_type / "tids2mfcframe.pkl", "rb") as f:
            vk_tids2mfcframe = pickle.load(f)

        for tid, info in tqdm(vk_tids2info.items()):
            x = parse_x(
                vk_tids2mfcframe.get(
                    tid, None), int(
                    vk_media_type == "state-affiliated"), info)
            y_view.append(-1 if not info["#views"] else np.log(info["#views"]))
            y_like.append(-1 if not info["#likes"] else np.log(info["#likes"]))
            y_repost.append(-1 if not info["#reposts"] else np.log(info["#reposts"]))
            y_engagement.append(
                0 if not info["#views"] else (info["#comments"] +
                                              info["#likes"] +
                                              info["#reposts"]) /
                info["#views"])
            X.append(x)

    # with open(Path(data_dir) / "processed" /
    #         "tids2info.pkl", "rb") as f:
    #     twitter_tids2info = pickle.load(f)

    # with open(Path(data_dir) / "processed" /
    #         "tids2independent.pkl", "rb") as f:
    #     twitter_tids2independent = pickle.load(f)

    # with open(Path(data_dir) / "processed" /
    #         "tids2state-affiliated.pkl", "rb") as f:
    #     twitter_tids2sa = pickle.load(f)

    # with open(Path(data_dir) / "processed" /
    #         "tids2mfcframe.pkl", "rb") as f:
    #     twitter_tids2mfcframe = pickle.load(f)

    print()

    with open("regression_features.pkl", "wb") as f:
        pickle.dump((X, y_view, y_like, y_repost, y_engagement), f)


if __name__ == "__main__":
    if PREPROCESS_FLAG:
        preprocess()
    else:
        with open("regression_features.pkl", "rb") as f:
            X, y_view, y_like, y_repost, y_engagement = pickle.load(f)
        if REG_FRAME_OWN_FLAG:
            weights = []
            X_ownership = (1-np.array(X)[:, 14]).reshape(-1, 1)
            for frame_idx, frame in enumerate(frames):
                y = np.array(X)[:, frame_idx]
                print("#" * 20)
                model = linear_model.LogisticRegression().fit(
                    X_ownership, y)
                print(frame, model.coef_[0][0])
                weights.append(model.coef_[0][0])
                print("#" * 20)
                print()

            plt.scatter(np.arange(len(weights)), weights)
            for idx, weight in enumerate(weights):
                plt.text(idx, weight+0.05, frames[idx].split()[0].replace(',', ''), color='r')

            plt.text(0, 1.0, "Independent", va="center", ha="center", size=15, bbox=dict(boxstyle="square",
                   ec=(1., 0.5, 0.5),
                   fc=(1., 0.8, 0.8),
                   ))

            plt.text(0, -0.4, "State-Affiliated", va="center", ha="center", size=15, bbox=dict(boxstyle="square",
                   ec=(1., 0.5, 0.5),
                   fc=(1., 0.8, 0.8),
                   ))

            plt.tight_layout()
            plt.savefig("frame_ownership.png")
            print()

        else:
            feature_name = frames + ["is state-affilated",
                                    "Has Image", "Has Video", "Has Link"]
            for y in [y_view, y_like, y_repost, y_engagement]:
                print("#" * 20)
                model = linear_model.LinearRegression().fit(X, y)
                print(model.coef_)
                salient_feature_idx = np.argsort(model.coef_)[::-1][:5]
                for i in range(5):
                    print(model.coef_[salient_feature_idx[i]],
                        feature_name[salient_feature_idx[i]])
                print("#" * 20)
                print()

