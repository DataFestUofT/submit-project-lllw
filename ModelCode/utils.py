from twarc import Twarc
import pandas as pd


def clean_text(df):
    for i in range(df['text'].shape[0]):
        df['text'][i] = " ".join([word for word in df['text'][i].split()
                                    if 'http' not in word
                                    and not word.startswith('@')
                                    # and not word.startswith('#')
                                    and word != 'RT'
                                    ])
    df['text'] = df['text'].str.replace('[^A-Za-z ]+', '')

    return df

def load_embedding_model(size):
    """ Load GloVe Vectors
        Return:
            wv_from_bin: All 400000 embeddings, each lengh 200
    """
    import gensim.downloader as api
    wv_from_bin = api.load("glove-wiki-gigaword-"+str(size))
    print("Loaded vocab size %i" % len(wv_from_bin.vocab.keys()))
    return wv_from_bin


def pull_tweet(input_file_name):
    CONSUMER_KEY = "9At2u3Y2DraTHLSg3D9w6LhE9"
    CONSUMER_KEY_SECRET = "DRFCbI2t0gMhfV2KnEub6cljowW9zRwmkeMJ0GT9MlMkrkzspM"
    ACCESS_TOKEN = "1259913765614751745-LwtSI48si3sYekzvxW86syIFsRgirl"
    ACCESS_TOKEN_SECRET = "e0gpJdT0IXOSxFrhplKMl8FlP0dVnuLg1vwBHzt5Fc9J9"

    t = Twarc(CONSUMER_KEY, CONSUMER_KEY_SECRET, ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
    inputF = open(input_file_name, "r")
    line = inputF.readline()
    data = []
    i=0
    while line != "" and i<10:
        try:
            tweet = t.tweet(line.strip())
            if tweet["lang"] == "en":
                if 'retweeted_status' in tweet.keys():
                    data.append(tweet['retweeted_status']['full_text'].replace('\n', ' '))
                else:
                    data.append(data, tweet['full_text'].replace('\n', ' '))
                i += 1
            line = inputF.readline()
        except Exception as e:
            line = inputF.readline()
    return data


def write_tweet_to_csv(input_file_name, output_file_name):
    data = pull_tweet(input_file_name)
    df = pd.DataFrame(data, columns=[input_file_name[-17:-4]])
    df.to_csv(output_file_name, index=False)
