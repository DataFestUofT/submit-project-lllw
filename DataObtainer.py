import os
import sys
import json
import csv
import datetime
import pickle
import data_utils
import numpy as np
import pandas as pd
from csv import writer
from csv import reader
from keras.preprocessing.text import Tokenizer
from keras import models
from keras.preprocessing.sequence import pad_sequences
from twarc import Twarc

# Get these four attributes from a twitter developer account
CONSUMER_KEY = ""
CONSUMER_KEY_SECRET = ""
ACCESS_TOKEN = ""
ACCESS_TOKEN_SECRET = ""

#example: tweet_sample.json
#More info: https://developer.twitter.com/en/docs/tweets/data-dictionary/overview/tweet-object
#Temporary
KEYWORD = {"China": ["china", "chinese", "cn", "xi"], "Trump": ["trump"],
 "StayHome": ["wfh", "work from home", "stay at home", "stay home"], "Vaccine": ["vaccine", "vaccination", "vaccinate"],
 "Ventilator": ["ventilator", "ventilated", "ventilation"], "ICU": ["ICU", "intensive care unit"], 
 "Flight": ["flight", "airline", "airplane", "airport"], "Isolation": ["isolation", "social distancing", "quarantine", "isolated"],
 "Border": ["border"],
 "SARS": ["sars", "severe acute respiratory syndromes", "2003"]}

# For each day, we extract DAILY_AMOUNT tweets that satisfy filter()
DAILY_AMOUNT = 100000
START_DATE = "2020-03-03"
END_DATE = "2020-05-31"
# For every HYDRATE_AMOUNT tweets, we hydrate once.
HYDRATE_AMOUNT = 300000


def main():
    t = Twarc(CONSUMER_KEY, CONSUMER_KEY_SECRET, ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
    models = load_model()
    print("-------------------- loading model complete -----------------")
    tweetsClean(models, t)

# result is a 2d matrix, each row represents a tweet result.
# Each row includes a list of three 0/1 value to indicate the sentiment. Order: [neutral postive negative]
def handle_result(result):
    max_index = np.argmax(result, axis=1)
    max_index[max_index==2] = -1
    return max_index.tolist()

# Load pre-trained models
def load_model():
    CNN_model = models.load_model(
        "./ModelCode/CNN.h5", custom_objects=None
    )
    print("----- CNN load successfully -----")

    LSTM_model = models.load_model(
        "./ModelCode/LSTM.h5", custom_objects=None
    )
    print("----- LSTM load successfully ------")

    DecisionTree_model = pickle.load(
        open("./ModelCode/DecisionTree.pickle", "rb")
    )
    print("----- DT load successfully -----")

    RandomForest_model = pickle.load(
        open("./ModelCode/RFmodel.pickle", "rb")
    )
    print("RF load successfully")

    DTVectorizer = pickle.load(
        open("./ModelCode/DTvectorizer.pickle", "rb")
    )
    print("------ DTVec load successfully ------")

    RFVectorizer = pickle.load(
        open("./ModelCode/RFvec.pickle", "rb")
    )
    print("------ RFVec load successfully ------")

    return {
        "CNN": CNN_model, "LSTM": LSTM_model, 
        "DecisionTree": DecisionTree_model, "RandomForest": RandomForest_model,
        "DTVectorizer": DTVectorizer, "RFVectorizer": RFVectorizer} 


def filter(tweet):
    # only includes English tweets that is not retweet or reply
    return tweet["lang"] == "en" and not tweet["is_quote_status"] and tweet["in_reply_to_status_id_str"] is None

# Get a date list of all dates between START_DATE and END_DATE
# Eg: ["2020-03-30", "2020-03-31", "2020-04-01", "2020-04-02"] if START_DATE = "2020-03-30" AND END_DATE = "2020-04-02"
def get_dates():
    s_date = START_DATE.split("-")
    e_date = END_DATE.split("-")
    d1 = datetime.date(int(s_date[0]), int(s_date[1]), int(s_date[2]))
    d2 = datetime.date(int(e_date[0]), int(e_date[1]), int(e_date[2]))
    delta = d2 - d1
    return [str(d1 + datetime.timedelta(days=i)) for i in range(delta.days + 1)]
    

def tweetsClean(models, t):
    #output csv
    print("------ extract data ------")
    cur = 2
    dates = get_dates()
    print(START_DATE, END_DATE)
    amount = 0
    # tweet id lists are downloaded from https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/LW0BTB

    # source files and results are located in google drive: 
    # https://drive.google.com/drive/folders/14LFNyMiO_RWGbJIshRu-U7yBhgW1iP-M?usp=sharing
    f = open("./TweetData/" + str(cur) + ".txt", "r")
    for date in dates:
        print("curdate: ", date)
        print("cur_file: ", "./TweetData/" + str(cur) + ".txt")
        fname = "./result/" + date + ".csv"
        csvfile = open(fname, "w", newline="", encoding="utf-8")
        writer = csv.DictWriter(csvfile, fieldnames=["id","date","text"] + [i for i in KEYWORD])
        attr = {"id": "id", "date": "date", "text": "text"}
        for i in KEYWORD:
            attr[i] = i
        writer.writerow(attr)
        line = f.readline()
        l = 0
        enough = False
        print("start read")
        while not enough:
            t_date = get_tweet_timestamp(line)
            a = open("tmp.txt", "a")
            if t_date == date:
                a.write(line)
                l += 1
                if l % 10000 == 0:
                    # print current process
                    print(date, l)
                if l == HYDRATE_AMOUNT:
                    print("hydrate: ", date)
                    a = open("tmp.txt", 'r')
                    tweets = t.hydrate(a)
                    k = 0
                    for tweet in tweets:
                        if filter(tweet):
                            data = {"id": tweet["id"], "date": t_date, "text": tweet["full_text"]}
                            text = tweet["full_text"]
                            write_data(data, text, writer)
                            amount += 1
                            if amount == DAILY_AMOUNT:
                                csvfile.close()
                                print("------ complete date: " + t_date + " ------")
                                # analyze data and switch to next date            
                                data_analysis(fname, models)
                                enough = True
                                amount = 0
                                l = 0
                                break
                        k += 1
                        if k % 1000 == 0:
                            # print current process
                            print(k, amount)
                    a.close()
                    os.remove("tmp.txt")
                    l = 0            
            line = f.readline()
            if line == "":
                cur += 1
                # finish reading one id list, switch to next one
                f = open("./TweetData/" + str(cur) + ".txt", "r")
                print("change file: ", "./TweetData/" + str(cur) + ".txt")
                line = f.readline()


# function source: https://ws-dl.blogspot.com/2019/08/2019-08-03-tweetedat-finding-tweet.html
def get_tweet_timestamp(tid):
    tid = int(tid.strip())
    offset = 1288834974657
    tstamp = (tid >> 22) + offset
    utcdttime = datetime.datetime.utcfromtimestamp(tstamp/1000)
    return str(utcdttime.date())


# sentiment analysis using four models
def data_analysis(fname, models):
    data = pd.read_csv(fname, encoding="utf-8")
    print("analysis: " + fname)
    data = data[["text"]]
    data = data_utils.clean_text(data)
    data["text"] = data["text"].str.replace('[^A-Za-z ]+', "")
    train_data = pd.read_csv('./Training Data/Sentiment.csv')
    data_utils.clean_text(train_data)
    train_data["text"] = train_data["text"].str.replace('[^A-Za-z ]+', "")


    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_data["text"])
    sequences = tokenizer.texts_to_sequences(data["text"])
    tweets_pad = pad_sequences(sequences, maxlen=29, padding="post")

    CNN_result = models["CNN"].predict(tweets_pad)
    print("----- CNN complete -----")
    print("CNN_result: "+ str(CNN_result.shape))
    CNN_result = handle_result(CNN_result)
    CNN_result.insert(0, "CNN")
    add_column_in_csv(fname, 'result2.csv', CNN_result)

    LSTM_result = models["LSTM"].predict(tweets_pad)
    print("----- LSTM complete -----")
    print("LSTM_result: "+ str(LSTM_result.shape))
    LSTM_result = handle_result(LSTM_result)
    LSTM_result.insert(0, "LSTM")
    add_column_in_csv('result2.csv', 'result3.csv', LSTM_result)

    vec = models["DTVectorizer"].transform(data["text"])
    DT_result = models["DecisionTree"].predict(vec)
    print("----- DT complete -----")
    print("DT_result: " + str(DT_result.shape))
    DT_result = DT_result.tolist()
    DT_result.insert(0, "DT")
    add_column_in_csv("result3.csv", "result4.csv", DT_result)

    vec = models["RFVectorizer"].transform(data["text"])
    RF_result = models["RandomForest"].predict(vec)
    print("----- RF complete -----")
    print("RF_result: " + str(RF_result.shape))
    RF_result = RF_result.tolist()
    RF_result.insert(0, "RF")
    add_column_in_csv("result4.csv", fname, RF_result)
    
    print("----- " + fname + " analysis complete -----")
    os.remove("result2.csv")
    os.remove("result3.csv")
    os.remove("result4.csv")

# find keyword occurence
def write_data(data, text, writer):
    text = text.lower()
    for i in KEYWORD:
        data[i] = 1 if any([j in text.split() for j in KEYWORD[i]]) else 0        
    writer.writerow(data)
    return 


#src: https://thispointer.com/python-add-a-column-to-an-existing-csv-file/
# append one model's result
def add_column_in_csv(input_file, output_file, data):
    """ Append a column in existing csv using csv.reader / csv.writer classes"""
    # Open the input_file in read mode and output_file in write mode
    with open(input_file, 'r', encoding="utf-8") as read_obj, \
            open(output_file, 'w', newline='', encoding="utf-8") as write_obj:
        # Create a csv.reader object from the input file object
        csv_reader = reader(read_obj)
        # Create a csv.writer object from the output file object
        csv_writer = writer(write_obj)
        # Read each row of the input csv file as list
        i = 0
        for row in csv_reader:
            # Pass the list / row in the transform function to add column text for this row
            row.append(data[i])
            # Write the updated row / list to the output file
            csv_writer.writerow(row)
            i += 1
            if i % 1000 == 0:
                print(input_file, " ----> ", output_file, ": ", i)


if __name__ == "__main__":
    main()
    exit()
