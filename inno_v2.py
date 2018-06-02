import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
import lightgbm as lgb
from sklearn.cross_validation import train_test_split
from nltk.corpus import stopwords
import nltk
import ast
import re, math
from collections import Counter
from nltk.stem import PorterStemmer

ps = PorterStemmer()
WORD = re.compile(r'\w+')
stop_words = set(stopwords.words("english"))

def get_cosine_new(vec1, vec2):
    try:
        present_count = 0
        total_count = 0
        for word,count in vec1.items():
            total_count += count
            if word in vec2:
                present_count += count
        return present_count/max(total_count,1)
    except:
        print("Exception")
        return 0.0


def text_to_vector(text,type=None):
    if not type:
        words = WORD.findall(text)
        final_words = [ps.stem(word.lower()) for word in words if word.lower() not in stop_words]

    else:
        final_words = text.lower().split(",")
    return Counter(final_words)


def format_data(inp_df):
    inp_df["vector_abstract"] = inp_df["abstract"].apply(lambda x: text_to_vector(x))
    inp_df["vector_title"] = inp_df["article_title"].apply(lambda x: text_to_vector(x))
    inp_df["vector_author"] = inp_df["author_str"].apply(lambda x: text_to_vector(str(x),type="Author"))
    return inp_df


def find_relation(inp_df):
    relation_df = []
    inp_df["pub_date"] = pd.to_datetime(inp_df["pub_date"])
    for i in range(len(inp_df)):
        print(i)
        set_df = inp_df[(inp_df['set']==inp_df['set'][i]) & (inp_df["pub_date"]<=inp_df["pub_date"][i])].reset_index()
        for j in range(len(set_df)):
            rel_df = dict()
            try:
                rel_df["cosine_abstract"] = get_cosine_new(inp_df['vector_abstract'][i],set_df['vector_abstract'][j])
            except:
                rel_df["cosine_abstract"] = 0.0
            try:
                rel_df["cosine_title"] = get_cosine_new(inp_df['vector_title'][i],set_df['vector_title'][j])
            except:
                rel_df["cosine_title"] = 0.0
            try:
                rel_df["cosine_author"] = get_cosine_new(inp_df['vector_author'][i],set_df['vector_author'][j])
            except:
                rel_df["cosine_author"] = 0.0
            rel_df["date_diff"] = (inp_df["pub_date"][i] - set_df["pub_date"][j]).days
            try:
                rel_df["cosine_abs_title"] = get_cosine_new(inp_df['vector_abstract'][i],set_df['vector_title'][j])
            except:
                rel_df["cosine_abs_title"] = 0.0
            try:
                rel_df["cosine_title_abs"] = get_cosine_new(inp_df['vector_title'][i],set_df['vector_abstract'][j])
            except:
                rel_df["cosine_title_abs"] = 0.0

            rel_df["related_pmid"] = set_df["pmid"][j]
            rel_df["pmid"] = inp_df["pmid"][i]
            relation_df.append(rel_df)
    final_df = pd.DataFrame(relation_df)
    return final_df

def mark_relation(inp_df,train):
    inp_df["related"] = 0
    for i in range(len(train)):
        print(i)
        inp_df.loc[(inp_df["related_pmid"].isin(ast.literal_eval(train["ref_list"][i])) & (inp_df["pmid"]==train["pmid"][i])),"related"] = 1
        print(inp_df["related"].value_counts())
    return inp_df


def format_sub(inp_df,filter,final_sub):
    print(filter)
    unique_pmid = inp_df['pmid'].unique()
    sel_df = inp_df[(inp_df["related"]>filter) & (inp_df["related_pmid"]!=inp_df["pmid"])]
    if len(sel_df) > 0:
        sel_df = sel_df.drop(["related"],axis=1)
        sel_df1 = sel_df.groupby('pmid')['related_pmid'].apply(list).reset_index()
        sel_df1.columns = ['pmid','ref_list']
        final_sub = pd.concat([final_sub,sel_df1],axis=0)
        print(len(final_sub))
        sel_pmid = sel_df['pmid'].unique()
    else:
        sel_pmid = []
    left_out_list = list(set(unique_pmid) - set(sel_pmid))
    if filter < 0.0001:
        left_out_df = inp_df[inp_df['pmid'].isin(left_out_list)]
        return final_sub, left_out_df, left_out_list
    elif left_out_list:
        print(len(left_out_list))
        left_out_df = inp_df[inp_df['pmid'].isin(left_out_list)]
        filter = filter/1.5
        return format_sub(left_out_df,filter,final_sub)
    return final_sub, inp_df, []


if __name__ == '__main__':
    train = pd.read_csv('D:/Docs/AV/innoplexus/train.csv',sep=",")
    info_train = pd.read_csv('D:/Docs/AV/innoplexus/information_train.csv',sep="\t")
    test = pd.read_csv("D:/Docs/AV/innoplexus/test.csv",sep=",")
    info_test = pd.read_csv('D:/Docs/AV/innoplexus/information_test.csv',sep="\t")

    train_vector = format_data(info_train)
    train_rel_vector = find_relation(train_vector)
    newtrain = mark_relation(train_rel_vector,train)
    newtrain_correct = newtrain[newtrain["pmid"]!=newtrain["related_pmid"]]

    test_vector = format_data(info_test)
    test_rel_vector = find_relation(test_vector)
    newtest = test_rel_vector

    X = newtrain_correct.drop(['pmid', 'related_pmid','related'], axis=1)
    features = X.columns
    X = X.values
    y = newtrain_correct['related'].values

    sub = newtest[['pmid','related_pmid']]
    sub['related'] = 0
    nrounds = 4000

    params = {'eta': 0.05, 'max_depth': 6, 'subsample': 0.8, 'colsample_bytree': 1,
              'objective': 'binary:logistic', 'eval_metric': 'auc', 'silent': True}

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=1)
    d_train = xgb.DMatrix(X_train, y_train)
    d_valid = xgb.DMatrix(X_valid, y_valid)
    watchlist = [(d_train, 'train'), (d_valid, 'valid')]
    xgb_model = xgb.train(params, d_train, nrounds, watchlist, early_stopping_rounds=50,
                              maximize=True, verbose_eval=50)
    sub['related'] = xgb_model.predict(xgb.DMatrix(newtest[features].values),
                                           ntree_limit=xgb_model.best_ntree_limit)
    final_sub = pd.DataFrame(columns=['pmid','ref_list'])
    submission, left_df, left = format_sub(sub,0.12,final_sub)
    if left:
        left_df = left_df.reset_index()
        unique_pmid = left_df['pmid'].unique()
        for pmid in unique_pmid:
            sel = dict(pmid=pmid, ref_list=[])
            print(sel)
            submission = pd.concat([submission, pd.DataFrame([sel])], axis=0)
    submission.to_csv("D:/Docs/AV/innoplexus/sub.csv",index=False)