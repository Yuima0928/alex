import sqlite3
import numpy as np
import pickle

# ここtopicの数が増える場合を考慮したほうがいいかも
domain_rank = {i: 0 for i in range(1, 6)}
field_rank = {i: 0 for i in range(1, 28)}
subfield_rank = {i: 0 for i in range(1, 246)}

number = 10000 # ここは適宜変える
old_year = 2019 # ここは適宜変える

def define_topic_rank(number, old_year):
    # データベース接続
    db_path = './paper2.db'
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()

    cursor.execute("SELECT paper_id FROM PaperVectors WHERE created_date <= '2019-01-01' LIMIT ?", (number,))
    paper_ids = cursor.fetchall()

    for i in range(len(paper_ids)):
        cursor.execute("SELECT domain, field, subfield FROM Topics WHERE paper_id = ?", paper_ids[i])
        rows = cursor.fetchall()

        if rows:
            for row in rows:
                domain, field, subfield = row
                domain_rank[domain] += 1
                field_rank[field] += 1
                subfield_rank[subfield] += 1

define_topic_rank(number, old_year)

print(domain_rank)

# 結果をファイルに保存
with open('topic_ranks_2019_new.pkl', 'wb') as f: #ここも適宜かえる
    pickle.dump({
        'domain_rank': domain_rank,
        'field_rank': field_rank,
        'subfield_rank': subfield_rank
    }, f)

print("Ranks saved to 'topic_ranks_2019_new.pkl'")
