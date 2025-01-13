import sqlite3
import numpy as np
import math
from sentence_transformers import SentenceTransformer
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.sparse import lil_matrix, csr_matrix
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
import joblib
from collections import defaultdict
import pickle
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from datetime import date, timedelta
from datetime import datetime
import multiprocessing
import numpy as np
from functools import partial
import random
# 5年後の引用数予測

num_processes = multiprocessing.cpu_count()

old_year = 2019
#学習データとテストデータ10000ずつ

# データベース接続
db_path = './paper2.db'
connection = sqlite3.connect(db_path)
cursor = connection.cursor()

# ファイルからデータを読み込み
with open('topic_ranks_2019_new.pkl', 'rb') as f:
    ranks = pickle.load(f)

# 読み込んだデータを利用
domain_rank = ranks['domain_rank']
field_rank = ranks['field_rank']
subfield_rank = ranks['subfield_rank']
print(field_rank)

output_file = 'institution_rank.pkl'
# ファイルからデータを読み込み
with open(output_file, 'rb') as f:
    institution_rank = pickle.load(f)

def calculate_pagerank_sparse(a_vector, paper_number, damping_factor=0.85, epsilon=1e-6, max_iter=100):
    # 疎行列の構築
    g_matrix = lil_matrix((paper_number, paper_number), dtype=np.float32)
    for i, sub_a_vector in enumerate(a_vector):
        for target_idx, weight in sub_a_vector:
            g_matrix[i, target_idx] = weight

    # 正規化（行方向の合計を1にする）
    row_sums = g_matrix.sum(axis=1).A.flatten()
    row_sums[row_sums == 0] = 1  # 0割を防ぐ
    for i in range(paper_number):
        g_matrix[i, :] /= row_sums[i]

    # 疎行列の形式をCSRに変換（効率的な計算のため）
    g_matrix = csr_matrix(g_matrix)

    # 初期のPageRankベクトル
    pagerank_vector = np.ones(paper_number, dtype=np.float32) / paper_number

    print(paper_number)
    # PageRank計算
    for _ in range(max_iter):
        new_pagerank_vector = damping_factor * g_matrix.T @ pagerank_vector + (1 - damping_factor) / paper_number
        if np.linalg.norm(new_pagerank_vector - pagerank_vector) < epsilon:
            break
        pagerank_vector = new_pagerank_vector

    return pagerank_vector

def calculate_pagerank_vector(paper_ids, author_ids, institution_ids):
    # PageRank計算準備
    authority_dic = defaultdict(int) 
    venue_centrality_dic = defaultdict(int) 
    paper_dict = {paper_id: idx for idx, paper_id in enumerate(paper_ids)}
    a_vector = [[] for _ in range(len(paper_ids))]
    sum_edge_per_paper = [0 for _ in range(len(paper_ids))]

    for paper_id in paper_ids:
        cursor.execute('SELECT referenced_paper_id FROM ReferencedWorks WHERE paper_id = ?', (paper_id,))
        reference_works = [r[0] for r in cursor.fetchall()]
        for ref_id in reference_works:
            if ref_id in paper_dict:
                if ref_id == paper_id:
                    continue
                sum_edge_per_paper[paper_dict[paper_id]] += 1
                a_vector[paper_dict[paper_id]].append([paper_dict[ref_id], 1])

    # PageRank計算（疎行列版）
    pagerank_vector = calculate_pagerank_sparse(a_vector, len(paper_ids))

    for i in range(len(paper_ids)):
        authority_dic[author_ids[i]] += pagerank_vector[i]
        venue_centrality_dic[institution_ids[i]] += pagerank_vector[i]
        
    return authority_dic, venue_centrality_dic

def pick_up_x_data(paper_ids, author_ids, institution_ids, authority_dic, venue_centrality_dic, cursor):
    x_data = []

    i = 0
    for paper_id in paper_ids:
        cursor.execute("SELECT topics_count, created_date FROM PaperVectors WHERE paper_id = ?", (paper_id,))
        topics_count, created_date = cursor.fetchone()
        if not topics_count:
            topics_count = 0
        
        # created_date を datetime オブジェクトに変換
        if isinstance(created_date, str):
            created_date = datetime.strptime(created_date, '%Y-%m-%d')
        one_month_after = created_date + timedelta(days=30)

        cursor.execute("SELECT COUNT(*) FROM ReferencedWorks WHERE referenced_paper_id = ? AND created_date <= ?", (paper_id, created_date))
        cited_after_one_month = cursor.fetchone()[0]

        print(i)
        cursor.execute("SELECT referenced_paper_id FROM ReferencedWorks WHERE paper_id = ?", (paper_id,))
        reference_works = [r[0] for r in cursor.fetchall()]

        cite_good_paper = 0
        cite_20_over_paper = 0
        cite_50_over_paper = 0
        cite_100_over_paper = 0
        max_citation = 0
        sum_citation = 0
        cite_recent_paper = 0
        for reference_work in reference_works:
            six_months_ago = created_date - timedelta(days=180)
            cursor.execute("SELECT created_date FROM PaperVectors WHERE paper_id = ? AND created_date >= ?", (reference_work, six_months_ago))
            result = cursor.fetchone()
            if result:
                cite_recent_paper += 1
            print(reference_work)
            cursor.execute("SELECT COUNT(*) FROM ReferencedWorks WHERE referenced_paper_id = ? AND created_date <= ?", (reference_work, created_date))
            citation_result = cursor.fetchone()
            if (citation_result[0] > 20):
                cite_20_over_paper += 1
            if (citation_result[0] > 50):
                cite_50_over_paper += 1
            if (citation_result[0] > 100):
                cite_100_over_paper += 1
            if (max_citation < citation_result[0]):
                max_citation = citation_result[0]
            sum_citation += citation_result[0]
        ave_citation = sum_citation / len(reference_works) if len(reference_works) else 0  

        h_index, author_total_cited_by_count, author_total_works_count = 0,0,0
        if (author_ids[i]):
            cursor.execute("SELECT h_index, cited_by_count, works_count FROM Authors WHERE author_id = ?", (author_ids[i],))
            author_data = cursor.fetchone()
            if author_data:
                h_index, author_total_cited_by_count, author_total_works_count = author_data
            else:
                h_index, author_total_cited_by_count, author_total_works_count = 0, 0, 0

        
        institution_total_cited_by_count, institution_total_works_count = 0,0
        if (institution_ids[i]):
            cursor.execute("SELECT cited_by_count, works_count FROM Institutions WHERE institution_id = ?", (institution_ids[i],))
            institution_data = cursor.fetchone()
            if institution_data:
                institution_total_cited_by_count, institution_total_works_count = institution_data
            else:
                institution_total_cited_by_count, institution_total_works_count = 0, 0
        
        author_recent_cited_by_count = 0
        author_recent_works_count = 0
        if (author_ids[i]):
            cursor.execute("SELECT cited_by_count, works_count FROM AuthorsYearCount WHERE author_id = ? AND year < ?", (author_ids[i], old_year))
            rows =  cursor.fetchall()
            for row in rows:
                author_recent_cited_by_count += row[0]
                author_recent_works_count += row[1]

        institution_recent_cited_by_count = 0
        institution_recent_works_count = 0
        if (institution_ids[i]):
            cursor.execute("SELECT cited_by_count, works_count FROM InstitutionsYearCount WHERE institution_id = ? AND year < ?", (institution_ids[i], old_year))
            rows =  cursor.fetchall()
            for row in rows:
                institution_recent_cited_by_count += row[0]
                institution_recent_works_count += row[1]

        author_cited_by_count = author_recent_cited_by_count
        author_works_count = author_recent_works_count
        
        if (author_cited_by_count < 0):
            print(author_id)
            print(author_cited_by_count)

        
        institution_cited_by_count =  institution_recent_cited_by_count
        institution_works_count = institution_recent_works_count

        # データを計算
        author_cited_by_ratio = author_cited_by_count / author_works_count if author_works_count else 0
        
        institution_cited_by_ratio = institution_cited_by_count / institution_works_count if institution_works_count else 0
        
        # # 必要なデータを追加
        x_data.append([
            cited_after_one_month,
            cite_20_over_paper,
            cite_50_over_paper,
            cite_100_over_paper,
            max_citation,
            ave_citation,
            cite_recent_paper,
            topics_count,
            h_index,
            author_cited_by_ratio,
            institution_cited_by_ratio
        ])
        i += 1
    for i in range(len(paper_ids)):
        cursor.execute("SELECT domain, field, subfield FROM Topics WHERE paper_id = ?", (paper_ids[i],))
        topic_rows = cursor.fetchall()

        # primary_topicのfieldでランク付け（仮に field_rank が事前定義されている場合）
        topic_rank = 0
        for topic_row in topic_rows:
            topic_rank += field_rank.get(topic_row[1], 0)
        topic_rank = topic_rank / len(topic_rows) if len(topic_rows) else 0 # すべてのfieldのランクの平均を取る

        authority = authority_dic[author_ids[i]]
        venue_centrality = venue_centrality_dic[institution_ids[i]]
        x_data[i].extend([topic_rank, authority, venue_centrality]) 
    return x_data

import multiprocessing
import numpy as np
from functools import partial

# データベース接続関数（プロセスごとに新しい接続が必要）
def connect_to_db():
    db_path = './paper2.db'
    return sqlite3.connect(db_path)

# 分割されたチャンクでX_dataを作成する関数
def process_chunk(chunk_args):
    paper_ids_chunk, author_ids_chunk, institution_ids_chunk, authority_dic, venue_centrality_dic = chunk_args
    connection = connect_to_db()
    cursor = connection.cursor()
    x_data_chunk = pick_up_x_data(paper_ids_chunk, author_ids_chunk, institution_ids_chunk, authority_dic, venue_centrality_dic, cursor)
    connection.close()
    return x_data_chunk

# 並列処理用のpick_up_x_data関数（cursorを引数として受け取るよう変更）
def pick_up_x_data_parallel(paper_ids, author_ids, institution_ids, authority_dic, venue_centrality_dic, num_processes=num_processes):
    # データをチャンクに分割
    chunks = np.array_split(list(zip(paper_ids, author_ids, institution_ids)), num_processes)
    chunk_args = [
        (
            [chunk[i][0] for i in range(len(chunk))],  # paper_ids
            [chunk[i][1] for i in range(len(chunk))],  # author_ids
            [chunk[i][2] for i in range(len(chunk))],  # institution_ids
            authority_dic,
            venue_centrality_dic
        )
        for chunk in chunks
    ]

    # 並列処理
    with multiprocessing.Pool(num_processes) as pool:
        results = pool.map(process_chunk, chunk_args)

    # 結果を結合
    x_data = []
    for result in results:
        x_data.extend(result)

    return x_data

def pick_up_y_data(paper_number, old_year):
    y_train, y_test = [], []
    paper_ids_train, paper_ids_test = [], []
    author_ids_train, author_ids_test = [], []
    institution_ids_train, institution_ids_test = [], []

    # データベースの総データ件数を取得
    cursor.execute(f"""
        SELECT COUNT(*) 
        FROM PaperVectors 
        WHERE created_date BETWEEN '{old_year}-01-01' AND '{old_year}-12-31 23:59:59'
    """)
    total_records = cursor.fetchone()[0]

    # ランダムなOFFSETリストを作成
    offsets = random.sample(range(total_records), min(total_records, paper_number * 2))

    count = 0
    for offset in offsets:
        cursor.execute(f"""
            SELECT paper_id, author_id, institution_id 
            FROM PaperVectors 
            WHERE created_date BETWEEN '{old_year}-01-01' AND '{old_year}-12-31 23:59:59'
            LIMIT 1 OFFSET ?
        """, (offset,))
        row = cursor.fetchone()
        if not row:
            continue

        paper_id, author_id, institution_id = row

        cursor.execute("SELECT COUNT(*) FROM ReferencedWorks WHERE referenced_paper_id = ?", (paper_id,))
        y = cursor.fetchone()[0]

        if count < paper_number * 1.5:
            y_train.append(y)
            paper_ids_train.append(paper_id)
            author_ids_train.append(author_id)
            institution_ids_train.append(institution_id)
            count += 1
        else:
            y_test.append(y)
            paper_ids_test.append(paper_id)
            author_ids_test.append(author_id)
            institution_ids_test.append(institution_id)
            count += 1

        if count >= paper_number * 2:  # 必要なデータ数に達したら終了
            break

    return paper_ids_train, paper_ids_test, y_train, y_test, author_ids_train, author_ids_test, institution_ids_train, institution_ids_test

paper_number = 10000
# データ取得
base = 10

paper_ids_train, paper_ids_test, y_train, y_test, author_ids_train, author_ids_test, institution_ids_train, institution_ids_test = pick_up_y_data(paper_number, old_year)



# 実行例
authority_dic_train, venue_centrality_dic_train = calculate_pagerank_vector(paper_ids_train, author_ids_train, institution_ids_train)
X_train = pick_up_x_data_parallel(paper_ids_train, author_ids_train, institution_ids_train, authority_dic_train, venue_centrality_dic_train, num_processes)

authority_dic_test, venue_centrality_dic_test = calculate_pagerank_vector(paper_ids_test, author_ids_test, institution_ids_test)
X_test = pick_up_x_data_parallel(paper_ids_test, author_ids_test, institution_ids_test, authority_dic_test, venue_centrality_dic_test, num_processes)

# 保存するデータ
data_to_save = {
    "paper_ids_train": paper_ids_train,
    "paper_ids_test": paper_ids_test,
    "y_train": y_train,
    "y_test": y_test,
    "author_ids_train": author_ids_train,
    "author_ids_test": author_ids_test,
    "institution_ids_train": institution_ids_train,
    "institution_ids_test": institution_ids_test,
    "X_train": X_train,
    "X_test": X_test
}

y_train = [math.log(y + 1, base) for y in y_train]
y_test = [math.log(y + 1, base) for y in y_test]

# データクリーンアップ関数
def clean_data(data):
    cleaned_data = []
    for row in data:
        cleaned_row = [
            float(value) if value not in [None, '', 'NaN'] and not (isinstance(value, float) and math.isnan(value)) else 0
            for value in row
        ]
        cleaned_data.append(cleaned_row)
    return cleaned_data

connection.close()

# 特徴量名を設定
feature_names = ["cited_after_one_month", "cite_20_over_paper", "cite_50_over_paper", "cite_100_over_paper", "max_citation", "ave_citation", "cite_recent_paper", "topics_count", "h_index", "author_rank", "institution_rank", "topic_rank", "authority", "venue_centrality"]
# 特徴量と引用数のグラフをプロット
for i, feature_name in enumerate(feature_names):
    plt.figure(figsize=(8, 6))
    plt.scatter([x[i] for x in X_test], y_test, alpha=0.5, color='blue')
    plt.xlabel(feature_name, fontsize=12)
    plt.ylabel("Citation Count (y_test)", fontsize=12)
    plt.title(f"Feature '{feature_name}' vs Citation Count", fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"feature_vs_citations_{feature_name}_new2.png", dpi=300, format='png')
    plt.close()

    print(f"グラフが 'feature_vs_citations_{feature_name}_new2.png' に保存されました。")

# ファイルに保存
with open('saved_data3.pkl', 'wb') as f:
    pickle.dump(data_to_save, f)

print("データが'saved_data3.pkl'に保存されました。")

# ファイルからデータをロード
with open('saved_data3.pkl', 'rb') as f:
    loaded_data = pickle.load(f)
# データ取得
base = 10
paper_ids_train = loaded_data["paper_ids_train"]
paper_ids_test = loaded_data["paper_ids_test"]
y_train = loaded_data["y_train"]
y_train_log = [math.log(y + 1, base) for y in y_train]
y_test = loaded_data["y_test"]
y_test_log = [math.log(y + 1, base) for y in y_test]
X_train = loaded_data["X_train"]
X_test = loaded_data["X_test"]

# データクリーンアップ
X_train = clean_data(X_train)
X_test = clean_data(X_test)

# データ正規化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# スケーラーを保存
joblib.dump(scaler, 'scaler2.pkl')
print("Scaler saved to 'scaler2.pkl'")

# グリッドサーチ用パラメータ
param_grid_gb = {
    'n_estimators': [100, 200, 500],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# 勾配ブースティング回帰モデル
gb_model = GradientBoostingRegressor(random_state=42)

# グリッドサーチ
grid_search_gb = GridSearchCV(
    estimator=gb_model,
    param_grid=param_grid_gb,
    scoring='r2',
    cv=5,
    n_jobs=-1
)

# グリッドサーチの実行
grid_search_gb.fit(X_train, y_train_log)

# 最適なモデル
best_gb_model = grid_search_gb.best_estimator_

# モデルの保存
joblib.dump(best_gb_model, 'gradient_boosting_model2.pkl')
print("Gradient Boosting Model saved as 'gradient_boosting_model2.pkl'")

# 学習データで評価
y_train_pred_log = best_gb_model.predict(X_train)
r2_train_gb = r2_score(y_train_log, y_train_pred_log)
print(f"Gradient Boosting R² Score on Training Data: {r2_train_gb:.2f}")

# テストデータで評価
y_test_pred_log = best_gb_model.predict(X_test)
r2_test_gb = r2_score(y_test_log, y_test_pred_log)
print(f"Gradient Boosting R² Score on Test Data: {r2_test_gb:.2f}")

# モデルの保存
joblib.dump(best_gb_model, 'gradient_boosting_model.pkl')
print("Gradient Boosting Model saved as 'gradient_boosting_model.pkl'")

# グラフ作成
data_pairs = list(zip(y_test_log, y_test_pred_log))
frequency = Counter(data_pairs)
sizes = [frequency[pair] * 20 for pair in data_pairs]

plt.figure(figsize=(8, 6))
for (x, y), size in zip(data_pairs, sizes):
    plt.scatter(x, y, s=size, alpha=0.5, color='blue')
plt.xlabel("Actual Citation Score (y_test)", fontsize=12)
plt.ylabel("Predicted Citation Score (y_pred)", fontsize=12)
plt.plot([min(y_test_log), max(y_test_log)],
         [min(y_test_log), max(y_test_log)],
         color="red", linestyle="--", label="Ideal Line (y=x)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("actual_vs_predicted_pruned_new8.png", dpi=300, format='png')
plt.close()
print(f"グラフが 'actual_vs_predicted_pruned_new8.png' に保存されました。")

# 特徴量名を設定
feature_names = [
    "cited_after_one_month", "cite_20_over_paper", "cite_50_over_paper",
    "cite_100_over_paper", "max_citation", "ave_citation", "cite_recent_paper",
    "topics_count", "h_index", "author_cited_by_ratio",
    "institution_cited_by_ratio", "topic_rank", "authority", "venue_centrality"
]

# 特徴量の重要度を取得
feature_importances = best_gb_model.feature_importances_

# 特徴量とその重要度を表示
for name, importance in sorted(zip(feature_names, feature_importances), key=lambda x: x[1], reverse=True):
    print(f"Feature: {name}, Importance: {importance:.4f}")

# 特徴量の重要度をグラフ化
plt.figure(figsize=(10, 6))
plt.barh(feature_names, feature_importances, color='skyblue')
plt.xlabel("Feature Importance", fontsize=12)
plt.ylabel("Features", fontsize=12)
plt.gca().invert_yaxis()  # 特徴量を上位から下位の順に並べる
plt.tight_layout()
plt.savefig("feature_importance_pruned_new8.png", dpi=300, format='png')
plt.close()
print(f"特徴量の寄与率グラフが 'feature_importance_pruned_new8.png' に保存されました。")

