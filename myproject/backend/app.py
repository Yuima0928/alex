from flask import Flask, request, jsonify
from flask_cors import CORS
import sqlite3
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.tree import DecisionTreeRegressor
import joblib
import json
from collections import defaultdict
import pickle
from scipy.sparse import lil_matrix, csr_matrix
from datetime import datetime
from datetime import date, timedelta
import multiprocessing
import numpy as np
from functools import partial
from transformers import AutoTokenizer

# 計算対象のTopNリスト
topN = 10
old_year = 2024
# グローバル変数の初期化
paper_number = 1000
index = faiss.read_index("../preprocessing/faiss_index_with_all_faster_new2.index")
# ファイルからデータを読み込み
with open('../preprocessing/topic_ranks_2024.pkl', 'rb') as f:
    ranks = pickle.load(f)

# 保存されたスケーラーをロード
scaler = joblib.load('../preprocessing/scaler2.pkl')
print("Scaler loaded successfully")

# 読み込んだデータを利用
domain_rank = ranks['domain_rank']
field_rank = ranks['field_rank']
subfield_rank = ranks['subfield_rank']

print("Ranks loaded successfully!")

# モデルを読み込み
decision_tree_model = joblib.load('../preprocessing/gradient_boosting_model2.pkl')
print("Model loaded successfully")

model = SentenceTransformer('all-MiniLM-L6-v2', device="cpu") # cudaにしたら、時間はかかる。でも"cpu"だとkilledされることがある。

# # 並列処理前にトークナイザを初期化
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")  # 適切なモデル名を指定


app = Flask(__name__)
CORS(app)  # CORSを有効化

@app.route('/api/message', methods=['POST'])
def get_message():
    data = request.get_json()
    user_message = data['message']
    conn = sqlite3.connect('../preprocessing/paper2.db')
    cursor = conn.cursor()

    author_rank_dic = defaultdict(int) 
    institution_rank_dic = defaultdict(int) 

    top_N_papers, top_latest_papers, pagerank_vector = pick_up_top_papers_id(user_message, cursor, author_rank_dic, institution_rank_dic)
    print(len(top_latest_papers))
    X = pick_up_x_data_parallel(top_latest_papers, author_rank_dic, institution_rank_dic, 16) #2024のもののみ追加

    # X_test のデータクリーンアップ
    X = clean_data(X)
    X = scaler.transform(X)

    y = decision_tree_model.predict(X)

    # インデックス付きでソート
    sorted_indices = sorted(range(len(pagerank_vector)), key=lambda i: pagerank_vector[i], reverse=True)
    sorted_array = [pagerank_vector[i] for i in sorted_indices]
    top_10_papers_id = [top_N_papers[i] for i in sorted_indices[:10]]

    # インデックス付きでソート
    sorted_indices = sorted(range(len(y)), key=lambda i: y[i], reverse=True)
    sorted_array = [y[i] for i in sorted_indices]
    print(sorted_array[:10])
    top_latest_paper_id = top_latest_papers[sorted_indices[0]]

    # レスポンスをマークダウン形式で作成
    response_data = []
    for top_paper_id in top_10_papers_id:
        cursor.execute('SELECT title, abstract FROM PaperVectors WHERE paper_id = ?', (top_paper_id,))
        result = cursor.fetchone()
        if result:
            title, abstract = result
            paper_md = f"**Paper ID**: [{top_paper_id}]({top_paper_id})\n\n" \
                    f"**Title**: {title}\n\n" \
                    f"**Abstract**: {abstract}\n\n---\n"
            response_data.append(paper_md)

    # 最新のペーパーの出力
    cursor.execute('SELECT title, abstract FROM PaperVectors WHERE paper_id = ?', (top_latest_paper_id,))
    result = cursor.fetchone()
    if result:
        title, abstract = result
        latest_paper_md = f"## Latest Paper\n\n" \
                        f"**Paper ID**: [{top_latest_paper_id}]({top_latest_paper_id})\n\n" \
                        f"**Title**: {title}\n\n" \
                        f"**Abstract**: {abstract}\n\n---\n"
        response_data.append(latest_paper_md)
    
    # レスポンスデータを連結して送信
    return jsonify(message="\n".join(response_data))

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

    # PageRank計算
    for _ in range(max_iter):
        new_pagerank_vector = damping_factor * g_matrix.T @ pagerank_vector + (1 - damping_factor) / paper_number
        if np.linalg.norm(new_pagerank_vector - pagerank_vector) < epsilon:
            break
        pagerank_vector = new_pagerank_vector

    return pagerank_vector

def pick_up_top_papers_id(input, cursor, author_rank_dic, institution_rank_dic):
    embedding = model.encode([input], batch_size=1) 
    # FAISS検索
    index.nprobe = 10
    distances, indices = index.search(embedding.reshape(1, -1), paper_number) # 取り除く必要ないから + 1なしで

    # 上位の類似論文を取得
    top_N_papers = []
    author_ids = []
    institution_ids = []

    top_latest_papers = []
    query = "SELECT paper_id, created_date, author_id, institution_id FROM PaperVectors WHERE id = ?;"
    for i, idx in enumerate(indices[0]):
        if int(idx) == -1:
            continue
        cursor.execute(query, (int(idx),))
        result = cursor.fetchone()
        if result:
            paper_id, created_date, author_id, institution_id = result
            top_N_papers.append(paper_id)
            author_ids.append(author_id)
            institution_ids.append(institution_id)

            created_date = datetime.strptime(created_date, "%Y-%m-%d")  # 文字列を datetime 型に変換
            if created_date.year >= 2023:
                top_latest_papers.append(paper_id)

    # PageRank計算準備
    paper_dict = {paper_id: idx for idx, paper_id in enumerate(top_N_papers)}
    a_vector = [[] for _ in range(len(top_N_papers))]
    sum_edge_per_paper = [0 for _ in range(len(top_N_papers))]

    for paper_id in top_N_papers:
        cursor.execute('SELECT referenced_paper_id FROM ReferencedWorks WHERE paper_id = ?', (paper_id,))
        reference_works = [r[0] for r in cursor.fetchall()]
        for ref_id in reference_works:
            if ref_id in paper_dict:
                if ref_id == paper_id:
                    continue
                sum_edge_per_paper[paper_dict[paper_id]] += 1
                a_vector[paper_dict[paper_id]].append([paper_dict[ref_id], 1])

    # PageRank計算（疎行列版）
    pagerank_vector = calculate_pagerank_sparse(a_vector, len(top_N_papers))

    for i in range(paper_number):
        author_rank_dic[author_ids[i]] += pagerank_vector[i]
        institution_rank_dic[institution_ids[i]] += pagerank_vector[i]

    return top_N_papers, top_latest_papers, pagerank_vector

def pick_up_x_data(paper_ids, cursor, author_rank_dic, institution_rank_dic):
    x_data = []

    i = 0
    for paper_id in paper_ids:
        cursor.execute("SELECT author_id, institution_id, topics_count, created_date FROM PaperVectors WHERE paper_id = ?", (paper_id,))
        author_id, institution_id, topics_count, created_date = cursor.fetchone()
        if not topics_count:
            topics_count = 0
        
        # created_date を datetime オブジェクトに変換
        if isinstance(created_date, str):
            created_date = datetime.strptime(created_date, '%Y-%m-%d')
        one_month_after = created_date + timedelta(days=30)

        cursor.execute("SELECT COUNT(*) FROM ReferencedWorks WHERE referenced_paper_id = ? AND created_date <= ?", (paper_id, created_date))
        cited_after_one_month = cursor.fetchone()[0]

        # print(i)
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
        if (author_id):
            cursor.execute("SELECT h_index, cited_by_count, works_count FROM Authors WHERE author_id = ?", (author_id,))
            author_data = cursor.fetchone()
            if author_data:
                h_index, author_total_cited_by_count, author_total_works_count = author_data
            else:
                print("yyy")
                h_index, author_total_cited_by_count, author_total_works_count = 0, 0, 0

        
        institution_total_cited_by_count, institution_total_works_count = 0,0
        if (institution_id):
            cursor.execute("SELECT cited_by_count, works_count FROM Institutions WHERE institution_id = ?", (institution_id,))
            institution_data = cursor.fetchone()
            if institution_data:
                institution_total_cited_by_count, institution_total_works_count = institution_data
            else:
                institution_total_cited_by_count, institution_total_works_count = 0, 0
        
        author_recent_cited_by_count = 0
        author_recent_works_count = 0
        if (author_id):
            cursor.execute("SELECT cited_by_count, works_count FROM AuthorsYearCount WHERE author_id = ? AND year < ?", (author_id, old_year))
            rows =  cursor.fetchall()
            for row in rows:
                author_recent_cited_by_count += row[0]
                author_recent_works_count += row[1]

        institution_recent_cited_by_count = 0
        institution_recent_works_count = 0
        if (institution_id):
            cursor.execute("SELECT cited_by_count, works_count FROM InstitutionsYearCount WHERE institution_id = ? AND year < ?", (institution_id, old_year))
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

        authority = author_rank_dic[author_id]
        venue_centrality = institution_rank_dic[institution_id]
        x_data[i].extend([topic_rank, authority, venue_centrality])
    return x_data

# データベース接続関数（プロセスごとに新しい接続が必要）
def connect_to_db():
    db_path = '/home/mdxuser/kennkyuu/predict/paper2.db'
    return sqlite3.connect(db_path)

# 分割されたチャンクでX_dataを作成する関数
def process_chunk(chunk_args):
    paper_ids_chunk, authority_dic, venue_centrality_dic = chunk_args
    connection = connect_to_db()
    cursor = connection.cursor()
    x_data_chunk = pick_up_x_data(paper_ids_chunk, cursor, authority_dic, venue_centrality_dic)
    connection.close()
    return x_data_chunk

# 並列処理用のpick_up_x_data関数（cursorを引数として受け取るよう変更）
def pick_up_x_data_parallel(paper_ids, authority_dic, venue_centrality_dic, num_processes=16):
    # データをチャンクに分割
    chunks = np.array_split(list(zip(paper_ids)), num_processes)
    chunk_args = [
        (
            [chunk[i][0] for i in range(len(chunk))],  # paper_ids
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

# データクリーンアップ関数
def clean_data(data):
    cleaned_data = []
    for row in data:
        cleaned_row = []
        for value in row:
            # 空文字列を 0 に置き換え
            if value == '' or value is None:
                cleaned_row.append(0)
            else:
                cleaned_row.append(float(value))  # 明示的に数値型に変換
        cleaned_data.append(cleaned_row)
    return cleaned_data

if __name__ == '__main__':
    app.run(debug=True)