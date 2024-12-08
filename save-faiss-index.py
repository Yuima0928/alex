import sqlite3
import numpy as np
import faiss
from multiprocessing import Pool
import time
import matplotlib.pyplot as plt

# FAISSインデックス設定
vector_dim = 384  # ベクトルの次元数
nlist = 5000  # センチロイド数
batch_size = 1000000  # バッチ処理サイズ
processes = 18  # 並列プロセス数
# SQLiteデータベースに接続
conn = sqlite3.connect('./paper_with_topic_indexed.db')
cursor = conn.cursor()

# 検索トピック
topic_value = "Epidemiology"

# # SQLクエリを実行
# query = "SELECT COUNT(*) FROM PaperVectors WHERE field = ?;"
# cursor.execute(query, (topic_value,))

# num_records = cursor.fetchone()[0]
# print(f"Total number of saved papers: {num_records}")
# conn.close()

# SQLクエリを実行
query = "SELECT id, vector FROM PaperVectors WHERE subfield = ?;"
cursor.execute(query, (topic_value,))

nlist = 1000  # データ量に応じて調整
m = vector_dim // 4  # ベクトル次元数の1/4を使用

# ID対応のFAISSインデックスを初期化
quantizer = faiss.IndexFlatL2(vector_dim)
index = faiss.IndexIDMap(faiss.IndexIVFPQ(quantizer, vector_dim, nlist, m, 8))

# トレーニング用のデータ収集
train_vectors = []
rows = cursor.fetchmany(100000)  # トレーニング用のサンプルサイズを指定
for row in rows:
    try:
        vector_blob = row[1]
        paper_vector = np.frombuffer(vector_blob, dtype=np.float32).copy()
        
        # ベクトル次元数を確認
        if len(paper_vector) != vector_dim:
            print(f"Skipping vector with unexpected length: {len(paper_vector)}")
            continue

        train_vectors.append(paper_vector)
    except Exception as e:
        print(f"Error processing training row: {row}, error: {e}")

# トレーニング
if train_vectors:
    train_vectors_np = np.array(train_vectors, dtype=np.float32)
    index.train(train_vectors_np)
    print(f"FAISS index trained with {len(train_vectors)} vectors.")
else:
    raise ValueError("No training data found. Cannot train FAISS index.")

# バッチ処理用関数
def process_batch(rows):
    vectors = []
    ids = []
    for row in rows:
        try:
            paper_id = int(row[0])  # IDは整数である必要があります
            vector_blob = row[1]
            paper_vector = np.frombuffer(vector_blob, dtype=np.float32).copy()

            vectors.append(paper_vector)
            ids.append(paper_id)
        except Exception as e:
            print(f"Error processing row: {row}, error: {e}")
    return vectors, ids

# データベースをバッチ処理で読み込み
start_time = time.time()
total_vectors_added = 0

while True:
    rows = cursor.fetchmany(batch_size)
    if not rows:
        break

    # バッチ処理を並列で実行
    with Pool(processes=processes) as pool:
        batch_size_per_process = batch_size // processes
        split_rows = [rows[i:i + batch_size_per_process] for i in range(0, len(rows), batch_size_per_process)]
        results = pool.map(process_batch, split_rows)

    # 結果をFAISSインデックスに追加
    for vectors, ids in results:
        if vectors and ids:
            index.add_with_ids(np.array(vectors, dtype=np.float32), np.array(ids, dtype=np.int64))
            total_vectors_added += len(vectors)
            elapsed_time = time.time() - start_time
            print(f"Added {len(vectors)} vectors. Total added: {total_vectors_added}. Elapsed time: {elapsed_time:.2f} seconds.")

# SQLite接続を閉じる
conn.close()

# FAISSインデックスの情報を出力
print(f"Total vectors in FAISS index: {index.ntotal}")

# FAISSインデックスを保存する場合
faiss.write_index(index, "faiss_index_with_Epidemiology.index")
print("FAISS index saved as 'faiss_index_with_Epidemiology.index'")
