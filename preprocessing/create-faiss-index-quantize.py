import sqlite3
import numpy as np
import faiss
from multiprocessing import Pool
import time

# ベクトルの次元数
vector_dim = 384
batch_size = 100000 # バッチ処理サイズ
processes = 16  # 並列プロセス数

# SQLiteデータベースへの接続
conn = sqlite3.connect('./paper2.db')
cursor = conn.cursor()

# SQLクエリを実行してベクトルデータを取得
cursor.execute("SELECT id, vector FROM PaperVectors;")

nlist = 256 # データ量に応じて調整
m = vector_dim // 8  # ベクトル次元数の1/8を使用

# ID対応のFAISSインデックスを初期化
quantizer = faiss.IndexFlatL2(vector_dim)
index = faiss.IndexIDMap(faiss.IndexIVFPQ(quantizer, vector_dim, nlist, m, 4))

print(index.is_trained)

# トレーニング用のデータ収集
train_vectors = []
rows = cursor.fetchmany(100000)  # トレーニング用のサンプルサイズを指定 1桁medicineのときよりも増やした
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
            paper_id = int(row[0])
            vector_blob = row[1]
            paper_vector = np.frombuffer(vector_blob, dtype=np.float32).copy()
            vectors.append(paper_vector)
            ids.append(paper_id)
        except Exception as e:
            print(f"Error processing row: {row}, error: {e}")
    return vectors, ids

# データベースからベクトルをバッチ処理で読み込み、FAISSインデックスに追加
start_time = time.time()
total_vectors_added = 0
while True:
    rows = cursor.fetchmany(batch_size)
    if not rows:
        break
    with Pool(processes=processes) as pool:
        batch_size_per_process = batch_size // processes
        split_rows = [rows[i:i + batch_size_per_process] for i in range(0, len(rows), batch_size_per_process)]
        results = pool.map(process_batch, split_rows)
    for vectors, ids in results:
        if vectors and ids:
            index.add_with_ids(np.array(vectors, dtype=np.float32), np.array(ids, dtype=np.int64))
            total_vectors_added += len(vectors)
            elapsed_time = time.time() - start_time
            print(f"Added {len(vectors)} vectors. Total added: {total_vectors_added}. Elapsed time: {elapsed_time:.2f} seconds.")

# SQLiteの接続を閉じる
conn.close()

# FAISSインデックスの情報を出力
print(f"Total vectors in FAISS index: {index.ntotal}")

# インデックスを保存 newはembeddingがtitle+embeddingになっている。
faiss.write_index(index, "faiss_index_with_all_faster_nbits_per_idx4.index")
print("FAISS index saved as 'faiss_index_with_all_faster_nbits_per_idx4'")

