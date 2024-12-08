import sqlite3
import numpy as np
import faiss
from multiprocessing import Pool
import time
import matplotlib.pyplot as plt

topN_list = [10, 15, 20, 25, 30]

conn = sqlite3.connect('./paper_with_topic_indexed.db')
cursor = conn.cursor()

# 検索トピック
topic_value = "Epidemiology"

paper_number = 1000

# PageRankとPrecision/Recall計算用関数
def calculate_precision_recall(attention_paper_id):

    # # SQLクエリを実行 (n番目を取得)
    # query = "SELECT id, paper_id, vector FROM PaperVectors WHERE subfield = ? LIMIT 1 OFFSET ?;"
    # cursor.execute(query, (topic_value, attention_paper_index))

    # # 結果を取得
    # row = cursor.fetchone()
    
    # if row:
    #     attention_id, attention_paper_id, attention_vector = row
    #     print(f"Retrieved record {attention_paper_index + 1}: ID={attention_paper_id}")
    # else:
    #     print(f"No matching record found at position {attention_paper_index + 1}.")

    cursor.execute("SELECT id, vector FROM PaperVectors WHERE paper_id = ?;", (attention_paper_id,))
    row = cursor.fetchone()
    attention_id, attention_vector = row
    attention_vector = np.frombuffer(attention_vector, dtype=np.float32)
    print(attention_vector.shape)
    # FAISS検索
    index.nprobe = 1000
    distances, indices = index.search(np.frombuffer(attention_vector, dtype=np.float32).copy().reshape(1, -1), paper_number + 1)

    # 上位の類似論文を取得
    top_1000_papers = []
    query = "SELECT paper_id FROM PaperVectors WHERE id = ?;"
    for i, idx in enumerate(indices[0]):
        cursor.execute(query, (int(idx), ))
        paper_id = cursor.fetchone()[0]
        print(paper_id)
        if paper_id is None:
            print(f"No matching paper_id for id: {idx}")
            continue
        top_1000_papers.append((paper_id, distances[0][i]))
    if len(top_1000_papers) == 0:
        print(f"No similar papers found for paper_id: {attention_paper_id}")
        return None, None

    # 3番目のコードの上位1000件のpaper_idを保存
    top_1000_ids_fourth_code = [paper_id for paper_id, _ in top_1000_papers]  # 2番目のコード

    # .txtファイルに保存
    with open('top_1000_ids_fourth_code.txt', 'w') as f:
        for paper_id in top_1000_ids_fourth_code:
            f.write(f"{paper_id}\n")

    print("3番目のコードの上位1000件のpaper_idをtop_1000_ids_third_code.txtに保存しました。")

    # PageRankのスパース行列構築
    top_1000_dict = {paper_id: idx for idx, (paper_id, _) in enumerate(top_1000_papers)}
    #print(top_1000_dict)
    g_matrix = np.zeros((paper_number, paper_number), dtype=np.float32)
    sum_edge_per_paper = np.zeros(paper_number, dtype=np.int32)


    # 参照論文を取得
    query = "SELECT referenced_paper_id FROM ReferencedWorks WHERE paper_id = ?;"
    cursor.execute(query, (attention_paper_id,))
    attention_reference_works = [row[0] for row in cursor.fetchall()]
    print("kkk")
    print(attention_paper_id)
    print(attention_reference_works)
    if not attention_reference_works:
        print(f"No referenced works for paper_id: {attention_paper_id}")
        return None, None
    for paper_id, _ in top_1000_papers:
        #print(paper_id[0])
        cursor.execute(query, (paper_id,))
        referenced_works = [row[0] for row in cursor.fetchall()]
        #print(referenced_works)
        for ref_id in referenced_works:
            if ref_id in top_1000_dict and paper_id[0] in top_1000_dict:
                print("kk")
                print(ref_id)
                ref_idx = top_1000_dict[ref_id]
                paper_idx = top_1000_dict[paper_id]
                g_matrix[ref_idx][paper_idx] += 1
                sum_edge_per_paper[paper_idx] += 1

    # 行の正規化
    for i in range(paper_number):
        if sum_edge_per_paper[i] > 0:
            g_matrix[:, i] /= sum_edge_per_paper[i]

    # PageRank計算
    pagerank_vector = np.ones(paper_number) / paper_number
    damping_factor = 0.85
    epsilon = 1e-6

    for _ in range(100):
        new_pagerank_vector = damping_factor * np.dot(g_matrix, pagerank_vector) + (1 - damping_factor) / paper_number
        if np.linalg.norm(new_pagerank_vector - pagerank_vector) < epsilon:
            break
        pagerank_vector = new_pagerank_vector

    # Precision/Recall計算
    precision_results = []
    recall_results = []

    for topN in topN_list:
        if topN > len(top_1000_papers):
            print(f"TopN ({topN}) exceeds number of similar papers ({len(top_1000_papers)}).")
            continue

        top_papers_id = [top_1000_papers[i][0] for i in range(topN)]
        # print(top_papers_id)
        # print("kkk")
        # print(attention_reference_works)
        common = 0
        for i in range(len(attention_reference_works)):
            if attention_reference_works[i] in top_papers_id:
                common += 1

                print(1)
    
        precision = common / topN if topN > 0 else 0
        recall = common / len(attention_reference_works) if attention_reference_works else 0

        precision_results.append(precision)
        recall_results.append(recall)

    return precision_results, recall_results

# インデックスをロード
index = faiss.read_index("faiss_index_with_Epidemiology.index")
print("FAISS index loaded successfully.")

# 基準論文のインデックスを指定（例: 最初の論文を使用）
attention_paper_id = "https://openalex.org/W3002836194"

output_file = "./results/precision_recall.png"

# PrecisionとRecallを計算
precision, recall = calculate_precision_recall(attention_paper_id)

print("Precision:", precision)
print("Recall:", recall)


if precision and recall:
    # グラフを描画
    plt.figure(figsize=(12, 6))

    # Precisionのプロット
    plt.subplot(1, 2, 1)
    plt.plot(topN_list, precision, marker='o', label='Precision')
    plt.title('Precision for a Single Paper')
    plt.xlabel('Top N')
    plt.ylabel('Precision')
    plt.grid(True)
    plt.legend()

    # Recallのプロット
    plt.subplot(1, 2, 2)
    plt.plot(topN_list, recall, marker='o', label='Recall')
    plt.title('Recall for a Single Paper')
    plt.xlabel('Top N')
    plt.ylabel('Recall')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    # PNGファイルに保存
    output_file = "precision_recall.png"
    plt.savefig(output_file, format='png', dpi=300)
    print(f"Graph saved as {output_file}")

    # 表示もする場合
    plt.show()
else:
    print(f"No Precision/Recall calculated for paper index {attention_paper_index}.")
