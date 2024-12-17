import sqlite3
import numpy as np
import faiss
from multiprocessing import Pool
import time
import matplotlib.pyplot as plt

# 計算対象のTopNリスト
topN_list = [10, 15, 20, 25, 30]

# グローバル変数の初期化
paper_number = 1000
index = faiss.read_index("./faiss_index_with_all_faster.index")
print("FAISS index loaded successfully.")

# Precision/Recall計算用関数
def calculate_precision_recall(args):
    attention_paper_id, attention_reference_works = args

    conn = sqlite3.connect('./paper_with_topic_indexed_new.db')
    cursor = conn.cursor()

    # 対象論文のベクトルを取得
    cursor.execute("SELECT id, vector FROM PaperVectors WHERE paper_id = ?;", (attention_paper_id,))
    row = cursor.fetchone()
    if not row:
        print(f"No vector found for paper_id: {attention_paper_id}")
        return None

    attention_id, attention_vector = row
    attention_vector = np.frombuffer(attention_vector, dtype=np.float32)

    # FAISS検索
    index.nprobe = 10
    distances, indices = index.search(attention_vector.reshape(1, -1), paper_number + 1)

    # 上位の類似論文を取得
    top_1000_papers = []
    query = "SELECT paper_id FROM PaperVectors WHERE id = ?;"
    for i, idx in enumerate(indices[0]):
        if int(idx) == -1:
            continue
        cursor.execute(query, (int(idx),))
        paper_id = cursor.fetchone()
        if paper_id and paper_id[0] != attention_paper_id:
            top_1000_papers.append((paper_id[0], distances[0][i]))

    if len(top_1000_papers) == 0:
        print(f"No similar papers found for paper_id: {attention_paper_id}")
        return None

    # PageRank計算準備
    top_1000_dict = {paper_id: idx for idx, (paper_id, _) in enumerate(top_1000_papers)}
    g_matrix = [[0 for _ in range(paper_number)] for _ in range(paper_number)]
    a_vector = [[] for _ in range(paper_number)]
    sum_edge_per_paper = [0 for _ in range(paper_number)]

    for paper_id, _ in top_1000_papers:
        for ref_id in attention_reference_works:
            if ref_id in top_1000_dict:
                if ref_id == paper_id:
                    continue
                sum_edge_per_paper[top_1000_dict[paper_id]] += 1
                a_vector[top_1000_dict[paper_id]].append([top_1000_dict[ref_id], 1])

    for i in range(paper_number):
        if sum_edge_per_paper[i] > 0:
            for sub_a_vector in a_vector[i]:
                g_matrix[i][sub_a_vector[0]] = sub_a_vector[1] / sum_edge_per_paper[i]

    # PageRank計算
    pagerank_vector = np.ones(paper_number) / paper_number
    damping_factor = 0.85
    epsilon = 1e-6

    for _ in range(100):
        new_pagerank_vector = np.zeros(paper_number)
        for i in range(paper_number):
            for j in range(paper_number):
                new_pagerank_vector[i] += damping_factor * g_matrix[j][i] * pagerank_vector[j]
            new_pagerank_vector[i] += (1 - damping_factor) / paper_number
        if np.linalg.norm(new_pagerank_vector - pagerank_vector) < epsilon:
            break
        pagerank_vector = new_pagerank_vector

    # Precision/Recall計算
    precision_results = []
    recall_results = []

    # PageRankとRelevanceスコアの正規化
    pagerank_sum = np.sum(pagerank_vector)
    normalized_pagerank_vector = pagerank_vector / pagerank_sum
    relevance_scores = np.array([1.0 / top_1000_papers[i][1] for i in range(len(top_1000_papers))])
    relevance_sum = np.sum(relevance_scores)
    normalized_relevance_scores = relevance_scores / relevance_sum

    # Finalスコア計算
    c_pagerank, c_relevance = 1, 0
    final_score = [
        c_pagerank * normalized_pagerank_vector[i] + c_relevance * normalized_relevance_scores[i]
        for i in range(len(top_1000_papers))
    ]

    for topN in topN_list:
        if topN > len(top_1000_papers):
            continue

        top_indices = np.argsort(final_score)[-topN:][::-1]
        top_papers_id = [top_1000_papers[i][0] for i in top_indices]

        common = sum(1 for ref_id in attention_reference_works if ref_id in top_papers_id)
        precision = common / topN if topN > 0 else 0
        recall = common / len(attention_reference_works) if attention_reference_works else 0

        precision_results.append(precision)
        recall_results.append(recall)

    conn.close()
    return precision_results, recall_results

# 並列計算関数
def parallel_process():
    conn = sqlite3.connect('./paper_with_topic_indexed_new.db')
    cursor = conn.cursor()

    flag, count = 0, 0
    tasks = []
    while flag < 50:
        cursor.execute('SELECT paper_id FROM PaperVectors LIMIT 1 OFFSET ?', (count + 2000000, ))
        row = cursor.fetchone()
        if not row:
            break

        attention_paper_id = row[0]
        cursor.execute('SELECT referenced_paper_id FROM ReferencedWorks WHERE paper_id = ?', (attention_paper_id,))
        referenced_papers = [r[0] for r in cursor.fetchall()]

        if referenced_papers:
            tasks.append((attention_paper_id, referenced_papers))
            flag += 1
        count += 1

    conn.close()

    with Pool(processes=16) as pool:  # 16プロセスで並列化
        results = pool.map(calculate_precision_recall, tasks)

    return results

if __name__ == "__main__":
    start_time = time.time()
    results = parallel_process()

    # 結果を集計
    all_precisions = [result[0] for result in results if result]
    all_recalls = [result[1] for result in results if result]

    mean_precisions = np.mean(all_precisions, axis=0)
    mean_recalls = np.mean(all_recalls, axis=0)
    print(mean_precisions)
    print(mean_recalls)
    # 結果をプロット
    plt.figure(figsize=(12, 6))
    plt.plot(topN_list, mean_precisions, marker='o', label='Precision')
    plt.plot(topN_list, mean_recalls, marker='o', label='Recall')
    plt.xlabel('Top N')
    plt.ylabel('Scores')
    plt.title('Precision and Recall')
    plt.legend()
    plt.grid()
    plt.savefig("precision_recall_parallel_all_attention_all.png")
    plt.show()

    end_time = time.time()
    print(f"Execution time: {end_time - start_time:.2f} seconds")
