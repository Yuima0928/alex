import sqlite3
import numpy as np
import faiss
from multiprocessing import Pool
import time
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import norm

# 計算対象のTopNリスト
topN_list = [10, 15, 20, 25, 30]

# グローバル変数の初期化
# paper_number = 500
index = faiss.read_index("./predict/faiss_index_with_all_faster_new2.index")
print("FAISS index loaded successfully.")

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

# Precision/Recall計算用関数
def calculate_precision_recall(args):
    attention_paper_id, attention_reference_works , paper_number = args

    conn = sqlite3.connect('./predict/paper2.db')
    cursor = conn.cursor()

    # 対象論文のベクトルを取得
    cursor.execute("SELECT vector FROM PaperVectors WHERE paper_id = ?;", (attention_paper_id,))
    attention_vector = np.frombuffer(cursor.fetchone()[0], dtype=np.float32)

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
    a_vector = [[] for _ in range(paper_number)]
    sum_edge_per_paper = [0 for _ in range(paper_number)]

    for paper_id, _ in top_1000_papers:
        cursor.execute('SELECT referenced_paper_id FROM ReferencedWorks WHERE paper_id = ?', (paper_id,))
        reference_works = [r[0] for r in cursor.fetchall()]
        for ref_id in reference_works:
            if ref_id in top_1000_dict:
                if ref_id == paper_id:
                    continue
                sum_edge_per_paper[top_1000_dict[paper_id]] += 1
                a_vector[top_1000_dict[paper_id]].append([top_1000_dict[ref_id], 1])

    # PageRank計算（疎行列版）
    pagerank_vector = calculate_pagerank_sparse(a_vector, paper_number)

    # Precision/Recall計算
    precision_results = []
    recall_results = []
    f1_score_results = []

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
        f1_score = (2 * precision * recall) / (precision + recall) if common else 0

        precision_results.append(precision)
        recall_results.append(recall)
        f1_score_results.append(f1_score)

    conn.close()
    return precision_results, recall_results, f1_score_results

# 並列計算関数
def parallel_process(paper_number):
    conn = sqlite3.connect('./predict/paper2.db')
    cursor = conn.cursor()

    flag, count = 0, 0
    tasks = []
    while flag < 50:
        cursor.execute('SELECT paper_id FROM PaperVectors LIMIT 1 OFFSET ?', (count + 2000000,))
        row = cursor.fetchone()
        if not row:
            break

        attention_paper_id = row[0]
        cursor.execute('SELECT referenced_paper_id FROM ReferencedWorks WHERE paper_id = ?', (attention_paper_id,))
        attention_reference_works = [r[0] for r in cursor.fetchall()]

        if attention_reference_works:
            tasks.append((attention_paper_id, attention_reference_works, paper_number))
            flag += 1
        count += 1

    conn.close()

    with Pool(processes=16) as pool:  # 16プロセスで並列化
        results = pool.map(calculate_precision_recall, tasks)

    return results

if __name__ == "__main__":

    paper_numbers = [50, 100, 500, 1000, 5000]
    times = []
    mean_precisions_list = []
    mean_recalls_list = []
    mean_f1_scores_list = []
    for paper_number in paper_numbers:
        start_time = time.time()

        results = parallel_process(paper_number)

        # 結果を集計
        all_precisions = [result[0] for result in results if result]
        all_recalls = [result[1] for result in results if result]
        all_f1_scores = [result[2] for result in results if result]

        mean_precisions = np.mean(all_precisions, axis=0)
        mean_recalls = np.mean(all_recalls, axis=0)
        mean_f1_scores = np.mean(all_f1_scores, axis = 0)

        end_time = time.time()
        times.append(end_time - start_time)

        print(f"Execution time: {end_time - start_time:.2f} seconds")
    
        print(mean_precisions)
        print(mean_recalls)
        print(mean_f1_scores)
        mean_precisions_list.append(mean_precisions)
        mean_recalls_list.append(mean_recalls)
        mean_f1_scores_list.append(mean_f1_scores)
    print(times)
    # 結果をプロット
   # Plotting the graphs for Precision, Recall, and F1-Score
    # Plotting the graphs for Precision, Recall, and F1-Score
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))

    # Plot Precision
    for i, paper_number in enumerate(paper_numbers):
        axes[0].plot(topN_list, mean_precisions_list[i], marker='o', label=f'Paper Number {paper_number}')
    axes[0].set_xlabel('Top N')
    axes[0].set_ylabel('Precision')
    axes[0].legend()
    axes[0].grid()
    axes[0].set_title('Precision vs Top N', loc='center', y=-0.2)

    # Plot Recall
    for i, paper_number in enumerate(paper_numbers):
        axes[1].plot(topN_list, mean_recalls_list[i], marker='o', label=f'Paper Number {paper_number}')
    axes[1].set_xlabel('Top N')
    axes[1].set_ylabel('Recall')
    axes[1].legend()
    axes[1].grid()
    axes[1].set_title('Recall vs Top N', loc='center', y=-0.2)

    # Plot F1-Score
    for i, paper_number in enumerate(paper_numbers):
        axes[2].plot(topN_list, mean_f1_scores_list[i], marker='o', label=f'Paper Number {paper_number}')
    axes[2].set_xlabel('Top N')
    axes[2].set_ylabel('F1-Score')
    axes[2].legend()
    axes[2].grid()
    axes[2].set_title('F1-Score vs Top N', loc='center', y=-0.2)

    # Add overall title
    # fig.suptitle('Precision, Recall, and F1-Score Analysis', y=0.02)

    # Save the figure as a single PNG file
    output_filename = 'precision_recall_f1_scores2.png'
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(output_filename)
    plt.show()

    print(f"Graph saved as {output_filename}")
