import sqlite3
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import time
import matplotlib.pyplot as plt

# 計算対象のTopNリスト
topN = 10

# グローバル変数の初期化
paper_number = 1000
start_time = time.time()
index = faiss.read_index("./faiss_index_with_all_faster.index")
end_time = time.time()
print(f"Execution time: {end_time - start_time:.2f} seconds")
print("FAISS index loaded successfully.")

def pick_up_top_papers_id(input):
    start_time = time.time()
    embedding = model.encode([input], batch_size=1) 
    end_time = time.time()
    print(f"embeddingExecution time: {end_time - start_time:.2f} seconds")

    start_time = time.time()
    # FAISS検索
    index.nprobe = 10
    distances, indices = index.search(embedding.reshape(1, -1), paper_number) # 取り除く必要ないから + 1なしで

    # 上位の類似論文を取得
    top_1000_papers = []
    query = "SELECT paper_id FROM PaperVectors WHERE id = ?;"
    for i, idx in enumerate(indices[0]):
        if int(idx) == -1:
            continue
        cursor.execute(query, (int(idx),))
        paper_id = cursor.fetchone()
        if paper_id:
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
        cursor.execute('SELECT referenced_paper_id FROM ReferencedWorks WHERE paper_id = ?', (paper_id,))
        reference_works = [r[0] for r in cursor.fetchall()]
        for ref_id in reference_works:
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

    end_time = time.time()
    print(f"pagerank embeddingExecution time: {end_time - start_time:.2f} seconds")

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

    top_indices = np.argsort(final_score)[-topN:][::-1]
    top_papers_id = [top_1000_papers[i][0] for i in top_indices]

    return top_papers_id

if __name__ == "__main__":
    conn = sqlite3.connect('./paper_with_topic_indexed_new.db')
    cursor = conn.cursor()
    model = SentenceTransformer('all-MiniLM-L6-v2', device="cpu") # cudaにしたら、時間はかかる。でも"cpu"だとkilledされることがある。
    # start_time = time.time()
    for i in range(3):

        print("input your research:")
        text = input()
        start_time = time.time()
        top_papers_id = pick_up_top_papers_id(text)
        end_time = time.time()
        print(f"{i}回目 Execution time: {end_time - start_time:.2f} seconds")

        with open("output.txt", "w", encoding="utf-8") as f:
            for top_paper_id in top_papers_id:
                cursor.execute('SELECT title, abstract FROM PaperVectors WHERE paper_id is ?', (top_paper_id,))
                title, abstract = cursor.fetchone()
                f.write(f"paper_id: {top_paper_id}\n")
                f.write(f"  title: {title}\n")
                f.write(f"  abstract: {abstract}\n")

    conn.close()