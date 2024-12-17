import sqlite3
import numpy as np
import faiss
from multiprocessing import Pool
import time
import matplotlib.pyplot as plt

topN_list = [10, 15, 20, 25, 30]

start_time = time.time()

conn = sqlite3.connect('./paper_with_topic_indexed_new.db')
cursor = conn.cursor()

paper_number = 1000

# PageRankとPrecision/Recall計算用関数
def calculate_precision_recall(attention_paper_id, attention_reference_works):
    # 本来はここで自分の研究をembeddingする
    cursor.execute("SELECT id, vector FROM PaperVectors WHERE paper_id = ?;", (attention_paper_id,))
    row = cursor.fetchone()
    attention_id, attention_vector = row
    attention_vector = np.frombuffer(attention_vector, dtype=np.float32)

    # FAISS検索
    index.nprobe = 10
    distances, indices = index.search(np.frombuffer(attention_vector, dtype=np.float32).copy().reshape(1, -1), paper_number + 1)

    # 上位の類似論文を取得
    top_1000_papers = []
    query = "SELECT paper_id FROM PaperVectors WHERE id = ?;"
    for i, idx in enumerate(indices[0]):
        # 距離が遠すぎると(distances)、paper_numberの分とれないで、-1になる場合がある
        if (int(idx) == -1):
            continue
        cursor.execute(query, (int(idx), ))
        paper_id = cursor.fetchone()[0]

        if paper_id is None:
            print(f"No matching paper_id for id: {idx}")
            continue
        if paper_id == attention_paper_id: # 注目論文はpagerankには含めない(それ以外でpagerankでedgeをはる)
            continue
        top_1000_papers.append((paper_id, distances[0][i]))

    if len(top_1000_papers) == 0:
        print(f"No similar papers found for paper_id: {attention_paper_id}")
        return None, None
    print(indices[0][0])
    #print("Distances:", distances[0])
    print(attention_paper_id)
    print(len(top_1000_papers))
    
    top_1000_dict = {paper_id: idx for idx, (paper_id, _) in enumerate(top_1000_papers)}

    def check_paper_in_network(paper_id):
        return paper_id in top_1000_dict

    g_matrix = [[0 for i in range(paper_number)] for j in range(paper_number)]
    a_vector = [[] for i in range(paper_number)]
    sum_edge_per_paper = [0 for i in range(paper_number)]

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
    num_iterations = 100

    for iteration in range(num_iterations):
        new_pagerank_vector = np.zeros(paper_number)
        for i in range(paper_number):
            for j in range(paper_number):
                new_pagerank_vector[i] += damping_factor * g_matrix[j][i] * pagerank_vector[j]
            new_pagerank_vector[i] += (1 - damping_factor) / paper_number

        if np.linalg.norm(new_pagerank_vector - pagerank_vector) < epsilon:
            break

        pagerank_vector = new_pagerank_vector
    print(pagerank_vector)
    # Precision/Recall計算
    precision_results = []
    recall_results = []
    common_results = []

    c_ratios = [(1, 0), (0.75, 0.25), (0.5, 0.5), (0.25, 0.75), (0, 1)]  # c_pagerank : c_relevance の比率
    # precision_results = {}
    # recall_results = {}

    c_pagerank = 1
    c_relevance = 0
    # top_1000_papers[i][1]は距離。距離が小さいもののスコアを高くしたいので、逆数にしている。

    # PageRank vector の正規化 (合計が1になる)
    pagerank_sum = np.sum(pagerank_vector)
    normalized_pagerank_vector = pagerank_vector / pagerank_sum

    # Relevance scores (1.0 / distance) の正規化 (合計が1になる)
    relevance_scores = np.array([1.0 / top_1000_papers[i][1] for i in range(len(top_1000_papers))])
    relevance_sum = np.sum(relevance_scores)
    normalized_relevance_scores = relevance_scores / relevance_sum

    # 正規化されたスコアを用いて final_score を計算
    final_score = [
        c_pagerank * normalized_pagerank_vector[i] + c_relevance * normalized_relevance_scores[i]
        for i in range(len(top_1000_papers))
    ]

    # final_score = [c_pagerank * pagerank_vector[i] + c_relevance * (1.0 / top_1000_papers[i][1]) for i in range(paper_number)]

    for topN in topN_list:
        if topN > len(top_1000_papers):
            print(f"TopN ({topN}) exceeds number of similar papers ({len(top_1000_papers)}).")
            continue

        top_indices = np.argsort(final_score)[-topN:][::-1]
        top_papers_id = [top_1000_papers[i][0] for i in top_indices]

        # print(attention_paper_id)
        # print(top_papers_id)
        # print(attention_reference_works)
        common = 0
        for i in range(len(attention_reference_works)):
            if attention_reference_works[i] in top_papers_id:
                common += 1

        common_results.append(common)
        precision = common / topN if topN > 0 else 0
        recall = common / len(attention_reference_works) if attention_reference_works else 0

        precision_results.append(precision)
        recall_results.append(recall)

    # paper_ids に含まれる参照論文をカウント
    paper_ids = [paper_id for paper_id, _ in top_1000_papers]  
    intersection = set(attention_reference_works) & set(paper_ids)
    contained_count = len(intersection)

    print(contained_count)
    common_results.append(contained_count)

    return common_results, precision_results, recall_results


def calculate_ref_counts(attention_paper_id, attention_reference_works):
    ref_count = []
    cursor.execute('SELECT domain, field, subfield FROM PaperVectors WHERE paper_id = ?', (attention_paper_id, ))

    topics = cursor.fetchone()
    topic_names = ["domain", "field", "subfield"]

    num = 0
    for topic in topics:
        query = f"SELECT paper_id FROM PaperVectors WHERE {topic_names[num]} = ? AND paper_id = ?;"
        num += 1
        # 指定されたトピックの論文を取得
        exist_num = 0
        for i in attention_reference_works:
            cursor.execute(query, (topic, i))
            # cursor.execute(query, (i, ))
            result = cursor.fetchone()

            # 存在チェック
            is_exist = (1 if result else 0) 
            exist_num += is_exist
        ref_count.append(exist_num)

    return ref_count

# インデックスをロード
index = faiss.read_index("./faiss_index_with_all_faster.index")
print("FAISS index loaded successfully.")

precisions = []
recalls = []
ref_counts = []

flag = 0
count = 0
sum_ref = 0
while(flag < 50):
    cursor.execute('SELECT paper_id FROM PaperVectors WHERE field = ? LIMIT 1 OFFSET ?', (20, count + 2000000))
    attention_paper_id = cursor.fetchone()[0]
    count += 1

    cursor.execute('SELECT referenced_paper_id FROM ReferencedWorks WHERE paper_id = ?', (attention_paper_id,))
    attention_reference_works = [row[0] for row in cursor.fetchall()]
    if (len(attention_reference_works) != 0):
        flag += 1
        sum_ref += len(attention_reference_works)
        common, precision, recall = calculate_precision_recall(attention_paper_id, attention_reference_works)
        precisions.append(precision)
        recalls.append(recall)

        ref_counts.append(calculate_ref_counts(attention_paper_id, attention_reference_works) + common[::-1])

output_file = "./results/precision_recall_cs_20_ave.png"

print(precisions)
print(recalls)
print(ref_counts)

precision = [0 for i in range(len(precisions[0]))]
recall = [0 for i in range(len(recalls[0]))]
ref_count = [0 for i in range(len(ref_counts[0]))]
for top_n in range(len(precisions[0])):
    precision_sum = 0
    recall_sum = 0
    for paper_number in range(len(precisions)):
        precision_sum += precisions[paper_number][top_n]
        recall_sum += recalls[paper_number][top_n]

    precision[top_n] = precision_sum / len(precisions)
    recall[top_n] = recall_sum / len(recalls)

for count in range(len(ref_counts[0])):
    ref_sum = 0
    for paper_number in range(len(ref_counts)):
        ref_sum += ref_counts[paper_number][count]

    ref_count[count] = ref_sum / len(ref_counts)
ref_count = [sum_ref / len(precisions)] + ref_count
print("ave_ref:", sum_ref / len(precisions))
print("Precision:", precision)
print("Recall:", recall)
print("ref_count:", ref_count)

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
    output_file = "precision_recall_cs.png"
    plt.savefig(output_file, format='png', dpi=300)
    print(f"Graph saved as {output_file}")

    # 表示もする場合
    plt.show()

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Execution time: {elapsed_time:.2f} seconds")
else:
    print(f"No Precision/Recall calculated for paper index {attention_paper_index}.")
