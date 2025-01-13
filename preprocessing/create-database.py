import sqlite3
import gzip
import json
import glob
from multiprocessing import get_context
from datetime import date, timedelta
import time
import os
import torch
from itertools import chain
from sentence_transformers import SentenceTransformer

# .gzipファイルのパスを取得
input_files = glob.glob('../data/openalex-snapshot/data/works/*/*.gz')
authors_input_files = glob.glob('../data/openalex-snapshot/data/authors/*/*.gz')
institutions_input_files = glob.glob('../data/openalex-snapshot/data/institutions/*/*.gz')

# JSONファイルから辞書を読み込む
with open("domain_to_number.json", "r", encoding="utf-8") as f:
    domain_to_number = json.load(f)

with open("field_to_number.json", "r", encoding="utf-8") as f:
    field_to_number = json.load(f)

with open("subfield_to_number.json", "r", encoding="utf-8") as f:
    subfield_to_number = json.load(f)

def process_author_file(args):
    input_file, temp_db_path= args
    device = "cpu"  # 明示的にCPUを使用
    connection = sqlite3.connect(temp_db_path)
    cursor = connection.cursor()

    print(f"Processing file: {input_file} on {device}")
    with gzip.open(input_file, 'rt', encoding='utf-8') as f_in:
        author_ids, h_indexs, cited_by_counts, works_counts = [], [], [], []
        author_year_data = []

        for line in f_in:
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON in file {input_file}, skipping line. Error: {e}")
                continue

            author_id = record.get("id", "unknown")
            h_index = record["summary_stats"].get("h_index", 0)
            cited_by_count = record.get("cited_by_count", 0)
            works_count = record.get("works_count", 0)

            author_years = record.get("counts_by_year", [])
            author_years_array = []
            for author_year in author_years:
                year = author_year.get("year", 0)
                cited_by_count = author_year.get("cited_by_count", 0)
                works_count = author_year.get("works_count", 0)
                author_years_array.append([year, cited_by_count, works_count])

            author_ids.append(author_id)
            h_indexs.append(h_index)
            cited_by_counts.append(cited_by_count)
            works_counts.append(works_count)
            for author_years in author_years_array:
                author_year_data.append((author_id, author_years[0], author_years[1], author_years[2]))

        for i in range(len(author_ids)):
            cursor.execute('''
                INSERT OR IGNORE INTO Authors (author_id, h_index, cited_by_count, works_count)
                VALUES (?, ?, ?, ?)
            ''', (author_ids[i], h_indexs[i], cited_by_counts[i], works_counts[i]))
        cursor.executemany('''
            INSERT OR IGNORE INTO AuthorsYearCount (author_id, year, cited_by_count, works_count)
            VALUES (?, ?, ?, ?)
        ''', author_year_data)
        connection.commit()
    connection.close()

def process_institution_file(args):
    input_file, temp_db_path= args
    device = "cpu"  # 明示的にCPUを使用
    connection = sqlite3.connect(temp_db_path)
    cursor = connection.cursor()

    print(f"Processing file: {input_file} on {device}")
    with gzip.open(input_file, 'rt', encoding='utf-8') as f_in:
        institution_ids, cited_by_counts, works_counts = [], [], []
        institution_year_data = []

        for line in f_in:
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON in file {input_file}, skipping line. Error: {e}")
                continue

            institution_id = record.get("id", "unknown")
            cited_by_count = record.get("cited_by_count", 0)
            works_count = record.get("works_count", 0)

            institution_years = record.get("counts_by_year", [])
            institution_years_array = []
            for institution_year in institution_years:
                year = institution_year.get("year", 0)
                cited_by_count = institution_year.get("cited_by_count", 0)
                works_count = institution_year.get("works_count", 0)
                institution_years_array.append([year, cited_by_count, works_count])

            institution_ids.append(institution_id)
            cited_by_counts.append(cited_by_count)
            works_counts.append(works_count)
            for institution_years in institution_years_array:
                institution_year_data.append((institution_id, institution_years[0], institution_years[1], institution_years[2]))

        for i in range(len(institution_ids)):
            cursor.execute('''
                INSERT OR IGNORE INTO Institutions (institution_id, cited_by_count, works_count)
                VALUES (?, ?, ?)
            ''', (institution_ids[i], cited_by_counts[i], works_counts[i]))
        cursor.executemany('''
            INSERT OR IGNORE INTO InstitutionsYearCount (institution_id, year, cited_by_count, works_count)
            VALUES (?, ?, ?, ?)
        ''', institution_year_data)
        connection.commit()
    connection.close()

# 論文データを処理して一時データベースに保存する関数
def process_file(args):
    input_file, batch_size, temp_db_path, gpu_id = args
    
    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    connection = sqlite3.connect(temp_db_path)
    cursor = connection.cursor()

    print(f"Processing file: {input_file} on {device}")
    with gzip.open(input_file, 'rt', encoding='utf-8') as f_in:
        paper_ids, sentences = [], []
        abstracts , titles = [], []
        institution_ids, embeddings = [], []
        topics_array_data = []
        referenced_works_data = []
        author_ids = []
        topics_counts = []
        topic_paper_ids = []
        date_objs = []

        for line in f_in:
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON in file {input_file}, skipping line. Error: {e}")
                continue

            paper_id = record.get("id", "Unknown")
            abstract_inverted_index = record.get("abstract_inverted_index", "")

            sentence = ""
            abstract = ""
            title = record.get("title", "")
            if abstract_inverted_index:
                # すべての位置と単語を抽出し、位置情報に基づいてソート
                sorted_words = sorted(chain.from_iterable(
                    [(pos, word) for pos in positions] for word, positions in abstract_inverted_index.items()
                ), key=lambda x: x[0])

                # ソートされた単語リストから単語を連結してテキストを再構築
                abstract = " ".join(word for _, word in sorted_words)
            
            sentence = (title or "") + " " + (abstract or "")

            publication_date = record.get("publication_date", "")
            date_obj = date.fromisoformat(publication_date)

            referenced_works = record.get("referenced_works", [])
            topics_count = record.get("topics_count", "")

            topics = record.get("topics", [])
            topics_array = []
            for topic in topics:
                domain = 0
                if "domain" in topic and topic["domain"]:
                    domain = domain_to_number.get(topic["domain"].get("display_name", ""), 0)

                field = 0
                if "field" in topic and topic["field"]:
                    field = field_to_number.get(topic["field"].get("display_name", ""), 0)

                subfield = 0
                if "subfield" in topic and topic["subfield"]:
                    subfield = subfield_to_number.get(topic["subfield"].get("display_name", ""), 0)

                topics_array.append([domain, field, subfield])

            # author_id, institution_id取得
            institution_id = ""
            author_id = ""
            if record.get("authorships"): 
                first_authorship = record["authorships"][0]
                if "institutions" in first_authorship and first_authorship["institutions"]:
                    institution_id = first_authorship["institutions"][0].get("id", "")
                author_id = first_authorship["author"].get("id", "")

            if sentence:
                paper_ids.append(paper_id)
                sentences.append(sentence)
                abstracts.append(abstract)
                titles.append(title)
                topics_counts.append(topics_count)
                institution_ids.append(institution_id)
                author_ids.append(author_id)
                date_objs.append(date_obj)

                # 参照関係を保存
                for ref in referenced_works:
                    referenced_works_data.append((date_obj, paper_id, ref))

                for topics in topics_array:
                    topics_array_data.append((paper_id, topics[0], topics[1], topics[2]))

            # バッチ処理
            if len(sentences) >= batch_size:
                embeddings = model.encode(sentences, batch_size=batch_size)
                for i, embedding in enumerate(embeddings):
                    cursor.execute('''
                        INSERT OR IGNORE INTO PaperVectors (created_date, paper_id, title, abstract, vector, topics_count, author_id, institution_id)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (date_objs[i], paper_ids[i], titles[i], abstracts[i], embedding.flatten().tobytes(), topics_counts[i], author_ids[i], institution_ids[i]))

                cursor.executemany('''
                    INSERT OR IGNORE INTO ReferencedWorks (created_date, paper_id, referenced_paper_id)
                    VALUES (?, ?, ?)
                ''', referenced_works_data)

                cursor.executemany('''
                    INSERT OR IGNORE INTO Topics (paper_id, domain, field, subfield)
                    VALUES (?, ?, ?, ?)
                ''', topics_array_data)

                connection.commit()
                paper_ids, sentences, topics_counts, author_ids, institution_ids, embeddings = [], [], [], [], [], []
                topics_array_data = []
                abstracts , titles = [], []
                date_objs = []
                referenced_works_data = []

        # 残ったデータの処理
        if sentences:
            for i, embedding in enumerate(embeddings):
                embeddings = model.encode(sentences, batch_size=batch_size)
                cursor.execute('''
                    INSERT OR IGNORE INTO PaperVectors (created_date, paper_id, title, abstract, vector, topics_count, author_id, institution_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (date_objs[i], paper_ids[i], titles[i], abstracts[i], embedding.flatten().tobytes(), topics_counts[i],  author_ids[i], institution_ids[i]))

            cursor.executemany('''
                INSERT OR IGNORE INTO ReferencedWorks (created_date, paper_id, referenced_paper_id)
                VALUES (?, ?, ?)
            ''', referenced_works_data)

            cursor.executemany('''
                INSERT OR IGNORE INTO Topics (paper_id, domain, field, subfield)
                VALUES (?, ?, ?, ?)
            ''', topics_array_data)
        
            # データをリセット
            paper_ids, sentences = [], []
            topics_counts, author_ids, institution_ids, abstracts, titles = [], [], [], [], []
            referenced_works_data, topics_array_data = [], []
            date_objs = []
            embeddings = []

            connection.commit()
    connection.close()
    torch.cuda.empty_cache()  # GPUメモリをクリア

def create_authors_table(db_path):
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()
    # `Authors` テーブル
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS  Authors(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            author_id TEXT,
            h_index INTEGER,
            cited_by_count INTEGER,
            works_count INTEGER
        )
    ''')

    # `AuthorsYearCount` テーブル
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS  AuthorsYearCount(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            author_id TEXT,
            year INTEGER,
            cited_by_count INTEGER,
            works_count INTEGER,
            FOREIGN KEY(author_id) REFERENCES Authors(author_id)
        )
    ''')


    connection.commit()
    connection.close()

def create_institutions_table(db_path):
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()
    # `Institutions` テーブル
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Institutions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            institution_id TEXT,
            cited_by_count INTEGER,
            works_count INTEGER
        )
    ''')

    # `InstitutionsYearCount` テーブル
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS  InstitutionsYearCount(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            institution_id TEXT,
            year INTEGER,
            cited_by_count INTEGER,
            works_count INTEGER,
            FOREIGN KEY(institution_id) REFERENCES Institutions(institution_id)
        )
    ''')

    connection.commit()
    connection.close()


# テーブル作成関数
def create_tables(db_path):
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()

    # `PaperVectors` テーブル
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS PaperVectors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_date DATETIME,
            paper_id TEXT UNIQUE,
            title TEXT,
            abstract TEXT,
            vector BLOB,
            topics_count INTEGER,
            author_id TEXT,
            institution_id TEXT,
            FOREIGN KEY(author_id) REFERENCES Authors(author_id),
            FOREIGN KEY(institution_id) REFERENCES Institutions(institution_id)
        )
    ''')

    # `Topics` テーブル
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Topics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            paper_id TEXT,
            domain INTEGER,
            field INTEGER,
            subfield INTEGER,
            FOREIGN KEY(paper_id) REFERENCES PaperVectors(paper_id)
            FOREIGN KEY (domain) REFERENCES Domain (domain_id),
            FOREIGN KEY (field) REFERENCES Field (field_id),
            FOREIGN KEY (subfield) REFERENCES Subfield (subfield_id)
        )
    ''')

    # `ReferencedWorks` テーブル
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS ReferencedWorks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_date DATETIME,
            paper_id TEXT,
            referenced_paper_id TEXT,
            FOREIGN KEY(paper_id) REFERENCES PaperVectors(paper_id)
        )
    ''')

    connection.commit()
    connection.close()

# インデックス作成関数
def create_index(db_path, table_name, column_name, index_name):
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()
    print(f"Creating index on '{column_name}' column in '{table_name}' table...")
    cursor.execute(f"CREATE INDEX IF NOT EXISTS {index_name} ON {table_name}({column_name});")
    connection.commit()
    connection.close()

def merge_author_databases(temp_dbs, main_db_path):
    main_conn = sqlite3.connect(main_db_path)
    main_cursor = main_conn.cursor()

    # PRAGMA設定で挿入速度を向上
    main_cursor.execute("PRAGMA synchronous = OFF;")
    main_cursor.execute("PRAGMA journal_mode = MEMORY;")

    for temp_db_path in temp_dbs:
        temp_conn = sqlite3.connect(temp_db_path)
        temp_cursor = temp_conn.cursor()

        # `PaperVectors` のデータをバッチで挿入
        temp_cursor.execute('SELECT author_id, h_index, cited_by_count, works_count FROM Authors')
        rows = temp_cursor.fetchall()
        if rows:
            main_cursor.executemany('''
                INSERT OR IGNORE INTO Authors (author_id, h_index, cited_by_count, works_count)
                VALUES (?, ?, ?, ?)
            ''', rows)

        temp_cursor.execute('SELECT author_id, year, cited_by_count, works_count FROM AuthorsYearCount')
        rows = temp_cursor.fetchall()
        if rows:
            main_cursor.executemany('''
                INSERT OR IGNORE INTO AuthorsYearCount (author_id, year, cited_by_count, works_count)
                VALUES (?, ?, ?, ?)
            ''', rows)

        temp_conn.close()
        os.remove(temp_db_path)  # 一時データベースを削除

    main_conn.commit()
    main_conn.close()

def merge_institution_databases(temp_dbs, main_db_path):
    main_conn = sqlite3.connect(main_db_path)
    main_cursor = main_conn.cursor()

    # PRAGMA設定で挿入速度を向上
    main_cursor.execute("PRAGMA synchronous = OFF;")
    main_cursor.execute("PRAGMA journal_mode = MEMORY;")

    for temp_db_path in temp_dbs:
        temp_conn = sqlite3.connect(temp_db_path)
        temp_cursor = temp_conn.cursor()

        # `PaperVectors` のデータをバッチで挿入
        temp_cursor.execute('SELECT institution_id, cited_by_count, works_count FROM Institutions')
        rows = temp_cursor.fetchall()
        if rows:
            main_cursor.executemany('''
                INSERT OR IGNORE INTO Institutions (institution_id, cited_by_count, works_count)
                VALUES (?, ?, ?)
            ''', rows)

        temp_cursor.execute('SELECT institution_id, year, cited_by_count, works_count FROM InstitutionsYearCount')
        rows = temp_cursor.fetchall()
        if rows:
            main_cursor.executemany('''
                INSERT OR IGNORE INTO InstitutionsYearCount (institution_id, year, cited_by_count, works_count)
                VALUES (?, ?, ?, ?)
            ''', rows)

        temp_conn.close()
        os.remove(temp_db_path)  # 一時データベースを削除

    main_conn.commit()
    main_conn.close()

def merge_databases(temp_dbs, main_db_path):
    main_conn = sqlite3.connect(main_db_path)
    main_cursor = main_conn.cursor()

    # PRAGMA設定で挿入速度を向上
    main_cursor.execute("PRAGMA synchronous = OFF;")
    main_cursor.execute("PRAGMA journal_mode = MEMORY;")

    for temp_db_path in temp_dbs:
        temp_conn = sqlite3.connect(temp_db_path)
        temp_cursor = temp_conn.cursor()

        # `PaperVectors` のデータをバッチで挿入
        temp_cursor.execute('SELECT created_date, paper_id, title, abstract, vector, topics_count, author_id, institution_id FROM PaperVectors')
        rows = temp_cursor.fetchall()
        if rows:
            main_cursor.executemany('''
                INSERT OR IGNORE INTO PaperVectors (created_date, paper_id, title, abstract, vector, topics_count, author_id, institution_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', rows)

        # `ReferencedWorks` のデータをバッチで挿入
        temp_cursor.execute('SELECT created_date, paper_id, referenced_paper_id FROM ReferencedWorks')
        ref_rows = temp_cursor.fetchall()
        if ref_rows:
            main_cursor.executemany('''
                INSERT OR IGNORE INTO ReferencedWorks (created_date, paper_id, referenced_paper_id)
                VALUES (?, ?, ?)
            ''', ref_rows)

        # `Topics` のデータをバッチで挿入
        temp_cursor.execute('SELECT paper_id, domain, field, subfield FROM Topics')
        topic_rows = temp_cursor.fetchall()
        if topic_rows:
            main_cursor.executemany('''
                INSERT OR IGNORE INTO Topics (paper_id, domain, field, subfield)
                VALUES (?, ?, ?, ?)
            ''', topic_rows)

        temp_conn.close()
        os.remove(temp_db_path)  # 一時データベースを削除

    main_conn.commit()
    main_conn.close()

# JSONファイルの読み込み関数
def load_json(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"JSON file not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

# テーブル作成関数
def create_domain_field_subfield_tables(db_path):
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()

    # Domain テーブル
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Domain (
            domain_id INTEGER PRIMARY KEY,
            domain_text TEXT
        )
    ''')

    # Field テーブル
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Field (
            field_id INTEGER PRIMARY KEY,
            field_text TEXT
        )
    ''')

    # Subfield テーブル
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Subfield (
            subfield_id INTEGER PRIMARY KEY,
            subfield_text TEXT
        )
    ''')

    connection.commit()
    connection.close()

# データベースにデータを挿入する関数
def insert_data_into_tables(db_path, domain_data, field_data, subfield_data):
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()

    try:
        # Domain テーブルにデータを挿入
        for text, domain_id in domain_data.items():
            cursor.execute('''
                INSERT OR IGNORE INTO Domain (domain_id, domain_text) VALUES (?, ?)
            ''', (domain_id, text))

        # Field テーブルにデータを挿入
        for text, field_id in field_data.items():
            cursor.execute('''
                INSERT OR IGNORE INTO Field (field_id, field_text) VALUES (?, ?)
            ''', (field_id, text))

        # Subfield テーブルにデータを挿入
        for text, subfield_id in subfield_data.items():
            cursor.execute('''
                INSERT OR IGNORE INTO Subfield (subfield_id, subfield_text) VALUES (?, ?)
            ''', (subfield_id, text))

        connection.commit()
        print("Data inserted successfully into Domain, Field, and Subfield tables.")
    except sqlite3.Error as e:
        print(f"An error occurred: {e}")
    finally:
        connection.close()

def main(input_files, authors_input_files, institutions_input_files, batch_size=256, num_processes=16):
    if not input_files:
        print("No .gz file found at the specified path.")
        return

    main_db_path = 'paper2.db'

    # authorテーブルを作る
    create_authors_table(main_db_path)
    temp_dbs = [f"temp_db_{i}.db" for i in range(len(authors_input_files))]
    for temp_db_path in temp_dbs:
        create_authors_table(temp_db_path)

    args = [(authors_input_files[i], temp_dbs[i]) for i in range(len(authors_input_files))]

    with get_context("spawn").Pool(processes=num_processes) as pool:
        pool.map(process_author_file, args)
        pool.close()
        pool.join()
    merge_author_databases(temp_dbs, main_db_path)
    create_index(main_db_path, 'Authors', 'author_id', 'idx_author_id')
    create_index(main_db_path, 'AuthorsYearCount', 'author_id', 'idx_author_year_count_id')
    create_index(main_db_path, 'AuthorsYearCount', 'year', 'idx_author_year')


    # institutionテーブルを作る
    create_institutions_table(main_db_path)
    temp_dbs = [f"temp_db_{i}.db" for i in range(len(institutions_input_files))]
    for temp_db_path in temp_dbs:
        create_institutions_table(temp_db_path)

    args = [(institutions_input_files[i], temp_dbs[i]) for i in range(len(institutions_input_files))]

    with get_context("spawn").Pool(processes=num_processes) as pool:
        pool.map(process_institution_file, args)
        pool.close()
        pool.join()
    merge_institution_databases(temp_dbs, main_db_path)
    create_index(main_db_path, 'Institutions', 'institution_id', 'idx_institution_id')
    create_index(main_db_path, 'InstitutionsYearCount', 'institution_id', 'idx_institution_year_count_id')
    create_index(main_db_path, 'InstitutionsYearCount', 'year', 'idx_institution_year')


    # JSONファイルの読み込み
    domain_to_number = load_json("domain_to_number.json")
    field_to_number = load_json("field_to_number.json")
    subfield_to_number = load_json("subfield_to_number.json")

    # 必要なテーブルを作成してからデータを挿入
    create_domain_field_subfield_tables(main_db_path)
    insert_data_into_tables(main_db_path, domain_to_number, field_to_number, subfield_to_number)

    create_tables(main_db_path)
    temp_dbs = [f"temp_db_{i}.db" for i in range(len(input_files))]
    for temp_db_path in temp_dbs:
        create_tables(temp_db_path)

    start_time = time.time()

    # 引数を設定（GPUをラウンドロビン方式で割り当て）
    gpu_count = torch.cuda.device_count()
    args = [(input_files[i], batch_size, temp_dbs[i], i % gpu_count) for i in range(len(input_files))]

    with get_context("spawn").Pool(processes=num_processes) as pool:
        pool.map(process_file, args)
        pool.close()
        pool.join()
    merge_databases(temp_dbs, main_db_path)
    create_index(main_db_path, 'PaperVectors', 'paper_id', 'idx_paper_id')
    create_index(main_db_path, 'PaperVectors', 'created_date', 'idx_created_date')
    create_index(main_db_path, 'ReferencedWorks', 'created_date', 'idx_referenced_created_date')
    create_index(main_db_path, 'ReferencedWorks', 'paper_id', 'idx_referenced_paper_id')
    create_index(main_db_path, 'ReferencedWorks', 'referenced_paper_id', 'idx_referencedworks_referenced_paper_id') 
    create_index(main_db_path, 'Topics', 'paper_id', 'idx_topics_paper_id')

    connection = sqlite3.connect(main_db_path)
    cursor = connection.cursor()
    cursor.execute('SELECT COUNT(*) FROM PaperVectors')
    num_records = cursor.fetchone()[0]
    print(f"Total number of saved papers: {num_records}")
    connection.close()

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Execution time: {elapsed_time:.2f} seconds")

# 実行
if __name__ == "__main__":
    main(input_files, authors_input_files, institutions_input_files, batch_size=1024, num_processes=16)
