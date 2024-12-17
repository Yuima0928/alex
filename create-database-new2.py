import sqlite3
import gzip
import json
import glob
from multiprocessing import get_context
import time
import os

# .gzipファイルのパスを取得
input_files = glob.glob('../works/*/*.gz')

# JSONファイルから辞書を読み込む
with open("domain_to_number.json", "r", encoding="utf-8") as f:
    domain_to_number = json.load(f)

with open("field_to_number.json", "r", encoding="utf-8") as f:
    field_to_number = json.load(f)

with open("subfield_to_number.json", "r", encoding="utf-8") as f:
    subfield_to_number = json.load(f)

# 論文データを処理して一時データベースに保存する関数
def process_file(args):
    input_file, batch_size, temp_db_path, source_db_path = args

    device = "cpu"  # 明示的にCPUを使用
    connection = sqlite3.connect(temp_db_path)
    cursor = connection.cursor()

    sub_connection = sqlite3.connect(source_db_path)
    sub_cursor = sub_connection.cursor()

    print(f"Processing file: {input_file} on {device}")
    with gzip.open(input_file, 'rt', encoding='utf-8') as f_in:
        paper_ids, sentences, publication_years = [], [], []
        abstracts , titles = [], []
        institutions, embeddings = [], []
        domains, fields, subfields = [], [], []
        referenced_works_data = []

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
            title = ""
            if abstract_inverted_index:
                words = sorted(abstract_inverted_index.items(), key=lambda x: x[1][0])
                sentence = " ".join([word[0] for word in words])
                abstract = sentence
            else:
                sentence = record.get("title", "")
                title = sentence

            publication_year = record.get("publication_year", 0) 
            referenced_works = record.get("referenced_works", [])

            # topic取得 (指定されたロジック)
            domain = 0
            if record.get("primary_topic") and record["primary_topic"].get("domain"):
                domain = domain_to_number.get(record["primary_topic"]["domain"].get("display_name", ""), 0)

            field = 0
            if record.get("primary_topic") and record["primary_topic"].get("field"):
                field = field_to_number.get(record["primary_topic"]["field"].get("display_name", ""), 0)

            subfield = 0
            if record.get("primary_topic") and record["primary_topic"].get("subfield"):
                subfield = subfield_to_number.get(record["primary_topic"]["subfield"].get("display_name", ""), 0)

            # institution取得
            institution = ""
            if record.get("authorships"):
                first_authorship = record["authorships"][0]
                if "institutions" in first_authorship and first_authorship["institutions"]:
                    institution = first_authorship["institutions"][0].get("display_name", "")

            if sentence:
                paper_ids.append(paper_id)
                sentences.append(sentence)
                abstracts.append(abstract)
                titles.append(title)
                publication_years.append(publication_year)
                domains.append(domain)
                fields.append(field)
                subfields.append(subfield)
                institutions.append(institution)

                # 参照関係を保存
                for ref in referenced_works:
                    referenced_works_data.append((paper_id, ref))

                # データベースから既存ベクトルを取得
                sub_cursor.execute('SELECT vector FROM PaperVectors WHERE paper_id = ?', (paper_id,))
                result = sub_cursor.fetchone()
                embeddings.append(result[0] if result else None)

            # バッチ処理
            if len(sentences) >= batch_size:
                for i in range(len(embeddings)):
                    cursor.execute('''
                        INSERT OR IGNORE INTO PaperVectors (paper_id, title, abstract, vector, publication_year, domain, field, subfield, institutions)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (paper_ids[i], titles[i], abstracts[i], embeddings[i], publication_years[i], domains[i], fields[i], subfields[i], institutions[i]))

                cursor.executemany('''
                    INSERT OR IGNORE INTO ReferencedWorks (paper_id, referenced_paper_id)
                    VALUES (?, ?)
                ''', referenced_works_data)

                connection.commit()
                paper_ids, sentences, publication_years, domains, fields, subfields, institutions, embeddings = [], [], [], [], [], [], [], []
                referenced_works_data = []
                abstracts , titles = [], []

        # 残ったデータの処理
        if sentences:
            for i in range(len(embeddings)):
                cursor.execute('''
                    INSERT OR IGNORE INTO PaperVectors (paper_id, title, abstract, vector, publication_year, domain, field, subfield, institutions)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (paper_ids[i], titles[i], abstracts[i], embeddings[i], publication_years[i], domains[i], fields[i], subfields[i], institutions[i]))

            cursor.executemany('''
                INSERT OR IGNORE INTO ReferencedWorks (paper_id, referenced_paper_id)
                VALUES (?, ?)
            ''', referenced_works_data)
            connection.commit()

    sub_connection.close()
    connection.close()

# テーブル作成関数
def create_tables(db_path):
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()

    # `PaperVectors` テーブル
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS PaperVectors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            paper_id TEXT UNIQUE,
            title TEXT,
            abstract TEXT,
            vector BLOB,
            publication_year INTEGER,
            domain INTEGER,
            field INTEGER,
            subfield INTEGER,
            institutions TEXT,
            FOREIGN KEY (domain) REFERENCES Domain (domain_id),
            FOREIGN KEY (field) REFERENCES Field (field_id),
            FOREIGN KEY (subfield) REFERENCES Subfield (subfield_id)
        )
    ''')

    # `ReferencedWorks` テーブル
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS ReferencedWorks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
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
        temp_cursor.execute('SELECT paper_id, title, abstract, vector, publication_year, domain, field, subfield, institutions FROM PaperVectors')
        rows = temp_cursor.fetchall()
        if rows:
            main_cursor.executemany('''
                INSERT OR IGNORE INTO PaperVectors (paper_id, title, abstract, vector, publication_year, domain, field, subfield, institutions)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', rows)

        # `ReferencedWorks` のデータをバッチで挿入
        temp_cursor.execute('SELECT paper_id, referenced_paper_id FROM ReferencedWorks')
        ref_rows = temp_cursor.fetchall()
        if ref_rows:
            main_cursor.executemany('''
                INSERT OR IGNORE INTO ReferencedWorks (paper_id, referenced_paper_id)
                VALUES (?, ?)
            ''', ref_rows)

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
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            domain_id INTEGER UNIQUE,
            domain_text TEXT
        )
    ''')

    # Field テーブル
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Field (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            field_id INTEGER UNIQUE,
            field_text TEXT
        )
    ''')

    # Subfield テーブル
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Subfield (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            subfield_id INTEGER UNIQUE,
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

def main(input_files, batch_size=256, num_processes=18):
    if not input_files:
        print("No .gz file found at the specified path.")
        return

    main_db_path = 'paper_with_topic_indexed_new.db'
    create_tables(main_db_path)

    source_db_path = "paper_with_topic_indexed.db"
    create_index(source_db_path, 'PaperVectors', 'paper_id', 'idx_paper_id')

    # JSONファイルの読み込み
    domain_to_number = load_json("domain_to_number.json")
    field_to_number = load_json("field_to_number.json")
    subfield_to_number = load_json("subfield_to_number.json")

    # 必要なテーブルを作成してからデータを挿入
    create_domain_field_subfield_tables(main_db_path)
    insert_data_into_tables(main_db_path, domain_to_number, field_to_number, subfield_to_number)

    # Domain, Field, Subfield のインデックスを作成
    create_index(main_db_path, 'Domain', 'domain_id', 'idx_domain_id')
    create_index(main_db_path, 'Field', 'field_id', 'idx_field_id')
    create_index(main_db_path, 'Subfield', 'subfield_id', 'idx_subfield_id')

    temp_dbs = [f"temp_db_{i}.db" for i in range(len(input_files))]
    for temp_db_path in temp_dbs:
        create_tables(temp_db_path)

    start_time = time.time()

    args = [(input_files[i], batch_size, temp_dbs[i], source_db_path) for i in range(len(input_files))]

    with get_context("spawn").Pool(processes=num_processes) as pool:
        pool.map(process_file, args)

    merge_databases(temp_dbs, main_db_path)
    create_index(main_db_path, 'PaperVectors', 'domain', 'idx_domain')
    create_index(main_db_path, 'PaperVectors', 'field', 'idx_field')
    create_index(main_db_path, 'PaperVectors', 'subfield', 'idx_subfield')
    create_index(main_db_path, 'PaperVectors', 'paper_id', 'idx_paper_id')
    create_index(main_db_path, 'ReferencedWorks', 'paper_id', 'idx_referenced_paper_id')

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
    main(input_files, batch_size=1024, num_processes=18)
