import sqlite3
import gzip
import json
import glob
from multiprocessing import get_context
import time
import os

# .gzipファイルのパスを取得
input_files = glob.glob('../works/*/*.gz')

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
            if abstract_inverted_index:
                words = sorted(abstract_inverted_index.items(), key=lambda x: x[1][0])
                sentence = " ".join([word[0] for word in words])
            else:
                sentence = record.get("title", "")

            publication_year = record.get("publication_year", "")
            referenced_works = record.get("referenced_works", [])

            # topic取得 (指定されたロジック)
            domain = ""
            if record.get("primary_topic") and record["primary_topic"].get("domain"):
                domain = record["primary_topic"]["domain"].get("display_name", "")

            field = ""
            if record.get("primary_topic") and record["primary_topic"].get("field"):
                field = record["primary_topic"]["field"].get("display_name", "")

            subfield = ""
            if record.get("primary_topic") and record["primary_topic"].get("subfield"):
                subfield = record["primary_topic"]["subfield"].get("display_name", "")

            # institution取得
            institution = ""
            if record.get("authorships"):
                first_authorship = record["authorships"][0]
                if "institutions" in first_authorship and first_authorship["institutions"]:
                    institution = first_authorship["institutions"][0].get("display_name", "")

            if sentence:
                paper_ids.append(paper_id)
                sentences.append(sentence)
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
                        INSERT OR IGNORE INTO PaperVectors (paper_id, vector, publication_year, domain, field, subfield, institutions)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (paper_ids[i], embeddings[i], publication_years[i], domains[i], fields[i], subfields[i], institutions[i]))

                cursor.executemany('''
                    INSERT OR IGNORE INTO ReferencedWorks (paper_id, referenced_paper_id)
                    VALUES (?, ?)
                ''', referenced_works_data)

                connection.commit()
                paper_ids, sentences, publication_years, domains, fields, subfields, institutions, embeddings = [], [], [], [], [], [], [], []
                referenced_works_data = []

        # 残ったデータの処理
        if sentences:
            for i in range(len(embeddings)):
                cursor.execute('''
                    INSERT OR IGNORE INTO PaperVectors (paper_id, vector, publication_year, domain, field, subfield, institutions)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (paper_ids[i], embeddings[i], publication_years[i], domains[i], fields[i], subfields[i], institutions[i]))

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
            vector BLOB,
            publication_year TEXT,
            domain TEXT,
            field TEXT,
            subfield TEXT,
            institutions TEXT
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
        temp_cursor.execute('SELECT paper_id, vector, publication_year, domain, field, subfield, institutions FROM PaperVectors')
        rows = temp_cursor.fetchall()
        if rows:
            main_cursor.executemany('''
                INSERT OR IGNORE INTO PaperVectors (paper_id, vector, publication_year, domain, field, subfield, institutions)
                VALUES (?, ?, ?, ?, ?, ?, ?)
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

# マルチプロセス処理を行うメイン関数
def main(input_files, batch_size=256, num_processes=18):
    if not input_files:
        print("No .gz file found at the specified path.")
        return

    main_db_path = 'paper_with_topic_indexed.db'
    create_tables(main_db_path)

    source_db_path = "./paper.db"
    create_index(source_db_path, 'PaperVectors', 'paper_id', 'idx_paper_id')

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
