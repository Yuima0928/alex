- search-faster.py　-> 並列実行
- search-with-pagerank.py　-> 出力が丁寧
- all-index.py -> 全部のfaissのindexを保存するために必要
- create-database-new2.py -> databaseを作るために必要
    - でもこれを作るために"paper_with_topic_indexed.db"が必要。embeddingはここからとってきている。
- search-faster-input.py -> 自分の研究内容を入れられる。


- pngファイルについて(すべてn.probeは10)
    - 注目論文をmedicineにしたとき(pagerankにいれる論文は1000)
        - precision_recall_parallel_medi_attention_medi.png
            - faissのindexをmedicineのものにしたとき
        - precision_recall_parallel_all_attention_medi.png
        　　- faissのindexをallのものにしたとき
    - 注目論文をallにした時(faissのindexはallのもの)
        - precision_recall_parallel_all_attention_all.png
            - pagerankにいれる論文は1000
        - precision_recall_parallel_all_attention_all_10000.png
            - pagerankにいれる論文は10000
    

