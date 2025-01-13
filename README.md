# 論文検索システム

## 概要
本システムは、論文の効率的な検索・推薦を目的とした論文検索システムです。バックエンドとフロントエンドの両方を備え、ローカル環境で動作します。また、精度検証用のコードも含まれています。

## ディレクトリ構成

```
├── README.md                # 本ファイル
├── requirements.txt                # 必要なライブラリ
├── data/                    # データセット・前処理データ
│   └── (openalex-snapshot/)                 # 生データ（OpenAlex）
├── preprocessing/           # 事前準備用コード
│   ├── create-database.py     # database作成
│   ├── create-faiss-index.py     # faissのindex作成
│   ├──create-faiss-index-quantize.py     # faiss 量子化ver
│   ├── define-topic-rank-2019.py     # 2019年時のtopic-rank作成
│   ├── define-topic-rank-2024.py     # 2024年時のtopic-rank作成
│   └── citation-train.py          # 被引用数モデルの作成と評価
├── myproject/               # プロジェクトフォルダ
│   ├── backend/             # バックエンド（API・DB）
│   │   ├── app.py           # APIサーバーのメインコード
│   │   └── database.py      # データベース管理
│   └── frontend/            # フロントエンド（UI）
│       ├── app/             # Reactアプリケーション
│       └── package.json     # フロントエンド依存ライブラリ
├── evaluation/              # 精度検証用コード
│   ├── calculate-precision-racall-f1score.py           # 類似度検索で検索してくる論文の個数を変化させる。
│   └── calculate-precision-racall-f1score_with_two_faiss.py           # 利用するfaissを変える。
└── docs/                    # ドキュメント・設計資料
    └── architecture.png     # システム構成図
```

## 環境構築

<!-- ### 1. バックエンドのセットアップ
```bash
cd myproject/backend
pip install -r requirements.txt
python app.py
```

### 2. フロントエンドのセットアップ
```bash
cd myproject/frontend
npm install
npm start
``` -->

## 実行方法
1. **事前処理**
```bash
cd data
aws s3 sync "s3://openalex" "openalex-snapshot" --no-sign-request
cd ..
pip install -r requirements.txt
python3 preprocessing/create-database.py
python3 preprocessing/create-faiss-index.py 
(python3 preprocessing/create-faiss-index-quantize.py)
python3 preprocessing/define-topic-rank-2019.py 
python3 preprocessing/define-topic-rank-2024.py 
python3 preprocessing/citation-train.py
```
2. **バックエンド起動**
```bash
python3 myproject/backend/app.py
```
3. **フロントエンド起動**
```bash
npm start --prefix myproject/frontend
```

## 精度評価
```bash
python3 evaluation/calculate-precision-racall-f1score.py 
python3 evaluation/calculate-precision-racall-f1score_with_two_faiss.py
```

## 使用技術
- **バックエンド**: Python (FastAPI, SQLite)
- **フロントエンド**: React
- **データベース**: SQLite, FAISS
- **埋め込み生成**: Sentence Transformers

## 今後の課題
- 大規模データへのスケーラビリティ対応
- 論文推薦の精度向上
- UI/UXの改善

## ライセンス
MIT License

## 開発者
- 名前: Yuima Takeshima
- メール: nattomax0928@gmail.com

