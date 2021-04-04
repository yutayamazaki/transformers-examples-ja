# BERTで類似文書検索

## 環境構築

```shell
docker-compose -f docker/docker-compose.yaml up
```

## API

- 検索の実行

```shell
curl "http://localhost:8000/api/search" --get --data-urlencode "q=送料はなんぼですか"
```
