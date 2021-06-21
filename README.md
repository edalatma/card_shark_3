The pipeline for generating and testing models can be found in **`main.ipynb`**.
This file generates the tables and figures seen in the paper which can also be found stored in `results.tar.gz`.

## Requirements

- Pipenv
- Python 3.7.3

## Running

1. `pipenv install && pipenv shell`
2. `cd paper_download && python pull_papers.py`
3. `python -m ipykernel install --user --name=card_shark_3_paper`
4. `jupyter notebook`
5. Open `main.ipynb`, select the `kernel > change kernel` dropdown menu and select the `card_shark_3_paper` environment. You can now run through each of the cells to obtain the same results as those stored in `results.tar.gz`.

## `card_pubs.csv` - Jan 12, 2021

```sql
SELECT *
FROM pub
JOIN pub_dbxref as dbx ON pub.pub_id = dbx.pub_id
JOIN dbxref as ref ON ref.dbxref_id = dbx.dbxref_id
ORDER BY pub.pub_id DESC
```

A total of 131 papers found in `card_pubs.csv` were found in the papers obtained between 2017-2020
Thus the retrospective venn-diagram and table are relative to 131 papers.
