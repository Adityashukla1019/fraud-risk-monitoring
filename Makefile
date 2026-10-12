PY ?= python3

.PHONY: setup data db sql train memo all dashboard test clean
setup:      ; $(PY) -m pip install -r requirements.txt
data:       ; $(PY) -m src.generate_data
db:         ; $(PY) -m src.load_db
sql:        ; $(PY) -m src.run_sql_analysis
train:      ; $(PY) -m src.train
memo:       ; $(PY) -m src.make_memo
all: data db sql train memo
dashboard:  ; streamlit run dashboard/app.py
test:       ; $(PY) -m pytest -q
clean:      ; rm -rf data/*.csv data/*.db models reports/sql reports/*.csv reports/metrics.json
