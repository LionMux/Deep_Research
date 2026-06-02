#!/usr/bin/env python3
"""
Генерация отчёта через правильный RAG-пайплайн SciRAG.
Все операции в одном процессе — state сохраняется.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from scirag.mcp_server import scirag_index, scirag_search, scirag_synthesize

PAPERS_DIR = r"C:\Users\Lev\Desktop\deep_research_v3\papers"

# 1. Index (idempotent — если уже проиндексировано, быстро пропустит)
print("=" * 60)
print("STEP 1: Indexing papers...")
idx_result = json.loads(scirag_index(PAPERS_DIR))
print(json.dumps(idx_result, ensure_ascii=False, indent=2))
if "error" in idx_result:
    sys.exit(1)

# 2. Test search
print("\n" + "=" * 60)
print("STEP 2: Test search...")
search_result = json.loads(scirag_search("CNN fault location transmission line", top_k=3))
print(f"Found: {search_result.get('total_found', 0)} chunks")
for r in search_result.get('results', []):
    print(f"  [{r['tag']}] {r['document_id'][:50]}...")
    print(f"      {r['text_preview'][:150]}...")
    print()

# 3. Synthesize sections
sections = [
    ("Определение места повреждения в линиях электропередач: актуальность, виды повреждений и классические методы", 4),
    ("Машинное обучение в задаче определения места повреждения ЛЭП: обзор методов SVM, Random Forest, k-NN и нейронных сетей", 4),
    ("Сверточные нейронные сети (CNN) для обнаружения и локализации повреждений в электросетях: принципы и архитектуры", 4),
    ("Методология применения CNN для ОМП в ЛЭП: сбор данных, предобработка сигналов, вейвлет-преобразования, построение модели", 4),
    ("Обзор литературы и сравнение результатов: точность, датасеты, метрики оценки CNN-моделей для ОМП", 4),
    ("Перспективы развития: цифровые двойники, edge computing, IoT и интеграция CNN в системы релейной защиты", 3),
]

outputs = []
for i, (query, sections_count) in enumerate(sections, 1):
    print("\n" + "=" * 60)
    print(f"STEP 3.{i}: Synthesizing section: {query[:60]}...")
    result = json.loads(scirag_synthesize(query, max_sections=sections_count))
    if "error" in result:
        print(f"ERROR: {result['error']}")
        continue
    final_text = result.get("final_text", "")
    stats = result.get("stats", {})
    print(f"  Sections: {stats.get('total_sections', '?')}")
    print(f"  Facts: {stats.get('verified_facts', '?')}/{stats.get('total_facts', '?')}")
    print(f"  Time: {stats.get('total_time_sec', '?')}s")
    outputs.append({
        "query": query,
        "text": final_text,
        "stats": stats,
    })
    # Save incremental
    with open("report_sections.json", "w", encoding="utf-8") as f:
        json.dump(outputs, f, ensure_ascii=False, indent=2)

print("\n" + "=" * 60)
print("All sections synthesized!")
print("Saved to: report_sections.json")
