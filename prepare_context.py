import json

with open('papers_metadata.json', 'r', encoding='utf-8') as f:
    papers = json.load(f)

papers_sorted = sorted(papers, key=lambda p: (p.get('citations', 0), p.get('relevance', 0)), reverse=True)

top30 = papers_sorted[:30]
context = []
for i, p in enumerate(top30, 1):
    authors = ', '.join(p.get('authors', [])[:2])
    line = '[%d] %s — %s (%s). DOI: %s. Citations: %d.' % (
        i, p.get('title', ''), authors, p.get('year', 'n.d.'),
        p.get('doi', 'N/A'), p.get('citations', 0)
    )
    context.append(line)

with open('top30_context.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(context))

print('Top 30 context saved. Total unique papers:', len(papers))
for line in context[:5]:
    print(line)
