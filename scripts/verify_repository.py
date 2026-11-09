#!/usr/bin/env python3
"""Memeriksa integritas arsip proyek. Tidak melatih model dan tidak mengevaluasi prediksi."""
from pathlib import Path
import json,csv,re,sys,zipfile,hashlib,xml.etree.ElementTree as ET
P=Path(__file__).resolve().parents[1]
errors=[]
def check(ok,what):
 print(('OK  ' if ok else 'FAIL')+what)
 if not ok:errors.append(what)
for name in ['README.md','notebooks/UTS_DEEP_LEARNING_KELOMPOK_4_FINAL.ipynb','reports/Laporan_UTS_Deep_Learning_Kelompok_4.pdf','reports/Laporan_UTS_Deep_Learning_Kelompok_4.docx','docs/HUGGING_FACE.md','docs/WEB_SOURCE.md','docs/DEPLOYMENT.md','docs/GALLERY.md','docs/EXPERIMENTS.md','results/recorded/MANIFEST.json','data/testing/batch_uji_18_laptop.csv','data/testing/batch_uji_40_laptop.xlsx','data/testing/batch_uji_72_laptop.json','web_static/app.js','web_static/index.html','config/frontend_provenance.json']:
 check((P/name).is_file(),'file '+name)
n= json.loads((P/'notebooks/UTS_DEEP_LEARNING_KELOMPOK_4_FINAL.ipynb').read_text('utf-8'))
check(sum(x['cell_type']=='code' for x in n['cells'])==14,'Notebook berisi 14 sel kode')
check(sum(x['cell_type']=='markdown' for x in n['cells'])==28,'Notebook berisi 28 sel Markdown')
m=json.loads((P/'results/recorded/MANIFEST.json').read_text('utf-8'))
check(hashlib.sha256((P/'notebooks/UTS_DEEP_LEARNING_KELOMPOK_4_FINAL.ipynb').read_bytes()).hexdigest()==m['notebook_sha256'],'SHA-256 notebook sama dengan sumber')
check(len(m['graphics'])==15 and len(list((P/'docs/images').glob('*.png')))==15,'15 grafik output notebook')
check(len(m['recorded_html_tables'])==30,'30 tabel output yang diekstrak')
check(len(list((P/'docs/huggingface').glob('*.png')))>=8,'Minimal 8 screenshot bukti UI website')
check(len(list((P/'docs/stages').glob('*.md')))==14,'14 halaman stage')
with (P/'data/testing/batch_uji_18_laptop.csv').open(encoding='utf-8-sig',newline='') as f:
 r=list(csv.DictReader(f)); check(len(r)==18 and all(len(x)==19 for x in r),'CSV 18 baris, 19 atribut')
j=json.loads((P/'data/testing/batch_uji_72_laptop.json').read_text('utf-8'))
check(isinstance(j,list) and len(j)==72 and all(isinstance(x,dict) and len(x)==19 for x in j),'JSON 72 baris, 19 atribut')
xp=P/'data/testing/batch_uji_40_laptop.xlsx'
try:
 with zipfile.ZipFile(xp) as z:
  names=z.namelist();sh=next(i for i in names if i=='xl/worksheets/sheet1.xml')
  root=ET.fromstring(z.read(sh));ns={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
  rows=root.findall('.//s:sheetData/s:row',ns)
  check(len(rows)==41,'XLSX 40 baris data + 1 header')
except Exception as e:check(False,'Baca XLSX '+str(e))
# README and docs local Markdown links; ignore anchors, external links and fenced code.
miss=[]
for md in [P/'README.md',*P.glob('docs/**/*.md'),*P.glob('data/**/*.md'),P/'results/README.md',P/'web_static/README.md']:
 raw=md.read_text('utf-8')
 raw=re.sub(r'```.*?```','',raw,flags=re.S)
 for ref in re.findall(r'(?<!!)\]\(([^)]+)\)|!\[[^]]*\]\(([^)]+)\)',raw):
  url=(ref[0] or ref[1]).strip().split('#',1)[0]
  if not url or url.startswith(('http:','https:','mailto:','data:')):continue
  if not (md.parent/url).exists():miss.append((str(md.relative_to(P)),url))
check(not miss,'Semua local link Markdown valid'+(' - '+repr(miss[:4]) if miss else ''))
readme=(P/'README.md').read_text('utf-8')
check(all(x in readme for x in ['82,35%','0,8256','88,24%','SDK Static','Xkazuya/LAPTOP-CLASSIFIER']), 'Ringkasan inti README konsisten')
check((P/'reports/Laporan_UTS_Deep_Learning_Kelompok_4.pdf').read_bytes()[:5]==b'%PDF-', 'PDF laporan valid header')
check(hashlib.sha256((P/'web_static/app.js').read_bytes()).hexdigest()==next(x['sha256'] for x in json.loads((P/'config/frontend_provenance.json').read_text())['files'] if x['path_in_repository']=='web_static/app.js'),'app.js sama dengan manifest unggahan')
check(hashlib.sha256((P/'web_static/index.html').read_bytes()).hexdigest()==next(x['sha256'] for x in json.loads((P/'config/frontend_provenance.json').read_text())['files'] if x['path_in_repository']=='web_static/index.html'),'index.html sama dengan manifest unggahan')
print('\n', 'SEMUA PEMERIKSAAN LULUS' if not errors else str(len(errors))+' PEMERIKSAAN GAGAL')
sys.exit(1 if errors else 0)
