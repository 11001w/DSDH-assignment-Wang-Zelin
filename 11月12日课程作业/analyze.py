from pathlib import Path
import re,json,csv,sys
import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

P=Path(__file__).resolve().parent
titles='孟城坳 华子冈 文杏馆 斤竹岭 鹿柴 木兰柴 茱萸沜 宫槐陌 临湖亭 南垞'.split()
# 人工分词，/为诗句边界；保留正文中的地名，标题不入模。
raw={
'王维':[
'新 家 孟城口 / 古木 余 衰柳 / 来者 复 为 谁 / 空 悲 昔人 有',
'飞鸟 去 不 穷 / 连山 复 秋色 / 上下 华子冈 / 惆怅 情 何 极',
'文杏 裁 为 梁 / 香茅 结 为 宇 / 不 知 栋 里 云 / 去 作 人间 雨',
'檀栾 映 空曲 / 青翠 漾 涟漪 / 暗 入 商山路 / 樵人 不 可 知',
'空山 不 见 人 / 但 闻 人语 响 / 返景 入 深林 / 复 照 青苔 上',
'秋山 敛 余照 / 飞鸟 逐 前侣 / 彩翠 时 分明 / 夕岚 无 处所',
'结实 红 且 绿 / 复 如 花 更 开 / 山中 傥 留 客 / 置 此 芙蓉 杯',
'仄径 荫 宫槐 / 幽阴 多 绿苔 / 应门 但 迎 扫 / 畏 有 山僧 来',
'轻舸 迎 上客 / 悠悠 湖上 来 / 当轩 对 尊酒 / 四面 芙蓉 开',
'轻舟 南垞 去 / 北垞 淼 难 即 / 隔浦 望 人家 / 遥遥 不 相识'],
'裴迪':[
'结庐 古城 下 / 时 登 古城 上 / 古城 非 畴昔 / 今人 自 来往',
'落日 松风 起 / 还家 草露 晞 / 云光 侵 履迹 / 山翠 拂 人衣',
'迢迢 文杏馆 / 跻攀 日 已 屡 / 南岭 与 北湖 / 前 看 复 回顾',
'明流 纡 且 直 / 绿筱 密 复 深 / 一径 通 山路 / 行歌 望 旧岑',
'日夕 见 寒山 / 便 为 独往客 / 不 知 深林 事 / 但 有 麏麚 迹',
'苍苍 落日 时 / 鸟声 乱 溪水 / 缘 溪 路 转 深 / 幽兴 何时 已',
'飘香 乱 椒桂 / 布叶 间 檀栾 / 云日 虽 回照 / 森沈 犹 自 寒',
'门前 宫槐陌 / 是 向 欹湖 道 / 秋 来 山雨 多 / 落叶 无 人 扫',
'当轩 弥 滉漾 / 孤月 正 裴回 / 谷口 猿声 发 / 风 传 入户 来',
'孤舟 信 一 泊 / 南垞 湖水 岸 / 落日 下 崦嵫 / 清波 殊 淼漫']}
stop=set('为 复 且 如 更 此 与 已 但 便 虽 犹 自 是 之 其 而 也 乎'.split())
rows=[]
for author,poems in raw.items():
 for title,seg in zip(titles,poems):
  lines=[''.join(s.split()) for s in seg.split('/')]
  assert list(map(len,lines))==[5]*4,(author,title,lines)
  tokens=[w for w in seg.split() if w!='/' and w not in stop]
  rows.append(dict(author=author,title=title,text='，'.join(lines[:2])+'。'+'，'.join(lines[2:])+'。',segmentation=seg,tokens=tokens))
(P/'语料与分词.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
(P/'停用词.txt').write_text('\n'.join(sorted(stop)),encoding='utf-8')
v=TfidfVectorizer(tokenizer=str.split,token_pattern=None,lowercase=False,min_df=1,max_df=1.0,ngram_range=(1,1),norm='l2',use_idf=True,smooth_idf=True,sublinear_tf=False)
X=v.fit_transform([' '.join(r['tokens']) for r in rows]); terms=v.get_feature_names_out()
means=np.vstack([np.asarray(X[:10].mean(axis=0)).ravel(),np.asarray(X[10:].mean(axis=0)).ravel()])
df=np.asarray((X>0).sum(axis=0)).ravel()
def ranking(a): return sorted(range(len(terms)),key=lambda i:(-means[a,i],terms[i]))
with (P/'关键词完整表.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.writer(f);w.writerow(['词','王维平均TFIDF','裴迪平均TFIDF','王维减裴迪','文档频数'])
 for i in ranking(0):w.writerow([terms[i],*means[:,i],means[0,i]-means[1,i],df[i]])
plt.rcParams['font.sans-serif']=['Microsoft YaHei'];plt.rcParams['axes.unicode_minus']=False
fig,axes=plt.subplots(1,2,figsize=(12,6),sharex=True)
for a,ax in enumerate(axes):
 ids=ranking(a)[:12][::-1]; ax.barh([terms[i] for i in ids],means[a,ids],color=['#397ca5','#ba754b'][a]);ax.set_title(list(raw)[a]);ax.set_xlabel('每首诗 L2 归一化后，作者内平均 TF-IDF');ax.grid(axis='x',alpha=.2)
fig.suptitle('《辋川集》前10首同题诗：高权重词（各200字）');fig.tight_layout();fig.savefig(P/'关键词对照图.png',dpi=180);plt.close(fig)
(P/'运行环境.txt').write_text(f'Python {sys.version}\nscikit-learn {sklearn.__version__}\nnumpy {np.__version__}\nmatplotlib {matplotlib.__version__}\n',encoding='utf-8')
for a in range(2):
 print(list(raw)[a],[(terms[i],round(means[a,i],4),int(df[i])) for i in ranking(a)[:12]])
print('validated:',len(rows),'poems; vocabulary:',len(terms))
