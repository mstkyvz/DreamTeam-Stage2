# -*- coding: utf-8 -*-
"""L3 prompt library -- Gemma-3 (a DIFFERENT model family from L1's Qwen).

Everything here is written against measurements, not intuition:

  * 40,000 gold rows: only **19.5%** of gold `semi-relevant` is an accessory, and **45%** of it
    still contains the query's head noun.  So the dominant `semi` pattern is
    "right product family, unwanted variant" -- the rubric leads with that and demotes the
    accessory pattern to second place.  (L1 measured this; building the rubric around
    accessories, which is the intuitive move, describes a fifth of the class.)
  * L1's 4B judge predicted `irrelevant` 4,679 times against a true 2,500 (1.9x over) and its
    false-`irrelevant` rate on 600 gold rel/semi rows was 41.2%.  `irrelevant` is therefore
    written as a NARROW label that has to be *earned* by pointing at a constraint the query
    spells out, and the tie-break goes to `semi-relevant`.
  * train.parquet contains NO gold `irrelevant` at all (250k relevant / 250k semi-relevant), so
    the few-shot `irrelevant` demonstrations have to be authored.  The rel/semi shots are real
    train rows; test queries overlap train queries 0%, so nothing leaks.
  * The rationale is emitted BEFORE the label in one generation, so the sentence actually
    conditions the label instead of narrating a decision already made.

Style note: this is deliberately phrased as a decision PROCEDURE (ordered questions with a
default answer), not as a list of criteria.  L1's Qwen prompts use the criteria framing; keeping
the two families' prompts structurally different is part of what makes them worth blending.
"""

ROL = (
    "Sen Trendyol'da arama sonuçlarının kalitesini denetleyen bir uzmansın. Sana bir Türkçe arama "
    "sorgusu ve o sorguya karşılık gösterilmiş TEK bir ürün verilir. Görevin, bu ürünü arayan "
    "kişinin ürünü görünce ne hissedeceğine karar vermek: aradığı şeyin ta kendisi mi, işine "
    "yarayabilecek ama tam istediği şey olmayan bir şey mi, yoksa arama sonucunda hiç görmek "
    "istemeyeceği yanlış bir ürün mü?"
)

ETIKETLER = """ÜÇ ETİKET

1) relevant — Ürün, sorgunun istediği şeyin ta kendisi. Sorguda açıkça yazan her şart
   (ürün tipi, marka, model, renk, cinsiyet, yaş grubu) üründe de tutuyor. Sorgunun yazmadığı
   ayrıntıların ne olduğu önemli değildir; sorgu belirtmediyse şart değildir.

2) semi-relevant — Ürün aynı ihtiyaca KISMEN hizmet ediyor ama sorgunun bir şartını tam
   karşılamıyor. İki kalıbı vardır; birincisi çok daha yaygındır:

   (A) DOĞRU ÜRÜN AİLESİ, İSTENMEYEN VARYANT — ana ürün tipi tutuyor, ürünün adında genellikle
       sorgunun ana kelimesi de geçiyor, ama alıcının istediği varyant değil: takım istenirken
       tek parça, set istenirken tekil ürün, bir alt-tür yerine yakın akrabası, farklı bir
       paket/ölçü/form.
         "tek kişilik yatak örtüsü takımı" → tek parça yatak örtüsü (takım değil)
         "sülfatsız şampuan" → sülfatsız şampuan + saç yağı ikili seti
         "ahşap jaluzi perde" → plise perde
         "kablosuz kulaklık" → kablolu kulaklık
   (B) TAMAMLAYICI / AKSESUAR / YEDEK PARÇA — sorgudaki ürünün kendisi değil, ona ait ya da
       onunla birlikte kullanılan ürün. Bu kalıp (A)'dan belirgin biçimde daha SEYREKTİR.
         "iphone 15" → iPhone 15 uyumlu kılıf
         "pejo 307" → Peugeot 307 kapı eşiği
         "kahve makinesi" → filtre kağıdı

3) irrelevant — Ürün, sorgunun AÇIKÇA YAZDIĞI bir şartı İHLAL EDİYOR. Bu veri setinde
   irrelevant ürün rastgele alakasız bir ürün DEĞİLDİR: neredeyse her zaman aynı üst
   kategoriden, sorguya kelime olarak yakın, ama yanlış olan bir üründür. İhlal şu dört
   alandan birinde olur:
     - ÜRÜN TİPİ: ürün başka bir ihtiyaca hizmet ediyor. "mont" → kazak; "sırt çantası" → cüzdan
     - RENK/DESEN: sorgu rengi yazmış, ürün başka renk. "yeşil elbise" → kırmızı elbise
     - MARKA/MODEL: "samsung a54 kılıfı" → iphone 13 kılıfı
     - CİNSİYET/YAŞ: "erkek kot pantolon" → kadın kot pantolon"""

KARAR = """KARAR SIRASI — sırayla sor, ilk "evet"te dur

S1. Sorgu hangi şartları AÇIKÇA yazıyor? Sorguların çoğu üç kelimeden kısadır ve yalnızca bir
    ÜRÜN TİPİ söyler. Sorguda yazmayan hiçbir şey şart değildir — sorgu renk demediyse rengin
    farklı olması ihlal sayılmaz, sorgu adet demediyse paket adedi ihlal sayılmaz.

S2. Bu açık şartlardan biri üründe DOĞRUDAN ÇELİŞİYOR mu? (istenen renk yerine başka renk,
    istenen marka yerine başka marka, istenen cinsiyet yerine öteki cinsiyet, istenen ürün
    tipi yerine başka bir ihtiyaca hizmet eden ürün tipi)
      → EVET ise: irrelevant. Gerekçende hangi şartın neyle çeliştiğini SÖYLEYEBİLMELİSİN.
      → Söyleyemiyorsan cevap irrelevant DEĞİLDİR.

S3. Ürün sorgunun istediği ana ürün tipi ve bütün açık şartlar tutuyor mu?
      → EVET ise: relevant.

S4. Geriye kalan her şey: semi-relevant.
    Yani ürün doğru aileden ama istenmeyen varyantsa, ya da sorgudaki ürünün
    aksesuarı/parçası/tamamlayıcısıysa, ya da "neredeyse aradığı şey" izlenimi veriyorsa
    → semi-relevant."""

KALIBRASYON = """KALİBRASYON — bunlar sık yapılan hataların düzeltmesidir

- irrelevant DAR bir etikettir ve hak edilmesi gerekir. Ürünün "tam olarak aranan şey
  olmaması" irrelevant için YETMEZ; sorgunun yazdığı bir şartın çiğnendiğini gösterebilmelisin.
  Gösteremiyorsan semi-relevant yaz.
- Aksesuar, yedek parça, uyumlu ürün ve tamamlayıcı ürünler irrelevant DEĞİLDİR; bunlar
  semi-relevant'tır.
- Aynı ürün tipinin farklı varyantı (takım/tekil, set/tek parça, farklı ölçü, farklı paket,
  yakın akraba alt-tür) irrelevant DEĞİLDİR; semi-relevant'tır.
- Sorgu yalnızca bir marka adıysa, o markanın herhangi bir ürünü irrelevant değildir.
- Buna karşılık aynı kategoride olmak tek başına relevant yapmaz: "yeşil topuklu ayakkabı"
  sorgusuna gelen yeşil spor ayakkabı, topuk şartını çiğnediği için irrelevant'tır.
- KARARSIZ KALIRSAN semi-relevant seç. Bu görevin varsayılan etiketi semi-relevant'tır;
  relevant ve irrelevant somut bir gerekçe ister."""

# ---------------------------------------------------------------------------------------------
# OUTPUT FIELD ORDER.  L1 measured this on Qwen and it moved more than any rewrite of the rubric
# text: same model, same rubric, only the JSON key order swapped.
#
#                                  gerekce-first        etiket-first
#   raw irrelevant (true 2500)     4679  (1.9x)         2707  (1.08x)
#   parse failures                 64/7500              0/7500
#   degenerate one-hot rows        14.6%                0.0%
#   mean max-prob                  0.968                0.724
#
# Mechanism: writing the sentence first makes the model COMMIT, and by the time the label token
# arrives it is defending a position it already argued for -- 0.97 confidence that is an artefact
# of the sentence, not of the evidence.  Deciding first reports the real uncertainty.
# Softening the rubric TEXT moved 4679 -> 4505 (3.7% of the mass); flipping the field order moved
# it to 2707.  So: label-first is the default here, and `reason` is the one variant kept for the
# cross-family replication test.
CIKTI_LABEL_FIRST = """ÇIKTI — yalnızca tek satır JSON yaz, öncesinde ve sonrasında hiçbir şey yazma.
ÖNCE etiketi ver, SONRA o etiketi tek cümlelik Türkçe gerekçeyle açıkla:
{"etiket": "semi-relevant", "gerekce": "<tek cümle>"}
"etiket" alanı yalnızca şu üçünden biri olabilir: relevant, semi-relevant, irrelevant"""

CIKTI_REASON_FIRST = """ÇIKTI — yalnızca tek satır JSON yaz, öncesinde ve sonrasında hiçbir şey yazma.
Önce tek cümlelik Türkçe gerekçeyi, SONRA etiketi ver:
{"gerekce": "<tek cümle>", "etiket": "semi-relevant"}
"etiket" alanı yalnızca şu üçünden biri olabilir: relevant, semi-relevant, irrelevant"""

ORNEK_BASLIK = """ÖRNEKLER — aşağıdaki çözülmüş örnekleri ölçüt olarak kullan"""


def out_block(order):
    return CIKTI_REASON_FIRST if order == 'reason' else CIKTI_LABEL_FIRST


def system_zs(order='label'):
    return "\n\n".join([ROL, ETIKETLER, KARAR, KALIBRASYON, out_block(order)])


def system_fs(shots_text, order='label'):
    return "\n\n".join([ROL, ETIKETLER, KARAR, KALIBRASYON,
                        ORNEK_BASLIK + "\n\n" + shots_text, out_block(order)])


# ---------------------------------------------------------------------------------------------
# Few-shot demonstrations.
#
# `relevant` / `semi-relevant` shots are REAL rows lifted verbatim from train.parquet.  Test
# queries overlap train queries 0/5853, so this cannot leak.  `irrelevant` shots are authored in
# the identical field format because train.parquet has no gold `irrelevant` row at all.
#
# Composition is deliberate: 3 of the 4 `semi` shots are pattern (A) "right family, wrong
# variant" and only 1 is pattern (B) accessory, matching the measured 80/20 split of the real
# class.  The `yarım kask` shot is the single most useful one in the block -- gold calls it
# `semi-relevant` even though the query literally writes "yarım" and the product is a full-face
# helmet, which is exactly the case a rubric-following model wants to call `irrelevant`.
SHOTS_RELEVANT = [
    dict(query="darbeli matkap",
         title="dcd776s2 18v 1.5 ah li-ion şarjlı darbeli matkap",
         category="bahçe & elektrikli el aletleri/elektrikli el aletleri/matkap",
         brand="dewalt", gender="", age_group="",
         attributes="matkap türü: darbeli, güç (watt): 18 w, akü volt: 18 volt, mandren tipi: anahtarsız, garanti süresi: 2 yıl",
         gerekce="Ürün bir matkap ve matkap türü doğrudan darbeli olarak yazılmış, yani sorgunun istediği şeyin ta kendisi.",
         etiket="relevant"),
    dict(query="yağmur botu",
         title="su geçirmez rugan kadin yağmur bot-çizme",
         category="ayakkabı/bot & çizme/bot & bootie", brand="florin",
         gender="kadın", age_group="yetişkin",
         attributes="desen: düz, kullanım alanı: günlük, saya materyali: suni deri, bağlama şekli: bağcıklı",
         gerekce="Ürünün adı doğrudan su geçirmez yağmur botu ve sorgu cinsiyet ya da başka bir şart yazmadığı için kadın modeli olması bir ihlal değil.",
         etiket="relevant"),
    dict(query="dolgu topuklu ayakkabı",
         title="kadın zincir detaylı çapraz dolgu topuk sandalet",
         category="ayakkabı/topuklu ayakkabı/dolgu topuklu ayakkabı", brand="lanina",
         gender="kadın", age_group="yetişkin",
         attributes="topuk boyu: orta topuklu (5-9 cm), saya materyali: suni deri, iç taban materyali: suni deri",
         gerekce="Ürün dolgu topuklu ayakkabı kategorisinde ve dolgu topuğa sahip, sandalet formunda olması sorgunun yazdığı bir şartı çiğnemiyor.",
         etiket="relevant"),
]

SHOTS_SEMI = [
    dict(query="çamaşır deterjanı",
         title="mr. oxy parfümüm konsatre çamaşır yumuşatıcısı liberte 750 ml",
         category="süpermarket/ev bakım ve temizlik/çamaşır yıkama/çamaşır yumuşatıcısı",
         brand="mr.oxy", gender="", age_group="",
         attributes="hacim: 750 ml, menşei: tr",
         gerekce="Ürün aynı çamaşır yıkama ailesinden ve aynı rafta duruyor ama deterjanın kendisi değil, onunla birlikte kullanılan yumuşatıcı.",
         etiket="semi-relevant"),
    dict(query="yarım kask",
         title="thunderbolt mat siyah- beyaz kask",
         category="otomobil & motosiklet/motosiklet/motosiklet koruma ekipmanları/motosiklet kaskı",
         brand="endro", gender="", age_group="",
         attributes="kask tipi: kapalı kasklar, renk: siyah, menşei: cn",
         gerekce="Ürün doğru ürün tipi olan motosiklet kaskı ama kask tipi kapalı, yani alıcının istediği yarım kask varyantı değil.",
         etiket="semi-relevant"),
    dict(query="yemek takımı",
         title="30 parça titanyum gold çatal kaşık takımı ( bıçaksız)",
         category="ev & mobilya/ev/sofra & mutfak/sofra/çatal & bıçak & kaşık",
         brand="genel markalar", gender="", age_group="",
         attributes="materyal: çelik, parça sayısı: 30, renk: altın, kişi sayısı: 6 kişilik, menşei: tr",
         gerekce="Ürün sofra takımı ailesinden bir takım ama yemek takımı yerine bıçaksız çatal kaşık takımı, yani istenen varyantın eksik bir hâli.",
         etiket="semi-relevant"),
    dict(query="ps 5",
         title="ps5 2'li dualsense kol standı joystick tutucu ps5 yazılı plastik",
         category="elektronik/oyun & oyun konsolları/playstation konsolları/playstation aksesuarı",
         brand="genel markalar", gender="", age_group="",
         attributes="renk: siyah, ürün türü: joystick, uyumlu cihaz: playstation 5",
         gerekce="Ürün PlayStation 5 konsolunun kendisi değil, onunla kullanılan bir kol standı aksesuarı.",
         etiket="semi-relevant"),
]

# Authored, in the same field format -- train.parquet has 0 gold `irrelevant` rows.
# One shot per violation field: ürün tipi, renk, marka/model.
SHOTS_IRRELEVANT = [
    dict(query="kadın kışlık mont",
         title="kadın uzun kollu düğmeli triko hırka",
         category="giyim/dış giyim/ceket & yelek/hırka", brand="modaline",
         gender="kadın", age_group="yetişkin",
         attributes="materyal: triko, renk: antrasit, kalıp: regular, sezon: sonbahar - kış, menşei: tr",
         gerekce="Cinsiyet ve mevsim tutuyor ama sorgunun istediği dış giyim parçası mont iken ürün bir triko hırka, yani ürün tipi şartı çiğneniyor.",
         etiket="irrelevant"),
    dict(query="lacivert erkek sweatshirt",
         title="erkek bordo baskılı kapüşonlu sweatshirt",
         category="giyim/üst giyim/sweatshirt", brand="tekno stil",
         gender="erkek", age_group="yetişkin",
         attributes="renk: bordo, kalıp: oversize, yaka tipi: kapüşonlu, materyal: pamuklu, menşei: tr",
         gerekce="Ürün tipi ve cinsiyet doğru ama sorgu lacivert isterken ürünün rengi bordo, sorgunun açıkça yazdığı renk şartı ihlal ediliyor.",
         etiket="irrelevant"),
    dict(query="redmi note 13 kılıf",
         title="samsung galaxy a15 uyumlu kamera korumalı silikon kılıf",
         category="elektronik/elektronik aksesuarlar/cep telefonu aksesuarları/kapak & kılıf",
         brand="ceppon", gender="", age_group="",
         attributes="materyal: silikon, cep telefonu modeli: galaxy a15, uyumlu marka: samsung uyumlu, renk: şeffaf",
         gerekce="Ürün doğru tipte bir telefon kılıfı ama sorgudaki Redmi Note 13 yerine Samsung Galaxy A15 ile uyumlu, marka ve model şartı ihlal ediliyor.",
         etiket="irrelevant"),
]

# interleaved so no class sits in a run (a run teaches position, not the rule)
SHOT_ORDER = [('R', 0), ('S', 0), ('I', 0), ('S', 1), ('R', 1), ('I', 1),
              ('S', 2), ('R', 2), ('I', 2), ('S', 3)]
_BANK = {'R': SHOTS_RELEVANT, 'S': SHOTS_SEMI, 'I': SHOTS_IRRELEVANT}


def shot_blocks():
    return [_BANK[k][i] for k, i in SHOT_ORDER]


SIRADAKI = (
    u"ŞİMDİ SIRADAKİ ÜRÜNÜ ETİKETLE. "
    u"Yalnızca bu ürün için tek satır JSON yaz, başka hiçbir şey yazma."
)


# ---------------------------------------------------------------------------------------------
# Third framing: the SHOPPER, not the auditor.  Same three classes and the same measured
# calibration facts, but the model is asked what the person who typed the query would say on
# seeing the product, instead of being handed a decision procedure.  The point is not that one
# framing is better -- it is that two framings of the same rubric make different mistakes, and
# a blend only pays for errors that are not already in the pool.
ROL_ALICI = (
    u"Sen Trendyol'da alışveriş yapan kişisin. Arama kutusuna bir şey yazdın ve karşına bir "
    u"ürün çıktı. Bu ürünü görünce ne diyeceğine karar ver. Ürün açıklamasını okurken kendi "
    u"aradığın şeyi aklında tut: hangi kelimeleri yazdıysan senin için önemli olan onlardır, "
    u"yazmadığın hiçbir ayrıntı senin için şart değildir."
)

UC_TEPKI = u"""ÜÇ OLASI TEPKİN VAR

- "Aradığım şey tam olarak buydu." -> relevant
  Ürün, yazdığın kelimelerin karşılığı olan şeyin kendisi ve yazdığın hiçbir şarta ters düşmüyor.

- "Bu tam istediğim değil ama aynı işe bakıyor / işime yarayabilir." -> semi-relevant
  En sık iki durumda böyle dersin:
    * ürün doğru şeyin farklı bir versiyonu (takım isterken tekil, set isterken tek parça,
      bir alt-türü yerine yakın akrabası, farklı paket/ölçü/form) -- bu durum çok yaygındır;
    * ürün aradığın şeyin kendisi değil ama ona ait: kılıfı, yedek parçası, aparatı,
      onunla birlikte kullanılan ürün -- bu durum daha seyrektir.

- "Hayır, bu yanlış ürün, ben bunu istemedim." -> irrelevant
  Bunu ancak yazdığın bir şeyin ürüne UYMADIĞINI gösterebiliyorsan dersin: yazdığın renk
  tutmuyordur, yazdığın marka/model tutmuyordur, yazdığın cinsiyet ya da yaş grubu tutmuyordur,
  ya da ürün senin aradığından bambaşka bir ihtiyaca hizmet eden bir üründür."""

ALICI_KALIBRASYON = u"""KENDİNİ TUT — "bu yanlış ürün" demek için sebep göstermek zorundasın

Ürünün aradığın şeyin tam karşılığı olmaması, "yanlış ürün" demen için yetmez. Sorguna
bakıp ihlal edilen kelimeyi gösteremiyorsan tepkin "tam istediğim değil ama aynı işe bakıyor"
olmalıdır. Özellikle:
  - aradığın ürünün kılıfı/parçası/aparatı yanlış ürün değildir;
  - aynı şeyin farklı varyantı (takım/tekil, set/tek parça, farklı ölçü, farklı paket, yakın
    akraba bir alt-tür) yanlış ürün değildir;
  - sorguna yazmadığın bir ayrıntının farklı çıkması (renk yazmadıysan renk, adet yazmadıysan
    adet) yanlış ürün değildir;
  - sorgun sadece bir marka adıysa o markanın her ürünü işine yarayabilir.
Buna karşılık aynı reyondan olmak tek başına "aradığım buydu" dedirtmez: yazdığın bir şart
tutmuyorsa, ürün ne kadar benzer görünürse görünsün yanlış üründür.
EMİN OLAMIYORSAN "tam istediğim değil ama aynı işe bakıyor" (semi-relevant) de."""

CIKTI_ALICI_LABEL_FIRST = u"""ÇIKTI -- yalnızca tek satır JSON yaz, öncesinde ve sonrasında hiçbir şey yazma.
ÖNCE tepkine karşılık gelen etiketi ver, SONRA tek cümlelik Türkçe gerekçeni yaz:
{"etiket": "semi-relevant", "gerekce": "<tek cümle>"}
"etiket" alanı yalnızca şu üçünden biri olabilir: relevant, semi-relevant, irrelevant"""

CIKTI_ALICI_REASON_FIRST = u"""ÇIKTI -- yalnızca tek satır JSON yaz, öncesinde ve sonrasında hiçbir şey yazma.
Önce tek cümlelik Türkçe gerekçeni, SONRA etiketini ver:
{"gerekce": "<tek cümle>", "etiket": "semi-relevant"}
"etiket" alanı yalnızca şu üçünden biri olabilir: relevant, semi-relevant, irrelevant"""


def system_sh(order='label'):
    out = CIKTI_ALICI_REASON_FIRST if order == 'reason' else CIKTI_ALICI_LABEL_FIRST
    return u"\n\n".join([ROL_ALICI, UC_TEPKI, ALICI_KALIBRASYON, out])
