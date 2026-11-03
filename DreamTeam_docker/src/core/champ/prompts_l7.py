# -*- coding: utf-8 -*-
"""L7 prompt library. Byte-identical rubric blocks lifted from L1's prompts.py so that the
model-size x quantisation grid is a CONTROLLED experiment: the only things that change between
cells are the checkpoint and its quantisation.

Two output forms are used, both LABEL-FIRST and both BARE (no rationale):
  * the label is the FIRST thing generated, so nothing in the context can talk the model into a
    position before the label token is emitted (F110), and
  * the answer prefix `{"etiket": "` is PREFILLED by the harness, so max_tokens=1 suffices and the
    read position is structurally fixed (no text parsing, F111 trap unreachable).
"""

LABEL_BLOCK = """ETİKETLER (üç tanesinden tam olarak birini seç)

- relevant : Ürün, sorgunun istediği şeyin ta kendisi ve sorgudaki her açık kısıtı
  (ürün tipi, marka, model, renk, cinsiyet, yaş grubu) karşılıyor.

- semi-relevant : Ürün sorgunun ihtiyacına KOMŞU ama tam karşılığı değil. İki kalıbı vardır ve
  BİRİNCİSİ ÇOK DAHA SIKTIR:
    (a) DOĞRU ÜRÜN AİLESİ, İSTENMEYEN VARYANT — ana ürün tipi tutuyor, ama paket/set/alt-tür/ikincil
        bir özellik alıcının istediğinden farklı.
        "tek kişilik yatak örtüsü takımı" → tek parça yatak örtüsü (takım değil);
        "sülfatsız şampuan" → şampuan + saç yağı seti; "ahşap jaluzi" → plise perde;
        "çöp kovası banyo seti" → tek bir banyo çöp kovası.
    (b) TAMAMLAYICI / AKSESUAR / PARÇA — sorgudaki ürünün kendisi değil, ona ait ya da onunla
        kullanılan ürün.
        "iphone 15" → iPhone 15 uyumlu kılıf; "akustik gitar" → gitar asma aparatı;
        "pejo 307" → Peugeot 307 kapı eşiği; "kahve makinesi" → filtre kağıdı.

- irrelevant : Sorgunun KİLİT bir niteliği İHLAL EDİLİYOR. DİKKAT: bu veri setinde irrelevant
  ürün genellikle rastgele alakasız bir ürün DEĞİLDİR. Neredeyse her zaman aynı üst kategoriden,
  sorguya kelime olarak çok yakın, ama YANLIŞ ŞEY olan bir üründür. Üç kalıbı vardır:
    (1) Doğru departman, YANLIŞ ÜRÜN TİPİ.
        "kışlık kadın bot" → kadın babet; "sırt çantası" → cüzdan; "mont" → kazak.
    (2) Doğru ürün tipi, ÇELİŞEN NİTELİK (renk, desen, kesim, materyal, stil).
        "yeşil topuklu ayakkabı" → yeşil SPOR ayakkabı (topuk yok);
        "puantiyeli gömlek" → transparan düz gömlek; "beyaz şık elbise" → çiçek desenli elbise.
    (3) YANLIŞ marka / model / cinsiyet / yaş grubu.
        "erkek kot pantolon" → kadın kot pantolon; "samsung a54 kılıfı" → iphone 13 kılıfı."""

BOUNDARY_BLOCK = """EN ZOR VE EN ÖNEMLİ KARAR semi-relevant ile irrelevant arasındadır. Ayırt edici ölçüt tek bir
soruya iner: uyuşmayan şey sorgunun AÇIKÇA YAZDIĞI bir şart mı, yoksa İKİNCİL bir ayrıntı mı?

  - Sorgunun açıkça yazdığı bir şart (ana ürün tipi, yazılı renk, yazılı desen, marka, model,
    cinsiyet, yaş grubu) ÇİĞNENİYORSA → irrelevant. Ürün ne kadar benzer görünürse görünsün.
  - Uyuşmazlık sorgunun yazmadığı ya da ikincil bir ayrıntıdaysa (paket adedi, set/tekil, alt-tür,
    ölçü, ufak stil farkı), ya da ürün sorgudakinin aksesuarı/parçası/tamamlayıcısıysa
    → semi-relevant.
  - Bütün açık şartlar tutuyor ve ürün sorgunun istediği şeyin ta kendisiyse → relevant.

ÇOĞUNLUK DURUMU — sorguların büyük kısmı yalnızca bir ÜRÜN TİPİ yazar, renk/marka/cinsiyet
belirtmez (ortalama sorgu 3 kelimeden kısadır). Böyle bir sorguda tek açık şart ürün tipidir:
  - ürün o ürün tipinin kendisiyse → relevant
  - ürün o ürün tipinin aksesuarı/parçası/tamamlayıcısı ya da istenmeyen bir alt-varyantıysa
    → semi-relevant
  - ürün BAŞKA bir ürün tipiyse (aynı reyondan olsa bile) → irrelevant

Kalibrasyon kuralı: ürün "neredeyse aradığı şey" izlenimi veriyorsa semi-relevant tarafına yaklaş.
Aynı kategoride olmak TEK BAŞINA relevant ya da semi-relevant yapmaz; ama tek başına irrelevant da
yapmaz — irrelevant için ihlal edilen AÇIK bir şart gösterebilmelisin."""

# "hard" rubric: the boundary is pushed toward irrelevant. This is the second prompt of the
# two-prompt pair (coordinator rule 1: one strict-rubric + one few-shot, probabilities averaged).
HARD_BOUNDARY_BLOCK = """EN ZOR VE EN ÖNEMLİ KARAR semi-relevant ile irrelevant arasındadır. Bu görevde
`irrelevant` SIK bir etikettir — test kümesinin tam üçte biri irrelevant'tır. Ürünün sorguya
kelime olarak benzemesi seni yanıltmasın; asıl soru şudur: alıcı bu ürünü görünce ARADIĞI ŞEYİ
bulmuş olur mu?

  - Ana ÜRÜN TİPİ farklıysa → irrelevant. Aynı reyondan olması, aynı kelimeleri taşıması,
    hatta aynı markadan olması bunu değiştirmez.
  - Sorgunun yazdığı herhangi bir açık şart (renk, desen, materyal, marka, model, cinsiyet,
    yaş grubu, beden sınıfı) çiğneniyorsa → irrelevant.
  - Yalnızca ürün tipi TUTUYOR ve uyuşmazlık ikincil bir ayrıntıdaysa (paket adedi, set/tekil,
    ölçü, alt-varyant) → semi-relevant.
  - Ürün, sorgudaki ürünün aksesuarı / parçası / tamamlayıcısıysa (kılıf, filtre, aparat, eşik)
    → semi-relevant.
  - Bütün açık şartlar tutuyor ve ürün sorgunun istediği şeyin ta kendisiyse → relevant.

Şüphedeysen ve ihlal edilen açık bir şart GÖSTEREBİLİYORSAN irrelevant seç; gösteremiyorsan
semi-relevant seç."""

OUT_BARE = """ÇIKTI BİÇİMİ — yalnızca tek satır JSON, başka hiçbir şey yazma, gerekçe YAZMA:
{"etiket": "relevant"}
etiket alanı yalnızca şu üçünden biri olabilir: relevant, semi-relevant, irrelevant"""

HEAD = ("Sen Trendyol'un arama alaka düzeyi (search relevance) uzmanısın. Bir Türkçe arama "
        "sorgusu ile bir ürünü karşılaştırır ve ürünün sorguya ne kadar uyduğunu etiketlersin.")

ANSWER_PREFIX = '{"etiket": "'


def system_bare(shots_text=None, hard=False):
    """Few-shot demonstrations go at the FRONT of the system message (before the row), so the whole
    block is one shared prefix and vLLM's prefix cache processes it exactly once for all 7,500
    rows."""
    b = HARD_BOUNDARY_BLOCK if hard else BOUNDARY_BLOCK
    parts = [HEAD, LABEL_BLOCK, b]
    if shots_text:
        parts.append("ÖRNEKLER\n\n" + shots_text)
    parts.append(OUT_BARE)
    return "\n\n".join(parts)


# --------------------------------------------------------------------------------------------
# TINY rubric. The speed axis showed wall-clock is ~linear in TOTAL prompt tokens on these
# models, so the rubric length is itself a speed dial. This keeps the three class definitions and
# the one decision rule that F22 says carries most of the errors, and drops every example.
TINY_BLOCK = """Bir Türkçe arama sorgusu ile bir ürünü karşılaştır ve etiketle:
- relevant: ürün sorgunun istediği şeyin ta kendisi, her açık şart (ürün tipi, marka, model, renk,
  cinsiyet, yaş) tutuyor.
- semi-relevant: ürün tipi tutuyor ama istenmeyen varyant (set/tekil, paket, alt-tür), YA DA ürün
  sorgudakinin aksesuarı/parçası/tamamlayıcısı.
- irrelevant: sorgunun AÇIKÇA yazdığı bir şart çiğneniyor (başka ürün tipi, yanlış renk/desen,
  yanlış marka/model/cinsiyet/yaş). Kelime olarak benzemesi kurtarmaz.
Yalnızca şu JSON'u yaz: {"etiket": "relevant"}"""


def system_tiny():
    return TINY_BLOCK


# --------------------------------------------------------------------------------------------
# SOFT rubric: the semi<->irrelevant boundary is narrowed TOWARD semi-relevant.
# Motivation is measured, not stylistic: with the fs and hard rubrics this checkpoint calls
# `semi-relevant` on only 1320-1842 of 7500 rows against a known truth of 2500, i.e. it
# systematically under-calls the class that F22 says carries 55% of all errors. This is the
# third member of the single-checkpoint prompt ensemble.
SOFT_BOUNDARY_BLOCK = """EN ZOR VE EN ÖNEMLİ KARAR semi-relevant ile irrelevant arasındadır. Bu veri setinde üç etiket de
EŞİT SIKLIKTA (her biri üçte bir). `semi-relevant` en kolay KAÇIRILAN etikettir, çünkü ürün ilk
bakışta ya "tam istenen" ya da "tamamen alakasız" görünür; oysa arada geniş bir bant vardır.

  - Ürün, sorgudaki ürünün aksesuarı / parçası / kılıfı / filtresi / aparatı / tamamlayıcısıysa
    → semi-relevant. (irrelevant DEĞİL — alıcı bunu ararken bulmuş olabilir.)
  - Ana ürün tipi tutuyor ama paket/set/tekil/adet/ölçü/alt-tür/ikincil bir stil farkı varsa
    → semi-relevant.
  - Ürün sorgunun istediği şeyin daha genel ya da daha özel bir alt-türüyse → semi-relevant.
  - `irrelevant` YALNIZCA sorgunun AÇIKÇA YAZDIĞI bir şart çiğnendiğinde: yazılı renk, yazılı
    desen, yazılı marka/model, yazılı cinsiyet/yaş, ya da bambaşka bir ürün tipi.
  - Bütün açık şartlar tutuyor ve ürün sorgunun istediği şeyin ta kendisiyse → relevant.

Kural: ihlal edilen AÇIK bir şart GÖSTEREMİYORSAN `irrelevant` SEÇME — o satır semi-relevant'tır."""


def system_soft():
    return "\n\n".join([HEAD, LABEL_BLOCK, SOFT_BOUNDARY_BLOCK, OUT_BARE])
